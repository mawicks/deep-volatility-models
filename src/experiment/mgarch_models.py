# Standard Python
import logging

import os
from typing import Callable, Dict, Iterable, Iterator, Union, Tuple

from enum import Enum

# Common packages
import torch

DEBUG = False
PROGRESS_ITERATIONS = 20

DECAY_FOR_INITIALIZATION = 0.30
LEARNING_RATE = 0.25

MAX_CLAMP = 1e10
MIN_CLAMP = -MAX_CLAMP

DEFAULT_SEED = 42

normal_distribution = torch.distributions.normal.Normal(
    loc=torch.tensor(0.0), scale=torch.tensor(1.0)
)


class Parameterization(Enum):
    SCALAR = "scalar"
    DIAGONAL = "diagonal"
    TRIANGULAR = "triangular"
    FULL = "full"


def make_diagonal_nonnegative(m):
    """Given a single lower triangular matrices m, return an `equivalent` matrix
    having non-negative diagonal entries.  Here `equivalent` means m@m.T is unchanged.
    """
    diag = torch.diag(m)
    diag_signs = torch.ones(diag.shape)
    diag_signs[diag < 0.0] = -1
    return m * diag_signs


def random_lower_triangular(n, scale=1.0, requires_grad=True, device=None):
    scale = torch.tensor(scale, device=device)
    m = torch.rand(n, n, device=device) - 0.5
    m = scale * make_diagonal_nonnegative(torch.tril(m))
    m.requires_grad = requires_grad
    return m


def marginal_conditional_log_likelihood(observations, sigma, distribution):
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       sigma: torch.Tensor of shape (n_obs, n_symbols)
           Contains the estimated univariate (marginal) standard deviation for each observation.

       distribution: torch.distributions.distribution.Distribution instance which should have a log_prob() method.
           Note we assume distrubution was constructed with center=0 and shape=1.  Any normalizing and recentering
           is achieved by explicit`transformations` here.

    Returns the mean log_likelihood"""

    if sigma is not None:
        e = observations / sigma

    logging.debug(f"e: \n{e}")

    # Compute the log likelihoods on the innovations
    ll = distribution.log_prob(e) - torch.log(sigma)

    # For consistency with the multivariate case, we *sum* over the
    # variables (columns) first and *average* over the rows (observations).
    # Summing over the variables is equivalent to multiplying the
    # marginal distributions to get a join distribution.

    return torch.mean(torch.sum(ll, dim=1))


def joint_conditional_log_likelihood(
    observations, transformations, distribution, sigma=None
):
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       transformations: torch.Tensor of shape (n_symbols, n_symbols)
           transformation is a lower-triangular matrix and the outcome is presumed
           to be z = transformation @ e where the elements of e are iid from distrbution.
       distribution: torch.distributions.distribution.Distribution instance which should have a log_prob() method.
           Note we assume distrubution was constructed with center=0 and shape=1.  Any normalizing and recentering
           is achieved by explicit`transformations` here.
       sigma: torch.Tensor of shape (n_obj, n_symbols) or None.
              When sigma is specified, the observed variables
              are related to the innovations by x = diag(sigma) T e.
              This is when the estimator for the transformation
              from e to x is factored into a diagonal scaling
              matrix (sigma) and a correlating transformation T.
              When sigma is not specified any scaling is already embedded in T.


    Returns:
       torch.Tensor - log_likelihood"""

    # Divide by the determinant by subtracting its log to get the the log
    # likelihood of the observations.  The `transformations` are lower-triangular, so
    # only the diagonal entries need to be used in the determinant calculation.
    # This should be faster than calling log_det().

    log_det = torch.log(torch.abs(torch.diagonal(transformations, dim1=1, dim2=2)))

    if sigma is not None:
        log_det = log_det + torch.log(torch.abs(sigma))
        observations = observations / sigma

    # First get the innovations sequence by forming transformation^(-1)*observations
    # The unsqueeze is necessary because the matmul is on the 'batch'
    # of observations.  The shape of `t` is (n_obs, n, n) while
    # the shape of `observations` is (n_obs, n). Without adding the
    # extract dimension to observations, the solver won't see conforming dimensions.
    # We remove the ambiguity, by making observations` have shape (n_obj, n, 1), then
    # we remove the extra dimension from e.
    e = torch.linalg.solve_triangular(
        transformations,
        observations.unsqueeze(2),
        upper=False,
    ).squeeze(2)

    logging.debug(f"e: \n{e}")

    # Compute the log likelihoods on the innovations
    log_pdf = distribution.log_prob(e)

    ll = torch.sum(log_pdf - log_det, dim=1)

    return torch.mean(ll)


def optimize(optim, closure):
    best_loss = float("inf")
    done = False
    logging.info("Starting optimization")
    while not done:
        optim.step(closure)
        current_loss = closure()
        logging.info(
            f"\tcurrent loss: {current_loss:.4f}   previous best loss: {best_loss:.4f}"
        )
        if float(current_loss) < best_loss:
            best_loss = current_loss
        else:
            logging.info("Finished")
            done = True


class UnivariateARCHModel:
    def __init__(self, distribution=normal_distribution, device=None):
        self.a = self.b = self.c = self.d = None
        self.sample_std = None
        self.distribution = distribution
        self.device = device

    def initialize_parameters(self, n):
        self.a = (1.0 - DECAY_FOR_INITIALIZATION) * torch.ones(n, device=self.device)
        self.b = DECAY_FOR_INITIALIZATION * torch.ones(n, device=self.device)
        self.c = torch.ones(n, device=self.device)
        self.d = torch.ones(n, device=self.device)

        for m in (self.a, self.b, self.c, self.d):
            m.requires_grad = True

    def __simulate(
        self,
        observations: torch.Tensor,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols) of observations

        """
        sigma_t = self.d * self.sample_std
        sigma_sequence = []

        for k, o in enumerate(observations):
            # Store the current ht before predicting next one
            sigma_sequence.append(sigma_t)

            # The variance is (a * sigma)**2 + (b * o)**2 + (c * sample_std)**2
            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.
            a_sigma = torch.clamp(self.a * sigma_t, min=MIN_CLAMP, max=MAX_CLAMP)
            b_o = self.b * o
            c_sample_std = self.c * self.sample_std

            # To avoid numerical issues associated with expressions of the form
            # sqrt(a**2 + b**2 + c**2), we use a similar trick as for the multivariate
            # case, which is to stack the variables (a, b, c) vertically and take
            # the column norms.  We depend on the vector_norm()
            # implementation being stable.

            m = torch.stack((a_sigma, b_o, c_sample_std), axis=0)
            sigma_t = torch.linalg.vector_norm(m, dim=0)

        sigma = torch.stack(sigma_sequence)
        return sigma

    def __mean_log_likelihood(self, observations):
        """
        This computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        sigma = self.__simulate(observations)
        mean_ll = marginal_conditional_log_likelihood(
            observations, sigma, self.distribution
        )
        return mean_ll

    def fit(self, observations):
        n = observations.shape[1]

        self.initialize_parameters(n)
        logging.debug(f"a:\n{self.a}")
        logging.debug(f"b:\n{self.b}")
        logging.debug(f"c:\n{self.c}")
        logging.debug(f"d:\n{self.d}")

        self.sample_std = torch.std(observations, dim=0)
        logging.info(f"sample_std:\n{self.sample_std}")

        optim = torch.optim.LBFGS(
            [self.a, self.b, self.c, self.d],
            max_iter=PROGRESS_ITERATIONS,
            lr=LEARNING_RATE,
            line_search_fn="strong_wolfe",
        )

        def loss_closure():
            optim.zero_grad()
            loss = -self.__mean_log_likelihood(observations)
            loss.backward()
            return loss

        optimize(optim, loss_closure)
        logging.info(
            f"a: {self.a.detach().numpy()}, b: {self.b.detach().numpy()}, c: {self.c.detach().numpy()}, d: {self.d.detach().numpy()}"
        )

    def simulate(
        self,
        observations: torch.Tensor,
    ):
        """
        This is the inference version of simulate(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        with torch.no_grad():
            sigma = self.__simulate(observations)

        return sigma

    def mean_log_likelihood(self, observations):
        """
        This is the inference version of mean_log_likelihood(), which is the version clients would normally use.
        It computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        with torch.no_grad():
            result = self.__mean_log_likelihood(observations, self.sample_std)

        return result


class MultivariateARCHModel:
    def __init__(
        self,
        parameterization=Parameterization.FULL,
        univariate_model=None,
        distribution=normal_distribution,
        device=None,
    ):
        self.parameterization = parameterization
        self.univariate_model = univariate_model
        self.distribution = distribution
        self.device = device

        self.a = self.b = self.c = self.d = None

    def initialize_parameters(self, n):
        # Initialize a and b as simple multiples of the identity
        self.a = (1.0 - DECAY_FOR_INITIALIZATION) * torch.eye(n, device=self.device)
        self.b = DECAY_FOR_INITIALIZATION * torch.eye(n, device=self.device)

        # c is triangular
        self.c = random_lower_triangular(n, 0.01, False, device=self.device)
        self.d = torch.eye(n, device=self.device)

        if self.parameterization == Parameterization.DIAGONAL:
            self.a = torch.diag(torch.diag(self.a))
            self.b = torch.diag(torch.diag(self.b))
            self.c = torch.diag(torch.diag(self.c))
            self.d = torch.diag(torch.diag(self.d))

        # We set requires_grad here so that the preceeding diags and tril calls
        # aren't subject to differentiation

        for t in [self.a, self.b, self.c, self.d]:
            t.requires_grad = True

    def __constrain_parameters(self):
        if self.parameterization == Parameterization.TRIANGULAR:
            a = torch.tril(self.a)
            b = torch.tril(self.b)
            c = torch.tril(self.c)
            d = torch.tril(self.d)
        elif self.parameterization == Parameterization.DIAGONAL:
            a = torch.diag(torch.diag(self.a))
            b = torch.diag(torch.diag(self.b))
            c = torch.diag(torch.diag(self.c))
            d = torch.diag(torch.diag(self.d))
        else:
            a = self.a
            b = self.b
            c = self.c
            d = self.d

        return a, b, c, d

    def __simulate(
        self,
        observations: torch.Tensor,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        lower triangular square roots of the sequence of covariance matrix estimates.

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols) of observations
        """
        a, b, c, d = self.__constrain_parameters()

        ht = d @ self.h_bar
        h_sequence = []

        for k, o in enumerate(observations):
            # Store the current ht before predicting next one
            h_sequence.append(ht)

            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.

            a_ht = torch.clamp(a @ ht, min=MIN_CLAMP, max=MAX_CLAMP)
            b_o = (b @ o).unsqueeze(1)

            # The covariance is a_ht @ a_ht.T + b_o @ b_o.T + (c @ h_bar) @ (c @ h_bar).T
            # Unnecessary squaring is discouraged for nunerical stability.
            # Instead, we use only square roots and never explicity
            # compute the covariance.  This is a common 'trick' achieved
            # by concatenating the square roots in a larger array and
            # computing the QR factoriation, which computes the square
            # root of the sum of squares.  The covariance matrix isn't
            # formed explicitly in this code except at the very end when
            # it's time to return the covariance matrices to the user.

            m = torch.cat((a_ht, b_o, c @ self.h_bar), axis=1)

            # Unfortunately there's no QL factorization in PyTorch so we
            # transpose m and use the QR.  We only need the 'R' return
            # value, so the Q return value is dropped.

            ht_t = torch.linalg.qr(m.T, mode="reduced")[1]

            # Transpose ht to get the lower triangular version.

            ht = make_diagonal_nonnegative(ht_t.T)

        h = torch.stack(h_sequence)
        return h

    def __mean_log_likelihood(self, observations, sigma=None):
        """
        This computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        # Running through a tril() will force autograd to compute a zero gradient
        # for the upper triangular portion so that those entries are ignored.
        # When sigma is not None, h_bar should also be normalized
        # so that its row norms are one.

        if sigma is not None:
            scaled_observations = observations / sigma
        else:
            scaled_observations = observations

        h = self.__simulate(scaled_observations)

        # It's important to use non-scaled observations in likelihood function
        mean_ll = joint_conditional_log_likelihood(
            observations, h, self.distribution, sigma=sigma
        )

        return mean_ll

    def fit(self, observations):
        n = observations.shape[1]

        self.initialize_parameters(n)
        logging.debug(f"a:\n{self.a}")
        logging.debug(f"b:\n{self.b}")
        logging.debug(f"c:\n{self.c}")
        logging.debug(f"d:\n{self.d}")

        self.h_bar = (torch.linalg.qr(observations, mode="reduced")[1]).T / torch.sqrt(
            torch.tensor(observations.shape[0])
        )
        self.h_bar = make_diagonal_nonnegative(self.h_bar)

        if self.univariate_model:
            self.h_bar = torch.nn.functional.normalize(self.h_bar, dim=1)
            self.univariate_model.fit(observations)
            sigma_est = self.univariate_model.simulate(observations)
        else:
            self.univariate_model = None
            sigma_est = None

        logging.info(f"h_bar:\n{self.h_bar}")

        optim = torch.optim.LBFGS(
            [self.a, self.b, self.c, self.d],
            max_iter=PROGRESS_ITERATIONS,
            lr=LEARNING_RATE,
            line_search_fn="strong_wolfe",
        )

        def loss_closure():
            optim.zero_grad()

            # Do not use scaled observations here.
            loss = -self.__mean_log_likelihood(observations, sigma=sigma_est)
            loss.backward()

            return loss

        optimize(optim, loss_closure)
        logging.info(
            f"a: {self.a.detach().numpy()}\nb: {self.b.detach().numpy()}\nc: {self.c.detach().numpy()}\nd: {self.d.detach().numpy()}"
        )

        logging.debug("Gradients: ")
        logging.debug(f"a.grad:\n{self.a.grad}")
        logging.debug(f"b.grad:\n{self.b.grad}")
        logging.debug(f"c.grad:\n{self.c.grad}")
        logging.debug(f"d.grad:\n{self.d.grad}")

    def simulate(
        self,
        observations: torch.Tensor,
    ):
        """
        This is the inference version of simulate(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        with torch.no_grad():
            if self.univariate_model is not None:
                sigma_est = self.univariate_model.simulate(observations)
                scaled_observations = observations / sigma_est
            else:
                sigma_est = None
                scaled_observations = observations

            result = self.__simulate(scaled_observations)

        return result, sigma_est

    def mean_log_likelihood(self, observations: torch.Tensor):
        """
        This is the inference version of mean_log_likelihood(), which is the version clients would normally use.
        It computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).

        Arguments:
            observations: torch.Tensor of shape (n_obs, n_symbols)

        Return value:
            float - mean (per sample) log likelihood

        """
        with torch.no_grad():
            if self.univariate_model:
                sigma = self.univariate_model.simulate(observations)
            else:
                sigma = None

            result = self.__mean_log_likelihood(observations, sigma)

        return float(result)


if __name__ == "__main__":
    # TODO
    pass
