# Standard Python
import logging

import os
from typing import Any, Callable, Dict, Iterable, Iterator, Union, Tuple

from enum import Enum

# Common packages
import numpy as np
import torch

DEBUG = False
PROGRESS_ITERATIONS = 20

INITIAL_DECAY = 0.30
LEARNING_RATE = 0.25
EPS = 1e-6

MAX_CLAMP = 1e10
MIN_CLAMP = -MAX_CLAMP

DEFAULT_SEED = 42

normal_distribution = torch.distributions.normal.Normal(
    loc=torch.tensor(0.0), scale=torch.tensor(1.0)
)


class ParameterConstraint(Enum):
    SCALAR = "scalar"
    DIAGONAL = "diagonal"
    TRIANGULAR = "triangular"
    FULL = "full"


class ScalarParameter:
    def __init__(self, n: int, scale: float = 1.0, device: torch.device = None):
        self.device = device
        self.value = scale * torch.tensor(1.0, device=device)
        self.value.requires_grad = True

    def set(self, value: float):
        if not isinstance(value, tensor):
            value = torch.tensor(
                value, device=self.device, dtype=torch.float, requires_grad=True
            )
        self.value = value

    def __matmul__(self, other: torch.Tensor):
        try:
            return self.value * other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


class DiagonalParameter:
    def __init__(self, n: int, scale: float = 1.0, device: torch.device = None):
        self.device = device
        self.value = scale * torch.ones(n, device=device)
        self.value.requires_grad = True

    def set(self, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(
                value, device=self.device, dtype=torch.float, requires_grad=True
            )

        if len(value.shape) != 1:
            raise ValueError(f"value: {value} should have one and only one dimension")

        self.value = value

    def __matmul__(self, other: torch.Tensor):
        try:
            return self.value * other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


class TriangularParameter:
    def __init__(self, n: int, scale: float = 1.0, device: torch.device = None):
        self.device = device
        self.value = scale * torch.eye(n, device=device)
        self.value.requires_grad = True

    def __matmul__(self, other: torch.Tensor):
        # self.value was initialized to be triangular, so this
        # torch.tril() may seem unnecessary.  Using the torch.tril()
        # ensures that the upper entries remain excluded from gradient
        # calculations and don't get updated by the optimizer.
        try:
            return torch.tril(self.value) @ other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


class FullParameter:
    def __init__(self, n: int, scale: float = 1.0, device: torch.device = None):
        self.device = device
        self.value = scale * torch.eye(n, device=device)
        self.value.requires_grad = True

    def set(self, value: Any):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(
                value, device=self.device, dtype=torch.float, requires_grad=True
            )
        self.value = value

    def __matmul__(self, other: torch.Tensor):
        try:
            return self.value @ other
        except Exception as e:
            print(e)
            print(f"self.value: {self.value}")
            print(f"other: {other}")
            raise e


def make_diagonal_nonnegative(m: torch.Tensor):
    """Given a single lower triangular matrices m, return an `equivalent` matrix
    having non-negative diagonal entries.  Here `equivalent` means m@m.T is unchanged.

    Arguments:
        m: torch.Tensor of shape (n,n) that is lower triangular
    Returns:
        torch.Tensor: of shape (n, n) which is `equivalent` and has non-negative
        values on its diagonal
    """
    diag = torch.diag(m)
    diag_signs = torch.ones(diag.shape)
    diag_signs[diag < 0.0] = -1
    return m * diag_signs


def random_lower_triangular(
    n: int, scale: float = 1.0, requires_grad: bool = True, device: torch.device = None
):
    scale = torch.tensor(scale, device=device)
    m = torch.rand(n, n, device=device) - 0.5
    m = scale * make_diagonal_nonnegative(torch.tril(m))
    m.requires_grad = requires_grad
    return m


def marginal_conditional_log_likelihood(
    observations: torch.Tensor, sigma: torch.Tensor, distribution: torch.Tensor
):
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
    observations: torch.Tensor,
    transformations: torch.Tensor,
    distribution: torch.distributions.Distribution,
    sigma=Union[torch.Tensor, None],
):
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       transformations: torch.Tensor of shape (n_symbols, n_symbols)
           transformation is a lower-triangular matrix and the outcome is presumed
           to be z = transformation @ e where the elements of e are iid from distrbution.
       distribution: torch.distributions.Distribution or other object with a log_prob() method.
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
       torch.Tensor - the mean (per sample) log_likelihood"""

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


def optimize(
    optim: torch.optim.Optimizer, closure: Callable[[], torch.Tensor], label: str = ""
):
    """
    This is a wrapper around optim.step() that higher level monitoring.
    Arguments:
       optim: torch.optim.Optimizer - optimizer to use.
       closure: a "closure" that evaluates the objective function
                and the Pytorch optimizer closure() conventions
                which include 1) zeroing the gradient; 2) evaluating
                the objective; 3) back-propagating the derivative
                informaiton; 4) returning the objective value.

    Returns: Nothing

    """
    best_loss = float("inf")
    done = False
    logging.info(f"Starting {label}" + (" " if label else "") + "optimization")
    while not done:
        optim.step(closure)
        current_loss = closure()
        logging.info(f"\tCurrent loss: {current_loss:.4f}")
        if float(current_loss) < best_loss:
            best_loss = current_loss
        else:
            logging.info("Finished")
            done = True


class UnivariateARCHModel:
    def __init__(
        self,
        distribution: torch.distributions.Distribution = normal_distribution,
        device: torch.device = None,
    ):
        self.n = self.a = self.b = self.c = self.d = None
        self.sample_std = None
        self.distribution = distribution
        self.device = device

    def initialize_parameters(self, n: int):
        self.n = n
        self.a = DiagonalParameter(n, 1.0 - INITIAL_DECAY, device=self.device)
        self.b = DiagonalParameter(n, INITIAL_DECAY, device=self.device)
        self.c = DiagonalParameter(n, 1.0, device=self.device)
        self.d = DiagonalParameter(n, 1.0, device=self.device)

    def set_parameters(self, a: Any, b: Any, c: Any, d: Any, initial_std: Any):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float, device=self.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float, device=self.device)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float, device=self.device)
        if not isinstance(initial_std, torch.Tensor):
            initial_std = torch.tensor(
                initial_std, dtype=torch.float, device=self.device
            )

        if (
            len(a.shape) != 1
            or a.shape != b.shape
            or a.shape != c.shape
            or a.shape != d.shape
            or a.shape != initial_std.shape
        ):
            raise ValueError(
                f"The shapes of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), d({d.shape}), and "
                f"initial_std({initial_std.shape}) must have "
                "only and only one dimension that's consistent"
            )

        self.n = a.shape[0]
        self.a = DiagonalParameter(self.n)
        self.b = DiagonalParameter(self.n)
        self.c = DiagonalParameter(self.n)
        self.d = DiagonalParameter(self.n)

        self.a.set(a)
        self.b.set(b)
        self.c.set(c)
        self.d.set(d)

        self.sample_std = initial_std

    def __predict(
        self,
        observations: torch.Tensor,
        sample=False,
        initial_sigma=None,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` unit variance noise
                           rather than actual observations.
            initial_sigma: torch.Tensor - Initial standard deviation vector
        Returns:
            sigma: torch.Tensor of predictions for each observation
            sigma_next: torch.Tensor prediction for next unobserved value

        """
        if initial_sigma:
            if not isinstance(initial_sigma, torch.Tensor):
                initial_sigma = torch.tensor(
                    initial_sigma, dtype=torch.float, device=self.device
                )
            sigma_t = initial_sigma
        else:
            sigma_t = self.d @ self.sample_std

        sigma_t = torch.maximum(sigma_t, torch.tensor(float(EPS)))
        sigma_sequence = []

        for k, obs in enumerate(observations):
            # Store the current ht before predicting next one
            sigma_sequence.append(sigma_t)

            # The variance is (a * sigma)**2 + (b * o)**2 + (c * sample_std)**2
            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.
            a_sigma = torch.clamp(self.a @ sigma_t, min=MIN_CLAMP, max=MAX_CLAMP)

            if sample:
                # obs is noise that must be scaled
                obs = sigma_t * obs

            b_o = self.b @ obs
            c_sample_std = self.c @ self.sample_std

            # To avoid numerical issues associated with expressions of the form
            # sqrt(a**2 + b**2 + c**2), we use a similar trick as for the multivariate
            # case, which is to stack the variables (a, b, c) vertically and take
            # the column norms.  We depend on the vector_norm()
            # implementation being stable.

            m = torch.stack((a_sigma, b_o, c_sample_std), axis=0)
            sigma_t = torch.linalg.vector_norm(m, dim=0)

        sigma = torch.stack(sigma_sequence)
        return sigma, sigma_t

    def __mean_log_likelihood(self, observations: torch.Tensor):
        """
        Compute and return the mean (per-sample) log likelihood (the total log likelihood divided by the number of samples).
        """
        sigma = self.__predict(observations)[0]
        mean_ll = marginal_conditional_log_likelihood(
            observations, sigma, self.distribution
        )
        return mean_ll

    def fit(self, observations: torch.Tensor):
        self.initialize_parameters(observations.shape[1])

        logging.debug(f"a:\n{self.a.value}")
        logging.debug(f"b:\n{self.b.value}")
        logging.debug(f"c:\n{self.c.value}")
        logging.debug(f"d:\n{self.d.value}")

        self.sample_std = torch.std(observations, dim=0)
        logging.info(f"sample_std:\n{self.sample_std}")

        optim = torch.optim.LBFGS(
            [self.a.value, self.b.value, self.c.value, self.d.value],
            max_iter=PROGRESS_ITERATIONS,
            lr=LEARNING_RATE,
            line_search_fn="strong_wolfe",
        )

        def loss_closure():
            if DEBUG:
                print(f"a: {self.a.value}")
                print(f"b: {self.b.value}")
                print(f"c: {self.c.value}")
                print(f"d: {self.d.value}")
                print()
            optim.zero_grad()
            loss = -self.__mean_log_likelihood(observations)
            loss.backward()
            return loss

        optimize(optim, loss_closure, "univariate model")
        logging.info(
            f"a: {self.a.value.detach().numpy()}, "
            f"b: {self.b.value.detach().numpy()}, "
            f"c: {self.c.value.detach().numpy()}, "
            f"d: {self.d.value.detach().numpy()}"
        )

    def predict(
        self,
        observations: torch.Tensor,
    ):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        with torch.no_grad():
            sigma, sigma_next = self.__predict(observations)

        return sigma, sigma_next

    def sample(
        self, n: Union[torch.Tensor, int], initial_sigma: Union[torch.Tensor, None]
    ):
        """
        Generate a random sampled output from the model.
        Arguments:
            n: torch.Tensor - Noise to use as input or
               int - Number of points to generate, in which case GWN is used.
        Returns:
            output: torch.Tensor - Sample model output
            sigma: torch.Tensor - Sigma value used to scale the sample
        """
        with torch.no_grad():
            if isinstance(n, int):
                n = torch.randn(n, self.n)

            sigma, sigma_next = self.__predict(
                n, sample=True, initial_sigma=initial_sigma
            )
            output = sigma * n
        return output, sigma

    def mean_log_likelihood(self, observations: torch.Tensor):
        """
        This is the inference version of mean_log_likelihood(), which is the version clients would normally use.
        It computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        with torch.no_grad():
            result = self.__mean_log_likelihood(observations)

        return float(result)


class MultivariateARCHModel:
    def __init__(
        self,
        constraint=ParameterConstraint.FULL,
        univariate_model: Union[UnivariateARCHModel, None] = None,
        distribution: torch.distributions.Distribution = normal_distribution,
        device: torch.device = None,
    ):
        self.constraint = constraint
        self.univariate_model = univariate_model
        self.distribution = distribution
        self.device = device

        self.n = self.a = self.b = self.c = self.d = None

        if constraint == ParameterConstraint.SCALAR:
            self.parameter_factory = ScalarParameter
        elif constraint == ParameterConstraint.DIAGONAL:
            self.parameter_factory = DiagonalParameter
        elif constraint == ParameterConstraint.TRIANGULAR:
            self.parameter_factory = TriangularParameter
        else:
            self.parameter_factory = FullParameter

    def initialize_parameters(self, n: int):
        self.n = n
        # Initialize a and b as simple multiples of the identity
        self.a = self.parameter_factory(n, 1.0 - INITIAL_DECAY, device=self.device)
        self.b = self.parameter_factory(n, INITIAL_DECAY, device=self.device)
        self.c = self.parameter_factory(n, 0.01, device=self.device)
        self.d = self.parameter_factory(n, 1.0, device=self.device)

    def set_parameters(self, a: Any, b: Any, c: Any, d: Any, initial_h: Any):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float, device=self.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float, device=self.device)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float, device=self.device)
        if not isinstance(initial_h, torch.Tensor):
            initial_h = torch.tensor(initial_h, dtype=torch.float, device=self.device)
        if (
            len(a.shape) != 2
            or a.shape != b.shape
            or a.shape != c.shape
            or a.shape != d.shape
            or a.shape != initial_h.shape
        ):
            raise ValueError(
                f"There must be two dimensions of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), d({d.shape}), and "
                f"initial_h({initial_h.shape}) that all agree"
            )

        self.n = a.shape[0]
        self.a = FullParameter(self.n)
        self.b = FullParameter(self.n)
        self.c = FullParameter(self.n)
        self.d = FullParameter(self.n)

        self.a.set(a)
        self.b.set(b)
        self.c.set(c)
        self.d.set(d)

        if not isinstance(initial_h, torch.Tensor):
            initial_h = torch.tensor(initial_h, device=self.device, dtype=torch.float)
        self.h_bar = initial_h

    def __predict(
        self,
        observations: torch.Tensor,
        sample=False,
        initial_h=None,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        lower triangular square roots of the sequence of covariance matrix estimates.

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols) of observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` unit variance noise
                           rather than actual observations.
            initial_h: torch.Tensor - Initial covariance lower-triangular sqrt.
        Returns:
            h: torch.Tensor of predictions for each observation
            h_next: torch.Tensor prediction for next unobserved value
        """
        if initial_h:
            if not isinstance(initial_h, torch.Tensor):
                initial_h = torch.tensor(
                    initial_h, dtype=torch.float, device=self.device
                )
            ht = initial_h
        else:
            ht = self.d @ self.h_bar

        # We require ht to be lower traingular (even when parameters are full)
        # Ensure this using QR.
        ht_t = torch.linalg.qr(ht, mode="reduced")[1]
        ht = ht_t.T

        if DEBUG:
            print(f"Initial ht: {ht}")
            print(f"self.d: {self.d.value}")
            print(f"self.h_bar: {self.h_bar}")
        h_sequence = []

        for k, obs in enumerate(observations):
            # Store the current ht before predicting next one
            h_sequence.append(ht)

            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.

            a_ht = torch.clamp(self.a @ ht, min=MIN_CLAMP, max=MAX_CLAMP)

            if sample:
                # obs is noise that must be scaled
                obs = ht @ obs

            b_o = (self.b @ obs).unsqueeze(1)
            c_hbar = self.c @ self.h_bar

            # The covariance is a_ht @ a_ht.T + b_o @ b_o.T + (c @ h_bar) @ (c @ h_bar).T
            # Unnecessary squaring is discouraged for nunerical stability.
            # Instead, we use only square roots and never explicity
            # compute the covariance.  This is a common 'trick' achieved
            # by concatenating the square roots in a larger array and
            # computing the QR factoriation, which computes the square
            # root of the sum of squares.  The covariance matrix isn't
            # formed explicitly in this code except at the very end when
            # it's time to return the covariance matrices to the user.

            m = torch.cat((a_ht, b_o, c_hbar), axis=1)

            # Unfortunately there's no QL factorization in PyTorch so we
            # transpose m and use the QR.  We only need the 'R' return
            # value, so the Q return value is dropped.

            ht_t = torch.linalg.qr(m.T, mode="reduced")[1]

            # Transpose ht to get the lower triangular version.

            ht = make_diagonal_nonnegative(ht_t.T)

        h = torch.stack(h_sequence)
        return h, ht

    def __mean_log_likelihood(
        self, observations: torch.Tensor, sigma: Union[torch.Tensor, None] = None
    ):
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

        h = self.__predict(scaled_observations)[0]

        # It's important to use non-scaled observations in likelihood function
        mean_ll = joint_conditional_log_likelihood(
            observations, h, self.distribution, sigma=sigma
        )

        return mean_ll

    def fit(self, observations: torch.Tensor):
        n = observations.shape[1]
        self.initialize_parameters(n)

        logging.debug(f"a:\n{self.a.value}")
        logging.debug(f"b:\n{self.b.value}")
        logging.debug(f"c:\n{self.c.value}")
        logging.debug(f"d:\n{self.d.value}")

        self.h_bar = (torch.linalg.qr(observations, mode="reduced")[1]).T / torch.sqrt(
            torch.tensor(observations.shape[0])
        )
        self.h_bar = make_diagonal_nonnegative(self.h_bar)

        if self.univariate_model:
            self.h_bar = torch.nn.functional.normalize(self.h_bar, dim=1)
            self.univariate_model.fit(observations)
            sigma_est = self.univariate_model.predict(observations)[0]
        else:
            self.univariate_model = None
            sigma_est = None

        logging.info(f"h_bar:\n{self.h_bar}")

        optim = torch.optim.LBFGS(
            [self.a.value, self.b.value, self.c.value, self.d.value],
            max_iter=PROGRESS_ITERATIONS,
            lr=LEARNING_RATE,
            line_search_fn="strong_wolfe",
        )

        def loss_closure():
            if DEBUG:
                print(f"a: {self.a.value}")
                print(f"b: {self.b.value}")
                print(f"c: {self.c.value}")
                print(f"d: {self.d.value}")
                print()
            optim.zero_grad()

            # Do not use scaled observations here.
            loss = -self.__mean_log_likelihood(observations, sigma=sigma_est)
            loss.backward()

            return loss

        optimize(optim, loss_closure, "multivariate model")

        logging.info("Final Parameter Values")
        logging.info(f"a: {self.a.value.detach().numpy()}")
        logging.info(f"b: {self.b.value.detach().numpy()}")
        logging.info(f"c: {self.c.value.detach().numpy()}")
        logging.info(f"d: {self.d.value.detach().numpy()}")

        logging.debug("Gradients: ")
        logging.debug(f"a.grad:\n{self.a.value.grad}")
        logging.debug(f"b.grad:\n{self.b.value.grad}")
        logging.debug(f"c.grad:\n{self.c.value.grad}")
        logging.debug(f"d.grad:\n{self.d.value.grad}")

    def predict(
        self,
        observations: torch.Tensor,
    ):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        with torch.no_grad():
            if self.univariate_model is not None:
                sigma, sigma_next = self.univariate_model.predict(observations)
                unscaled_h, unscaled_h_next = self.__predict(observations / sigma)
                h = sigma.unsqueeze(2).expand(unscaled_h.shape) * unscaled_h
                h_next = (
                    sigma_next.unsqueeze(1).expand(unscaled_h_next.shape)
                    * unscaled_h_next
                )
            else:
                h, h_next = self.__predict(observations)
                sigma = sigma_next = None

        return h, h_next, sigma, sigma_next

    def sample(
        self,
        n: Union[torch.Tensor, int],
        initial_h: Union[torch.Tensor, None],
        initial_sigma: Union[torch.Tensor, None] = None,
    ):
        """
        Generate a random sampled output from the model.
        Arguments:
            n: torch.Tensor - Noise to use as input or
               int - Number of points to generate, in which case GWN is used.
            initial_h: torch.Tensor - Initial condition for sqrt of
                       covariance matrix (or correlation matrix when
                       internal univariate model is used)
            initial_sigma: torch.Tensor - Initial sigma for internal
                            univariate model if one is used
        Returns:
            output: torch.Tensor - Sample model output
            h: torch.Tensor - Sqrt of covariance used to scale the sample
        """
        if initial_sigma is not None and self.univariate_model is None:
            raise ValueError(
                "Can't specific initial_sigma without an internal univariate model"
            )
        if (
            initial_sigma is None
            and self.univariate_model is not None
            and initial_h is not None
        ):
            logging.WARNING(
                "You provided an initial_h but didn't provide an initial_sigma. This probably isn't what you want."
            )

        with torch.no_grad():
            if isinstance(n, int):
                n = torch.randn(n, self.n)

            h, h_next = self.__predict(n, sample=True, initial_h=initial_h)

            output = (h @ n.unsqueeze(2)).squeeze(2)

            if self.univariate_model:
                output, sigma = univariate_model.sample(output, initial_sigma)
            else:
                sigma = None

        return output, h, sigma

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
                sigma = self.univariate_model.predict(observations)[0]
            else:
                sigma = None

            result = self.__mean_log_likelihood(observations, sigma)

        return float(result)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(message)s",
        force=True,
    )
    # Example usage:
    univariate_model = UnivariateARCHModel()

    univariate_model.set_parameters(
        a=[0.90], b=[0.33], c=[0.25], d=[1.0], initial_std=[0.01]
    )
    uv_x, uv_sigma = univariate_model.sample(10000, [0.01])
    univariate_model.fit(uv_x)

    # Here's a multivariate case
    multivariate_model = MultivariateARCHModel(
        constraint=ParameterConstraint.TRIANGULAR
    )
    multivariate_model.set_parameters(
        a=[[0.92, 0.0, 0.0], [-0.03, 0.95, 0.0], [-0.04, -0.02, 0.97]],
        b=[[0.4, 0.0, 0.0], [0.1, 0.3, 0.0], [0.13, 0.08, 0.2]],
        c=[[0.07, 0.0, 0.0], [0.04, 0.1, 0.0], [0.05, 0.005, 0.08]],
        d=[[1.0, 0.0, 0.0], [0.1, 0.6, 0.0], [-1.2, -0.8, 2]],
        initial_h=[[0.008, 0.0, 0.0], [0.008, 0.01, 0.0], [0.008, 0.009, 0.005]],
    )

    mv_x, mv_h, mv_sigma = multivariate_model.sample(
        50000, [[0.008, 0.0, 0.0], [0.008, 0.01, 0.0], [0.008, 0.009, 0.005]]
    )
