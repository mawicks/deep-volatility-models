# Standard Python
from abc import abstractmethod
import logging

import os
from typing import Any, Callable, Dict, Iterable, Iterator, Protocol, Union, Tuple

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
    def __init__(
        self, n: int, scale: float = 1.0, device: Union[torch.device, None] = None
    ):
        self.device = device
        self.value = scale * torch.tensor(1.0, device=device)
        self.value.requires_grad = True

    def set(self, value: Union[float, torch.Tensor]):
        if not isinstance(value, torch.Tensor):
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
    def __init__(
        self, n: int, scale: float = 1.0, device: Union[torch.device, None] = None
    ):
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
    def __init__(
        self, n: int, scale: float = 1.0, device: Union[torch.device, None] = None
    ):
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
    def __init__(
        self, n: int, scale: float = 1.0, device: Union[torch.device, None] = None
    ):
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
    n: int,
    scale: Union[torch.Tensor, float] = 1.0,
    requires_grad: bool = True,
    device: Union[torch.device, None] = None,
):
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=device)
    m = torch.rand(n, n, device=device) - 0.5
    m = scale * make_diagonal_nonnegative(torch.tril(m))
    m.requires_grad = requires_grad
    return m


def marginal_conditional_log_likelihood(
    observations: torch.Tensor,
    scale: torch.Tensor,
    distribution: torch.distributions.Distribution,
):
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       scale: torch.Tensor of shape (n_obs, n_symbols)
           Contains the estimated univariate (marginal) standard deviation for each observation.
       distribution: torch.distributions.distribution.Distribution instance which should have a log_prob() method.
           Note we assume distrubution was constructed with center=0 and shape=1.  Any normalizing and recentering
           is achieved by explicit`transformations` here.

    Returns the mean log_likelihood"""

    scaled_observations = observations / scale
    logging.debug(f"scaled_observations: \n{scaled_observations}")

    # Compute the log likelihoods on the innovations
    ll = distribution.log_prob(scaled_observations) - torch.log(scale)

    # For consistency with the multivariate case, we *sum* over the
    # variables (columns) first and *average* over the rows (observations).
    # Summing over the variables is equivalent to multiplying the
    # marginal distributions to get a join distribution.

    return torch.mean(torch.sum(ll, dim=1))


def joint_conditional_log_likelihood(
    observations: torch.Tensor,
    mv_scale: torch.Tensor,
    uv_scale=Union[torch.Tensor, None],
    distribution: torch.distributions.Distribution = normal_distribution,
):
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       mv_scale: torch.Tensor of shape (n_symbols, n_symbols)
           transformation is a lower-triangular matrix and the outcome is presumed
           to be z = transformation @ e where the elements of e are iid from distrbution.
       distribution: torch.distributions.Distribution or other object with a log_prob() method.
           Note we assume distrubution was constructed with center=0 and shape=1.  Any normalizing and recentering
           is achieved by explicit`transformations` here.
       uv_scale: torch.Tensor of shape (n_obj, n_symbols) or None.
              When uv_scale is specified, the observed variables
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

    log_det = torch.log(torch.abs(torch.diagonal(mv_scale, dim1=1, dim2=2)))

    if uv_scale is not None:
        log_det = log_det + torch.log(torch.abs(uv_scale))
        observations = observations / uv_scale

    # First get the innovations sequence by forming transformation^(-1)*observations
    # The unsqueeze is necessary because the matmul is on the 'batch'
    # of observations.  The shape of `t` is (n_obs, n, n) while
    # the shape of `observations` is (n_obs, n). Without adding the
    # extract dimension to observations, the solver won't see conforming dimensions.
    # We remove the ambiguity, by making observations` have shape (n_obj, n, 1), then
    # we remove the extra dimension from e.
    e = torch.linalg.solve_triangular(
        mv_scale,
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


class MeanModel(Protocol):
    @abstractmethod
    def set_parameters():
        raise NotImplementedError

    @abstractmethod
    def _predict(
        self,
        observations: torch.Tensor,
        initial_mean: Union[torch.Tensor, Any, None] = None,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            initial_mean: torch.Tensor (or something convertible to one)
                          Initial mean vector if specified
        Returns:
            mu: torch.Tensor of predictions for each observation
            mu_next: torch.Tensor prediction for next unobserved value

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, observations: torch.Tensor, initial_mean=None):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_parameters(self, observations: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self):
        raise NotImplementedError


class ZeroMeanModel(MeanModel):
    def __init__(
        self,
        device: Union[torch.device, None] = None,
    ):
        self.device = device

    def set_parameters():
        pass

    def _predict(
        self,
        observations: torch.Tensor,
        initial_mean: Union[torch.Tensor, Any, None] = None,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            initial_mean: torch.Tensor (or something convertible to one)
                          Initial mean vector if specified
        Returns:
            mu: torch.Tensor of predictions for each observation
            mu_next: torch.Tensor prediction for next unobserved value

        """
        mu = torch.zeros(observations.shape, device=self.device)
        mu_next = torch.zeros(observations.shape[1])
        return mu, mu_next

    def predict(self, observations: torch.Tensor, initial_mean=None):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        return self._predict(observations, initial_mean)

    def get_parameters(self):
        return []

    def initialize_parameters(self, observations: torch.Tensor):
        pass

    def log_parameters(self):
        pass


class ARMAMeanModel(MeanModel):
    def __init__(
        self,
        device: Union[torch.device, None] = None,
    ):
        self.n = self.a = self.b = self.c = self.d = None
        self.sample_mean = None
        self.device = device

    def initialize_parameters(self, observations: torch.Tensor):
        self.n = observations.shape[1]
        self.a = DiagonalParameter(self.n, 1.0 - INITIAL_DECAY, device=self.device)
        self.b = DiagonalParameter(self.n, INITIAL_DECAY, device=self.device)
        self.c = DiagonalParameter(self.n, 1.0, device=self.device)
        self.d = DiagonalParameter(self.n, 1.0, device=self.device)
        self.sample_mean = torch.mean(observations, dim=0)

    def set_parameters(self, a: Any, b: Any, c: Any, d: Any, initial_mean: Any):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float, device=self.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float, device=self.device)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float, device=self.device)
        if not isinstance(initial_mean, torch.Tensor):
            initial_mean = torch.tensor(
                initial_mean, dtype=torch.float, device=self.device
            )

        if (
            len(a.shape) != 1
            or a.shape != b.shape
            or a.shape != c.shape
            or a.shape != d.shape
            or a.shape != initial_mean.shape
        ):
            raise ValueError(
                f"The shapes of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), d({d.shape}), and "
                f"initial_mean({initial_mean.shape}) must have "
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

        self.sample_mean = initial_mean

    def _predict(
        self,
        observations: torch.Tensor,
        initial_mean: Union[torch.Tensor, Any, None] = None,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            initial_mean: torch.Tensor (or something convertible to one)
                          Initial mean vector if specified
        Returns:
            mu: torch.Tensor of predictions for each observation
            mu_next: torch.Tensor prediction for next unobserved value

        """
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise Exception("Mean model has not been fit()")

        if initial_mean:
            if not isinstance(initial_mean, torch.Tensor):
                initial_mean = torch.tensor(
                    initial_mean, dtype=torch.float, device=self.device
                )
            mu_t = initial_mean
        else:
            mu_t = self.d @ self.sample_mean  # type: ignore

        mu_sequence = []

        for k, obs in enumerate(observations):
            # Store the current ht before predicting next one
            mu_sequence.append(mu_t)

            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.
            a_mu = torch.clamp(self.a @ mu_t, min=MIN_CLAMP, max=MAX_CLAMP)
            b_o = self.b @ obs
            c_sample_mean = self.c @ self.sample_mean  # type: ignore

            mu_t = a_mu + b_o + c_sample_mean

        mu = torch.stack(mu_sequence)
        return mu, mu_t

    def predict(self, observations: torch.Tensor, initial_mean=None):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        with torch.no_grad():
            mu, mu_next = self._predict(observations, initial_mean)

        return mu, mu_next

    def get_parameters(self):
        return [self.a.value, self.b.value, self.c.value, self.d.value]

    def log_parameters(self):
        if self.a and self.b and self.c and self.d:
            logging.info(
                "ARMA mean model\n"
                f"a: {self.a.value.detach().numpy()}, "
                f"b: {self.b.value.detach().numpy()}, "
                f"c: {self.c.value.detach().numpy()}, "
                f"d: {self.d.value.detach().numpy()}"
            )
        else:
            logging.info("ARMA mean model has no initialized parameters")


class UnivariateScalingModel(Protocol):
    @abstractmethod
    def set_parameters():
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self, observations: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        observations: torch.Tensor,
    ):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def mean_log_likelihood(self, observations: torch.Tensor):
        """
        This is the inference version of mean_log_likelihood(), which is the version clients would normally use.
        It computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self):
        raise NotImplementedError


class UnivariateUnitScalingModel(UnivariateScalingModel):
    def __init__(
        self,
        distribution: torch.distributions.Distribution = normal_distribution,
        device: Union[torch.device, None] = None,
        mean_model: MeanModel = ZeroMeanModel(),
    ):
        self.distribution = distribution
        self.device = device
        self.mean_model = mean_model

    def set_parameters():
        pass

    def _predict(
        self,
        observations: torch.Tensor,
        sample=False,
        initial_scale=None,
    ):
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` unit variance noise
                           rather than actual observations.
            initial_scale: torch.Tensor - Initial standard deviation vector
        Returns:
            scale: torch.Tensor of scale predictions for each observation
            scale_next: torch.Tensor scale prediction for next unobserved value

        """
        # Use the mean models means
        mu, mu_next = self.mean_model._predict(observations)

        # Set all of the scaling to ones.
        scale = torch.ones(observations.shape)
        scale_next = torch.ones(observations.shape[1])
        return scale, mu, scale_next, mu_next

    def __mean_log_likelihood(self, observations: torch.Tensor):
        """
        Compute and return the mean (per-sample) log likelihood (the total log likelihood divided by the number of samples).
        """
        scale, mu = self._predict(observations)[:2]
        centered_observations = observations - mu
        mean_ll = marginal_conditional_log_likelihood(
            centered_observations, scale, self.distribution
        )
        return mean_ll

    def get_parameters(self):
        return []

    def fit(self, observations: torch.Tensor):
        self.mean_model.initialize_parameters(observations)
        self.mean_model.log_parameters()

        mean_parameters = self.mean_model.get_parameters()

        # There's nothing to do unless the mean model has parameters.
        if len(mean_parameters) > 0:
            optim = torch.optim.LBFGS(
                mean_parameters,
                max_iter=PROGRESS_ITERATIONS,
                lr=LEARNING_RATE,
                line_search_fn="strong_wolfe",
            )

            def loss_closure():
                optim.zero_grad()
                loss = -self.__mean_log_likelihood(observations)
                loss.backward()
                return loss

            optimize(optim, loss_closure, "univariate model")

            self.mean_model.log_parameters()

    def predict(
        self,
        observations: torch.Tensor,
    ):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        with torch.no_grad():
            sigma, mu, sigma_next, mu_next = self._predict(observations)

        return sigma, mu, sigma_next, mu_next

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

            scale, mu = self._predict(n, sample=True, initial_scale=initial_sigma)[:2]
            output = scale * n + mu
        return output, scale, mu

    def mean_log_likelihood(self, observations: torch.Tensor):
        """
        This is the inference version of mean_log_likelihood(), which is the version clients would normally use.
        It computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        with torch.no_grad():
            result = self.__mean_log_likelihood(observations)

        return float(result)

    def log_parameters(self):
        pass


class UnivariateARCHModel(UnivariateScalingModel):
    def __init__(
        self,
        distribution: torch.distributions.Distribution = normal_distribution,
        device: Union[torch.device, None] = None,
        mean_model: MeanModel = ZeroMeanModel(),
    ):
        self.n = self.a = self.b = self.c = self.d = None
        self.sample_mean_scale = None
        self.distribution = distribution
        self.device = device
        self.mean_model = mean_model

    def initialize_parameters(self, observations: torch.Tensor):
        self.n = observations.shape[1]
        self.a = DiagonalParameter(self.n, 1.0 - INITIAL_DECAY, device=self.device)
        self.b = DiagonalParameter(self.n, INITIAL_DECAY, device=self.device)
        self.c = DiagonalParameter(self.n, 1.0, device=self.device)
        self.d = DiagonalParameter(self.n, 1.0, device=self.device)
        self.sample_mean_scale = torch.std(observations, dim=0)

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

        self.sample_mean_scale = initial_std

    def _predict(
        self,
        observations: torch.Tensor,
        sample=False,
        initial_scale=None,
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
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise Exception("Model has not been fit()")

        if initial_scale:
            if not isinstance(initial_scale, torch.Tensor):
                initial_scale = torch.tensor(
                    initial_scale, dtype=torch.float, device=self.device
                )
            scale_t = initial_scale
        else:
            scale_t = self.d @ self.sample_mean_scale  # type: ignore

        mu, mu_next = self.mean_model._predict(observations)
        centered_observations = observations - mu

        scale_t = torch.maximum(scale_t, torch.tensor(float(EPS)))
        scale_sequence = []

        for k, obs in enumerate(centered_observations):
            # Store the current ht before predicting next one
            scale_sequence.append(scale_t)

            # The variance is (a * sigma)**2 + (b * o)**2 + (c * sample_mean_scale)**2
            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.
            a_sigma = torch.clamp(self.a @ scale_t, min=MIN_CLAMP, max=MAX_CLAMP)

            if sample:
                # obs is noise that must be scaled
                obs = scale_t * obs

            b_o = self.b @ obs
            c_sample_mean_scale = self.c @ self.sample_mean_scale  # type: ignore

            # To avoid numerical issues associated with expressions of the form
            # sqrt(a**2 + b**2 + c**2), we use a similar trick as for the multivariate
            # case, which is to stack the variables (a, b, c) vertically and take
            # the column norms.  We depend on the vector_norm()
            # implementation being stable.

            m = torch.stack((a_sigma, b_o, c_sample_mean_scale), dim=0)
            scale_t = torch.linalg.vector_norm(m, dim=0)

        scale = torch.stack(scale_sequence)
        return scale, mu, scale_t, mu_next

    def __mean_log_likelihood(self, observations: torch.Tensor):
        """
        Compute and return the mean (per-sample) log likelihood (the total log likelihood divided by the number of samples).
        """
        scale, mu = self._predict(observations)[:2]
        if mu is not None:
            centered_observations = observations - mu
        else:
            centered_observations = observations
        mean_ll = marginal_conditional_log_likelihood(
            centered_observations, scale, self.distribution
        )
        return mean_ll

    def get_parameters(self):
        return [self.a.value, self.b.value, self.c.value, self.d.value]

    def fit(self, observations: torch.Tensor):
        self.mean_model.initialize_parameters(observations)
        self.mean_model.log_parameters()

        self.initialize_parameters(observations)
        self.log_parameters()

        parameters = self.get_parameters() + self.mean_model.get_parameters()

        optim = torch.optim.LBFGS(
            parameters,
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

        self.log_parameters()

        self.mean_model.log_parameters()

    def predict(
        self,
        observations: torch.Tensor,
    ):
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        with torch.no_grad():
            sigma, mu, sigma_next, mu_next = self._predict(observations)

        return sigma, mu, sigma_next, mu_next

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
            print(self)
            sigma, mu = self._predict(n, sample=True, initial_scale=initial_sigma)[:2]
            output = sigma * n + mu
        return output, sigma

    def mean_log_likelihood(self, observations: torch.Tensor):
        """
        This is the inference version of mean_log_likelihood(), which is the version clients would normally use.
        It computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        with torch.no_grad():
            result = self.__mean_log_likelihood(observations)

        return float(result)

    def log_parameters(self):
        if self.a and self.b and self.c and self.d:
            logging.info(
                "Univariate variance model\n"
                f"a: {self.a.value.detach().numpy()}, "
                f"b: {self.b.value.detach().numpy()}, "
                f"c: {self.c.value.detach().numpy()}, "
                f"d: {self.d.value.detach().numpy()}"
            )
            logging.info(f"sample_mean_scale:\n{self.sample_mean_scale}")
        else:
            logging.info("Univariate variance model has no initialized parameters.")


class MultivariateARCHModel:
    def __init__(
        self,
        constraint=ParameterConstraint.FULL,
        univariate_model: UnivariateScalingModel = UnivariateUnitScalingModel(),
        distribution: torch.distributions.Distribution = normal_distribution,
        device: torch.device = None,
    ):
        self.constraint = constraint
        self.univariate_model = univariate_model
        self.distribution = distribution
        self.device = device

        # There should be a better way to do this.  Maybe add a set_device method.
        self.univariate_model.device = device

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
        self.log_parameters()

    def set_parameters(self, a: Any, b: Any, c: Any, d: Any, initial_scale: Any):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float, device=self.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float, device=self.device)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float, device=self.device)
        if not isinstance(initial_scale, torch.Tensor):
            initial_scale = torch.tensor(
                initial_scale, dtype=torch.float, device=self.device
            )
        if (
            len(a.shape) != 2
            or a.shape != b.shape
            or a.shape != c.shape
            or a.shape != d.shape
            or a.shape != initial_scale.shape
        ):
            raise ValueError(
                f"There must be two dimensions of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), d({d.shape}), and "
                f"initial_scale({initial_scale.shape}) that all agree"
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

        if not isinstance(initial_scale, torch.Tensor):
            initial_scale = torch.tensor(
                initial_scale, device=self.device, dtype=torch.float
            )
        self.sample_mean_scale = initial_scale

    def _predict(
        self,
        observations: torch.Tensor,
        sample=False,
        initial_scale=None,
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
        if initial_scale:
            if not isinstance(initial_scale, torch.Tensor):
                initial_scale = torch.tensor(
                    initial_scale, dtype=torch.float, device=self.device
                )
            scale_t = initial_scale
        else:
            scale_t = self.d @ self.sample_mean_scale

        # We require ht to be lower traingular (even when parameters are full)
        # Ensure this using QR.
        scale_t_T = torch.linalg.qr(scale_t, mode="reduced")[1]
        scale_t = scale_t_T.T

        if DEBUG:
            print(f"Initial scalet: {scale_t}")
            print(f"self.d: {self.d.value}")
            print(f"self.sample_mean_scale: {self.sample_mean_scale}")
        scale_sequence = []

        for k, obs in enumerate(observations):
            # Store the current ht before predicting next one
            scale_sequence.append(scale_t)

            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.

            a_scale_t = torch.clamp(self.a @ scale_t, min=MIN_CLAMP, max=MAX_CLAMP)

            if sample:
                # obs is noise that must be scaled
                obs = scale_t @ obs

            b_o = (self.b @ obs).unsqueeze(1)
            c_hbar = self.c @ self.sample_mean_scale

            # The covariance is a_ht @ a_ht.T + b_o @ b_o.T + (c @ sample_mean_scale) @ (c @ sample_mean_scale).T
            # Unnecessary squaring is discouraged for nunerical stability.
            # Instead, we use only square roots and never explicity
            # compute the covariance.  This is a common 'trick' achieved
            # by concatenating the square roots in a larger array and
            # computing the QR factoriation, which computes the square
            # root of the sum of squares.  The covariance matrix isn't
            # formed explicitly in this code except at the very end when
            # it's time to return the covariance matrices to the user.

            m = torch.cat((a_scale_t, b_o, c_hbar), dim=1)

            # Unfortunately there's no QL factorization in PyTorch so we
            # transpose m and use the QR.  We only need the 'R' return
            # value, so the Q return value is dropped.

            scale_t_T = torch.linalg.qr(m.T, mode="reduced")[1]

            # Transpose ht to get the lower triangular version.

            scale_t = make_diagonal_nonnegative(scale_t_T.T)

        scale = torch.stack(scale_sequence)
        return scale, scale_t

    def __mean_log_likelihood(
        self, observations: torch.Tensor, uv_scale: Union[torch.Tensor, None] = None
    ):
        """
        This computes the mean per-sample log likelihood (the total log likelihood divided by the number of samples).
        """
        # We pass uv_scale into this function rather than computing it
        # here because __mean_log_likelihood() is called in a training
        # loop and the univariate parameters are held constant while
        # the multivariate parameters are trained.  In other words,
        # uv_scale is constant through the optimization and we'd be
        # computing it on every iteration if we computed it here.
        # Similarly _predict() doesn't know about the univariate model
        # since it sees only scaled observations.  Clients should use only
        # mean_log_likelihood() and predict() which are more intuitive.

        if uv_scale is not None:
            scaled_observations = observations / uv_scale
        else:
            scaled_observations = observations

        mv_scale = self._predict(scaled_observations)[0]

        # It's important to use non-scaled observations in likelihood function
        mean_ll = joint_conditional_log_likelihood(
            observations,
            mv_scale=mv_scale,
            uv_scale=uv_scale,
            distribution=self.distribution,
        )

        return mean_ll

    def fit(self, observations: torch.Tensor):
        self.univariate_model.fit(observations)
        sigma_est, mu = self.univariate_model.predict(observations)[:2]
        mean_sigma_est = torch.mean(sigma_est, dim=0)

        n = observations.shape[1]
        self.initialize_parameters(n)

        centered_observations = observations - mu

        self.sample_mean_scale = (
            torch.linalg.qr(centered_observations, mode="reduced")[1]
        ).T / torch.sqrt(torch.tensor(centered_observations.shape[0]))
        self.sample_mean_scale = make_diagonal_nonnegative(self.sample_mean_scale)

        self.sample_mean_scale = (
            torch.diag(mean_sigma_est**-1) @ self.sample_mean_scale
        )

        logging.info(f"sample_mean_scale:\n{self.sample_mean_scale}")

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

            # Do not use scaled observations here; centering is okay.
            loss = -self.__mean_log_likelihood(
                centered_observations, uv_scale=sigma_est
            )
            loss.backward()

            return loss

        optimize(optim, loss_closure, "multivariate model")

        self.log_parameters()

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
            (
                uv_scale,
                mu,
                uv_scale_next,
                mu_next,
            ) = self.univariate_model.predict(observations)
            centered_observations = observations - mu
            normalized_mv_scale, normalized_mv_scale_next = self._predict(
                centered_observations / uv_scale
            )
            mv_scale = (
                uv_scale.unsqueeze(2).expand(normalized_mv_scale.shape)
                * normalized_mv_scale
            )
            mv_scale_next = (
                uv_scale_next.unsqueeze(1).expand(normalized_mv_scale_next.shape)
                * normalized_mv_scale_next
            )

        return mv_scale, uv_scale, mu, mv_scale_next, uv_scale_next, mu_next

    def sample(
        self,
        n: Union[torch.Tensor, int],
        initial_mv_scale: Union[torch.Tensor, None],
        initial_uv_scale: Union[torch.Tensor, None] = None,
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

        with torch.no_grad():
            if isinstance(n, int):
                n = torch.randn(n, self.n)

            mv_scale, mu = self._predict(
                n, sample=True, initial_scale=initial_mv_scale
            )[:2]

            output = (mv_scale @ n.unsqueeze(2)).squeeze(2)

            output, uv_scale, mu = self.univariate_model.sample(
                output, initial_uv_scale
            )

        return output, mv_scale, uv_scale, mu

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
            uv_scale = self.univariate_model.predict(observations)[0]
            print(uv_scale)

            result = self.__mean_log_likelihood(observations, uv_scale)

        return float(result)

    def log_parameters(self):
        if self.a and self.b and self.c and self.d:
            logging.info(
                "Multivariate ARCH model\n"
                f"a: {self.a.value.detach().numpy()},\n"
                f"b: {self.b.value.detach().numpy()},\n"
                f"c: {self.c.value.detach().numpy()},\n"
                f"d: {self.d.value.detach().numpy()}"
            )
        else:
            logging.info("Multivariate ARCH model has no initialized parameters")


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
        initial_scale=[[0.008, 0.0, 0.0], [0.008, 0.01, 0.0], [0.008, 0.009, 0.005]],
    )

    mv_x, mv_scale, uv_scale, mu = multivariate_model.sample(
        50000, [[0.008, 0.0, 0.0], [0.008, 0.01, 0.0], [0.008, 0.009, 0.005]]
    )
    multivariate_model.fit(mv_x)
