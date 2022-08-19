# Standard Python
from abc import abstractmethod
import logging
from typing import Any, Dict, List, Protocol, Tuple, Union

# Common packages
import torch

# Local modules
from . import constants
from .parameters import (
    Parameter,
    ScalarParameter,
    DiagonalParameter,
    TriangularParameter,
    FullParameter,
)


class MeanModel(Protocol):
    @abstractmethod
    def initialize_parameters(self, observations: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def log_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict(
        self,
        observations: torch.Tensor,
        sample: bool = False,
        mean_initial_value: Union[torch.Tensor, Any, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` are scaled zero-mean noise
                           rather than actual observations.
            mean_initial_value: torch.Tensor (or something convertible to one)
                          Initial mean vector if specified
        Returns:
            mu: torch.Tensor of predictions for each observation
            mu_next: torch.Tensor prediction for next unobserved value

        """
        raise NotImplementedError

    @torch.no_grad()
    def predict(
        self, observations: torch.Tensor, initial_mean=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is the inference version of predict(), which is the version clients would normally use.
        It doesn't compute any gradient information, so it should be faster.
        """
        return self._predict(observations, initial_mean)

    @torch.no_grad()
    def sample(
        self,
        scaled_zero_mean_noise: torch.Tensor,
        initial_mean: Union[torch.Tensor, None],
    ) -> torch.Tensor:
        # mu = self.__predict(scaled_zero_mean_noise, sample=True, initial_mean=initial_mean)
        # return mu
        raise Exception("sample() called.")


class ZeroMeanModel(MeanModel):
    def __init__(
        self,
        device: Union[torch.device, None] = None,
    ):
        self.device = device

    def initialize_parameters(self, observations: torch.Tensor) -> None:
        self.n = observations.shape[1]

    def set_parameters(self, **kwargs: Any) -> None:
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        return []

    def log_parameters(self) -> None:
        pass

    def _predict(
        self,
        observations: torch.Tensor,
        sample: bool = False,
        mean_initial_value: Union[torch.Tensor, Any, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` are scaled zero-mean noise
                           rather than actual observations.
            mean_initial_value: torch.Tensor (or something convertible to one)
                                Ignored for ZeroMeanModel
        Returns:
            mu: torch.Tensor of predictions for each observation
            mu_next: torch.Tensor prediction for next unobserved value

        """
        mu = torch.zeros(observations.shape, dtype=torch.float, device=self.device)
        mu_next = torch.zeros(
            observations.shape[1], dtype=torch.float, device=self.device
        )
        return mu, mu_next


class ARMAMeanModel(MeanModel):
    n: Union[int, None]
    a: Union[Parameter, None]
    b: Union[Parameter, None]
    c: Union[Parameter, None]
    d: Union[Parameter, None]
    sample_mean: Union[torch.Tensor, None]
    device: Union[torch.device, None]

    def __init__(
        self,
        device: Union[torch.device, None] = None,
    ):
        self.n = self.a = self.b = self.c = self.d = None
        self.sample_mean = None
        self.device = device

    def initialize_parameters(self, observations: torch.Tensor) -> None:
        self.n = observations.shape[1]
        self.a = DiagonalParameter(
            self.n, 1.0 - constants.INITIAL_DECAY, device=self.device
        )
        self.b = DiagonalParameter(self.n, constants.INITIAL_DECAY, device=self.device)
        self.c = DiagonalParameter(self.n, 1.0, device=self.device)
        self.d = DiagonalParameter(self.n, 1.0, device=self.device)
        self.sample_mean = torch.mean(observations, dim=0)

    def set_parameters(self, **kwargs: Any) -> None:
        a = kwargs["a"]
        b = kwargs["b"]
        c = kwargs["c"]
        d = kwargs["d"]
        sample_mean = kwargs["sample_mean"]

        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float, device=self.device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float, device=self.device)
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(d, dtype=torch.float, device=self.device)
        if not isinstance(sample_mean, torch.Tensor):
            sample_mean = torch.tensor(
                sample_mean, dtype=torch.float, device=self.device
            )

        if (
            len(a.shape) != 1
            or a.shape != b.shape
            or a.shape != c.shape
            or a.shape != d.shape
            or a.shape != sample_mean.shape
        ):
            raise ValueError(
                f"The shapes of a({a.shape}), b({b.shape}), "
                f"c({c.shape}), d({d.shape}), and "
                f"sample mean({sample_mean.shape}) must have "
                "only and only one dimension that's consistent"
            )

        self.n = a.shape[0]
        if isinstance(self.n, int):
            self.a = DiagonalParameter(self.n)
            self.b = DiagonalParameter(self.n)
            self.c = DiagonalParameter(self.n)
            self.d = DiagonalParameter(self.n)

            self.a.set(a)
            self.b.set(b)
            self.c.set(c)
            self.d.set(d)

        self.sample_mean = sample_mean

    def get_parameters(self) -> Dict[str, Any]:
        safe_value = lambda x: x.value if x is not None else None
        return {
            "a": safe_value(self.a),
            "b": safe_value(self.b),
            "c": safe_value(self.c),
            "d": safe_value(self.d),
            "n": self.n,
            "sample_mean": self.sample_mean,
        }

    def get_optimizable_parameters(self) -> List[torch.Tensor]:
        if self.a is None or self.b is None or self.c is None or self.d is None:
            raise ValueError("ARMAMeanModel parameters have not been initialized")
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

    def _predict(
        self,
        observations: torch.Tensor,
        sample=False,
        initial_mean: Union[torch.Tensor, Any, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given a, b, c, d, and observations, generate the *estimated*
        standard deviations (marginal) for each observation

        Argument:
            observations: torch.Tensor of dimension (n_obs, n_symbols)
                          of observations
            sample: bool - Run the model in 'sampling' mode, in which
                           case `observations` are scaled zero-mean noise
                           rather than actual observations.
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
            # Store the current mu_t before predicting next one
            mu_sequence.append(mu_t)

            # While searching over the parameter space, an unstable value for `a` may be tested.
            # Clamp to prevent it from overflowing.
            a_mu = torch.clamp(
                self.a @ mu_t, min=constants.MIN_CLAMP, max=constants.MAX_CLAMP
            )

            if sample:
                obs = obs + mu_t

            b_o = self.b @ obs
            c_sample_mean = self.c @ self.sample_mean  # type: ignore

            mu_t = a_mu + b_o + c_sample_mean

        mu = torch.stack(mu_sequence)
        return mu, mu_t


if __name__ == "__main__":
    zmm = ZeroMeanModel()

    amm = ARMAMeanModel()
