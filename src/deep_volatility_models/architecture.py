# Standard importors
from typing import Any, Callable, Tuple, Union

# Third party modules
import torch
import torch.nn

# Local imports
from deep_volatility_models import mixture_model_stats
from deep_volatility_models import sample

# Instances of various activation functions for convenience
relu = torch.nn.ReLU()
softplus = torch.nn.Softplus()
sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()

# Default values for Tuning parametrers
DEFAULT_FEATURE_DIMENSION = 20  # May be overrideen by caller
DEFAULT_MIXTURE_COMPONENTS = 4  # May be overridden by caller
DEFAULT_GAUSSIAN_NOISE = 0.0025
DEFAULT_DROPOUT_P = 0.125
DEFAULT_ACTIVATION_FUNCTION = relu

BATCH_NORM_EPS = 1e-4
MIXTURE_MU_CLAMP = 0.10  # Clamp will be +/- this value
SIGMA_INV_CLAMP = 1000.0

logsoftmax = torch.nn.LogSoftmax(dim=1)


def batch_norm_factory_1d(
    feature_dimension: int, use_batch_norm: bool
) -> torch.nn.Module:
    """Generate a batchnorm layer or generate a null layer as appropriate"""
    if use_batch_norm:
        return torch.nn.BatchNorm1d(feature_dimension, eps=BATCH_NORM_EPS)
    else:
        return torch.nn.Sequential()


class MinMaxClamping(torch.nn.Module):
    """
    TODO
    """

    def __init__(self):
        super().__init__()
        self.training_max = None
        self.training_min = None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_max = torch.max(input_tensor, dim=0)[0].detach()
            batch_min = torch.min(input_tensor, dim=0)[0].detach()
            self.training_max = (
                torch.max(batch_max, self.training_max)
                if self.training_max is not None
                else batch_max
            )
            self.training_min = (
                torch.min(batch_min, self.training_min)
                if self.training_min is not None
                else batch_min
            )
            result = input_tensor
        else:
            if self.training_max is not None and self.training_min is not None:
                result = torch.max(
                    torch.min(input_tensor, self.training_max), self.training_min
                )
            else:
                result = input_tensor
        return result


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.training:
            return input_tensor + self.sigma * torch.randn(input_tensor.shape).to(
                input_tensor.device
            )
        return input_tensor


class TimeSeriesFeatures(torch.nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        input_channels: int,
        window_size: int,
        exogenous_dimension: int = 0,
        feature_dimension: int = DEFAULT_FEATURE_DIMENSION,
        activation: torch.nn.Module = DEFAULT_ACTIVATION_FUNCTION,
        gaussian_noise: float = DEFAULT_GAUSSIAN_NOISE,
        dropout: float = DEFAULT_DROPOUT_P,
        use_batch_norm: bool = True,
        extra_mixing_layers: int = 0,
    ):
        super().__init__()

        if window_size == 0 and exogenous_dimension == 0:
            raise ValueError("window_size and exogenous_dimension cannot both be zero.")

        self._window_size = torch.nn.parameter.Parameter(
            torch.tensor(window_size), requires_grad=False
        )

        def conv_block(input_channels: int, width: int):
            return [
                torch.nn.Conv1d(
                    input_channels,
                    feature_dimension,
                    width,
                    stride=width,
                    padding=0,
                ),
                batch_norm_factory_1d(feature_dimension, use_batch_norm),
                activation,
                torch.nn.Dropout2d(p=dropout),
            ]

        layers = []
        # If window_size is 0 (which we allow), we leave the sequential block empty.
        if window_size > 0:
            layers.extend(
                [
                    MinMaxClamping(),
                    GaussianNoise(gaussian_noise),
                ]
            )

            layers.extend(conv_block(input_channels, 4))
            window_size //= 4

            while window_size > 1:
                layers.extend(conv_block(feature_dimension, 4))
                window_size //= 4

            if window_size != 1:
                raise ValueError("window_size must be a power of 4")
            # Should have one "pixel" with a depth of feature_dimension

            # Do one more mixing layer.
            layers.extend(conv_block(feature_dimension, 1))

        self.convolutional_layers = torch.nn.Sequential(*layers)

        blend_exogenous_layers = [
            torch.nn.Linear(
                feature_dimension + exogenous_dimension
                if window_size > 0
                else exogenous_dimension,
                feature_dimension,
            ),
            batch_norm_factory_1d(feature_dimension, use_batch_norm),
            activation,
            torch.nn.Dropout(dropout),
        ]

        # Following block doesn't show improvement, so we'll probably remove it
        # eventualy
        for _ in range(extra_mixing_layers):
            blend_exogenous_layers.extend(
                [
                    torch.nn.Linear(
                        feature_dimension,
                        feature_dimension,
                    ),
                    batch_norm_factory_1d(feature_dimension, use_batch_norm),
                    activation,
                    torch.nn.Dropout(dropout),
                ]
            )

        self.blend_exogenous = torch.nn.Sequential(*blend_exogenous_layers)

    @property
    def window_size(self) -> int:
        return int(self._window_size)

    def forward(
        self, window: torch.Tensor, exogenous: Union[torch.Tensor, None] = None
    ):
        """
        Argument:
           window: torch.Tensor of shape (minibatch_size, channels, window_size)
           exogenous: torch.Tensor to be mixed in or None
        Returns:
           latents: torch.Tensor of shape (minibatch_size, feature_dimension)
        """
        batch_size = window.shape[0]

        if self.window_size > 0:
            time_series_features = self.convolutional_layers(window)

            # The dimension of `time_series_features`` is (batch,
            # feature_dimensions, 1).  We'll adopt the convention that this network
            # produces a flattened feature vector (i.e., not a series), so we remove
            # the last dimension.  In some cases, the caller may want to add it back
            # when additional convolutional processing is useful.  Also, removing
            # the final dimension makes its dimension conform with the exogenous
            # inputs (which `time_series_features` will be combined with), which do
            # not have this extra dimension.

            time_series_features = time_series_features.squeeze(2)
        else:
            # window should be empty (size 0), so we can reshape it to another
            # empty tensor with the correct dimensions so that the cat below works.
            time_series_features = window.reshape((batch_size, 0))

        # Concatenate time series feature vector with exogenous features (if
        # provided) and add another mixing layer to allow them to interact

        if exogenous is not None:
            output = torch.cat((time_series_features, exogenous), dim=1)
            output = self.blend_exogenous(output)
        else:
            output = time_series_features

        return output


class UnivariateHead(torch.nn.Module):
    """
    Univariate, non-mnixture head.
    """

    def __init__(
        self,
        feature_dimension: int,
    ):
        super().__init__()

        # mu_head turns feature vector into a single mu estiamte.
        self.mu_head = torch.nn.Linear(feature_dimension, 1)

        # sigma_inv_head turns feature vectors into a single 1/sigma estimate
        self.sigma_inv_head = torch.nn.Linear(feature_dimension, 1)

    def forward(self, latents: torch.Tensor):
        """
        Argument:
           latents: torch.Tensor of shape (minibatch_size, feature_dimension)
        Returns:
           mu: torch.Tensor of shape (minibatch_size, output_symbols=1)
           sigma_inv: torch.Tensor of shape (minibatch_size, output_symbols=1, input_symbols=1)

        """
        mu = self.mu_head(latents)
        sigma_inv = self.sigma_inv_head(latents)

        # The unsqueeze() call is required to maintain dimensions that comform
        # with the multivarate case.  In the multivate case, sigma_inv is a
        # matrix (with row and colum dimensions equal to the number of symbols)

        sigma_inv = sigma_inv.unsqueeze(2)

        return mu, sigma_inv


class UnivariateMixtureHead(torch.nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        input_symbols: int,
        output_symbols: Union[int, None],
        feature_dimension: int,
        mixture_components: int,
    ):
        super().__init__()

        if output_symbols is None:
            output_symbols = input_symbols

        if input_symbols != 1 or output_symbols != 1:
            raise ValueError(
                "UnivariateMixtureHead requires input_symbols == output_symbols == 1"
            )

        self.p_head = torch.nn.Linear(feature_dimension, mixture_components)
        self.mu_head = torch.nn.Linear(feature_dimension, mixture_components)
        self.sigma_inv_head = torch.nn.Linear(feature_dimension, mixture_components)

    def forward(self, latents: torch.Tensor):
        """
        Argument:
           latents: torch.Tensor of shape (minibatch_size, feature_dimension)
        Returns:
           log_p: torch.Tensor of shape (minibatch_size, components)
           mu: torch.Tensor of shape (minibatch_size, components, output_symbols)
           sigma_inv: torch.Tensor of shape (minibatch_size, components, output_symbols, input_symbols)

        """
        # The unsqueeze() calls are required to maintain dimensions that comform
        # with the multivarate case.  In the multivate case, mu is a vector
        # (with dimension equal to the number of symbols) and sigma_inv is a
        # matrix (with row and colum dimensions equal to the number of symbols)
        log_p = logsoftmax(self.p_head(latents))
        mu = self.mu_head(latents).unsqueeze(2)
        sigma_inv = self.sigma_inv_head(latents).unsqueeze(2).unsqueeze(3)

        return log_p, mu, sigma_inv


class MultivariateMixtureHead(torch.nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        input_symbols: int,
        output_symbols: Union[int, None],
        feature_dimension: int,
        mixture_components: int,
    ):
        super().__init__()

        if output_symbols is None:
            output_symbols = input_symbols

        self.p_head = torch.nn.Linear(feature_dimension, mixture_components)

        self.mu_head = torch.nn.ConvTranspose1d(
            feature_dimension, mixture_components, output_symbols
        )
        # It seems odd here to use "channels" as the matrix dimension,
        # but that's exactly what we want.  The number of input
        # channels is the number of time series.  Here we want a
        # square covariance matrix of the same dimension as the
        # output.
        self.sigma_inv_head = torch.nn.ConvTranspose2d(
            feature_dimension,
            mixture_components,
            (output_symbols, input_symbols),
        )

    def forward(self, latents: torch.Tensor):
        """
        Argument:
           latents: torch.Tensor of shape (minibatch_size, feature_dimension)
        Returns:
           log_p: torch.Tensor of shape (minibatch_size, components)
           mu: torch.Tensor fo shape (minibatch_size, components, output_symbols)
           sigma_inv: torch.Tensor fo shape (minibatch_size, components, output_symbols, input_symbols)

        """
        log_p = logsoftmax(self.p_head(latents))

        # The network for mu uses a 1d one-pixel de-convolutional layer which
        # require the input to be sequence-like.  Create an artificial sequence
        # dimension before calling.
        latents_1d = latents.unsqueeze(2)
        mu = self.mu_head(latents_1d)

        # The network for sigma_inv uses a 2d one-pixel de-convolutional layer
        # which requires the input to be image-like.  Create artificial x and y
        # dimensions before calling.
        latents_2d = latents_1d.unsqueeze(3)
        sigma_inv = self.sigma_inv_head(latents_2d)

        # FIXME:  For compatibility with previously saved models
        # we get the shape from sigma_inv rather than from object state.
        output_symbols, input_symbols = sigma_inv.shape[2:]
        sigma_inv = torch.tril(sigma_inv, diagonal=(input_symbols - output_symbols))

        return log_p, mu, sigma_inv


def risk_neutral_drift(mu, sigma_inv):
    if sigma_inv.shape[1] != 1 or sigma_inv.shape[2] != 1:
        raise ValueError(
            f"risk_neutral_drift() requires last two dimensions of sigma_inv to be 1, but shape is {sigma_inv.shape}"
        )
    if mu.shape[1] != 1:
        raise ValueError(
            f"risk_neutral_drift() requires last dimension of mu to be 1, but shape is {mu.shape}"
        )
    if mu.shape[0] != sigma_inv.shape[0]:
        raise ValueError(
            f"mu and sigma_inv need same number of rows but shapes are {mu.shape} and {sigma_inv.shape}"
        )

    variance = sigma_inv ** (-2)
    return -0.5 * variance.squeeze(2)


class UnivariateModel(torch.nn.Module):
    """Univariate model that's not a mixture model"""

    def __init__(
        self,
        window_size: int,
        output_head_factory: Callable[
            [int],
            torch.nn.Module,
        ] = UnivariateHead,  # head_factory parameter is feature_dimension
        feature_dimension: int = DEFAULT_FEATURE_DIMENSION,
        exogenous_dimension: int = 0,
        extra_mixing_layers: int = 0,
        gaussian_noise: float = DEFAULT_GAUSSIAN_NOISE,
        activation: torch.nn.Module = relu,
        dropout: float = DEFAULT_DROPOUT_P,
        use_batch_norm: bool = True,
        risk_neutral=True,
        mixture_components=None,  # FIXME: This parameter should be removed.
    ):
        super().__init__()

        self.time_series_features = TimeSeriesFeatures(
            1,  # input symbols
            window_size,
            feature_dimension=feature_dimension,
            exogenous_dimension=exogenous_dimension,
            extra_mixing_layers=extra_mixing_layers,
            gaussian_noise=gaussian_noise,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        self.head = output_head_factory(
            feature_dimension,
        )

        self.risk_neutral = risk_neutral

    @property
    def sampler(self):
        return sample.multivariate_sample

    def simulate_one(
        self,
        predictors: Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, None]]],
        time_samples: int,
    ) -> torch.Tensor:
        return sample.simulate_one(self, predictors, time_samples)

    @property
    def is_mixture(self) -> bool:
        return False

    @property
    def window_size(self) -> int:
        return self.time_series_features.window_size

    def forward_unpacked(
        self, window: torch.Tensor, exogenous: Union[torch.Tensor, None] = None
    ):
        """
        Argument:
           windowt: torch.Tensor of shape (minibatch_size, channels,
           window_size)
           exogenous: torch.Tensor to be mixed in or None
        Returns:
           mu: torch.Tensor of shape (minibatch_size, components, symbols=1)
           sigma_inv: (minibatch_size, components, input_symbols=1, output_symbols=1)

        """
        # Get a "flat" feature vector for the series.
        latents = self.time_series_features(window, exogenous)

        mu, sigma_inv = self.head(latents)

        if not self.training:
            mu = torch.clamp(mu, -MIXTURE_MU_CLAMP, MIXTURE_MU_CLAMP)
            sigma_inv = torch.clamp(sigma_inv, -SIGMA_INV_CLAMP, SIGMA_INV_CLAMP)

        return mu, sigma_inv, latents

    def forward(
        self,
        predictors: Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, None]]],
    ):
        """This is a wrapper for the forward_unpacked() method.  It assumes that
        data is a tuple of (time_series, embedding).  In the case that data is
        not a tuple, it is assumed to be the time_series portion.
        """
        # Allow this to work when passed a tensor. In that case, assume the
        # intent to be that there are no exogenous inputs.
        if not isinstance(predictors, tuple):
            predictors = (predictors, None)

        mu, sigma_inv, latents = self.forward_unpacked(*predictors)

        if hasattr(self, "risk_neutral") and self.risk_neutral:
            mu = risk_neutral_drift(mu, sigma_inv)

        return mu, sigma_inv, latents


def mixture_risk_neutral_adjustment(log_p, mu, sigma_inv):
    p = torch.exp(log_p)
    mu_c, var_c = mixture_model_stats.univariate_combine_metrics(p, mu, sigma_inv)
    # log_mean_return is log of mean return as opposed to mean of log return.
    log_mean_return = mu_c + 0.5 * var_c
    log_mean_return = log_mean_return.unsqueeze(1).unsqueeze(2).expand(mu.shape)

    return mu - log_mean_return


class MixtureModel(torch.nn.Module):
    """TODO"""

    def __init__(
        self,
        window_size: int,
        input_symbols: int,
        output_symbols: Union[int, None] = None,
        output_head_factory: Callable[
            [int, Union[int, None], int, int], torch.nn.Module
        ] = UnivariateMixtureHead,
        mixture_components: int = DEFAULT_MIXTURE_COMPONENTS,
        feature_dimension: int = DEFAULT_FEATURE_DIMENSION,
        exogenous_dimension: int = 0,
        extra_mixing_layers: int = 0,
        gaussian_noise: float = DEFAULT_GAUSSIAN_NOISE,
        activation: torch.nn.Module = relu,
        dropout: float = DEFAULT_DROPOUT_P,
        use_batch_norm: bool = True,
        risk_neutral=True,
    ):
        super().__init__()

        self.time_series_features = TimeSeriesFeatures(
            input_symbols,
            window_size,
            feature_dimension=feature_dimension,
            exogenous_dimension=exogenous_dimension,
            extra_mixing_layers=extra_mixing_layers,
            gaussian_noise=gaussian_noise,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        self.head = output_head_factory(
            input_symbols,
            output_symbols,
            feature_dimension,
            mixture_components,
        )

        if risk_neutral and (input_symbols != 1 or output_symbols not in (1, None)):
            raise ValueError(
                f"Specifying risk_neutral is only possible with input_symbols == 1 and output_symbols in (1, None)"
                f"but input_symbosl={input_symbols} and output_symbols={output_symbols}"
            )
        self.risk_neutral = risk_neutral

    @property
    def is_mixture(self) -> bool:
        return True

    @property
    def window_size(self) -> int:
        return self.time_series_features.window_size

    @property
    def sampler(self):
        return sample.multivariate_mixture_sample

    def simulate_one(
        self,
        predictors: Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, None]]],
        time_samples: int,
    ) -> torch.Tensor:
        return sample.simulate_one(self, predictors, time_samples)

    def forward_unpacked(
        self, window: torch.Tensor, exogenous: Union[torch.Tensor, None] = None
    ):
        """
        Argument:
           windowt: torch.Tensor of shape (minibatch_size, channels,
           window_size)
           exogenous: torch.Tensor to be mixed in or None
        Returns:
           log_p_raw: torch.Tensor of shape (minibatch_size, components)
           mu: torch.Tensor of shape (minibatch_size, components, channels)
           sigma_inv: (minibatch_size, components, channels, channels)
        Notes:
           log_p_raw is "raw" in the sense in that the caller must apply a
           softmax to it to produce probabilities.

        """
        # Get a "flat" feature vector for the series.
        latents = self.time_series_features(window, exogenous)

        log_p, mu, sigma_inv = self.head(latents)

        if not self.training:
            mu = torch.clamp(mu, -MIXTURE_MU_CLAMP, MIXTURE_MU_CLAMP)
            sigma_inv = torch.clamp(sigma_inv, -SIGMA_INV_CLAMP, SIGMA_INV_CLAMP)

        return log_p, mu, sigma_inv, latents

    def forward(
        self,
        predictors: Union[torch.Tensor, Tuple[torch.Tensor, Union[torch.Tensor, None]]],
    ):
        """This is a wrapper for the forward_unpacked() method.  It assumes that
        data is a tuple of (time_series, embedding).  In the case that data is
        not a tuple, it is assumed to be the time_series portion.
        """
        # Allow this to work when passed a tensor. In that case, assume the
        # intent to be that there are no exogenous inputs.
        if not isinstance(predictors, tuple):
            predictors = (predictors, None)

        log_p, mu, sigma_inv, latents = self.forward_unpacked(*predictors)

        if hasattr(self, "risk_neutral") and self.risk_neutral:
            mu = mixture_risk_neutral_adjustment(log_p, mu, sigma_inv)

        return log_p, mu, sigma_inv, latents


class ModelWithEmbedding(torch.nn.Module):
    """
    This is a wrapper for a model.  The wrapper preprocesses an encoding into an
    embedding before calling the model.  This decouples the embedding from the
    training of the rest of the model.  The embedding can be replaced or
    retrained without changing anything in the model.  The caller is responsible
    for creating the model and the embedding. Here's an example:

    >>> import torch
    >>> import architecture
    >>> mixture_model = architecture.MixtureModel(64, 1, 1)
    >>> embedding = torch.nn.Embedding(20, 3)
    >>> combined_model = architecture.ModelWithEmbedding(mixture_model, embedding)

    At this point, the caller is free to manage (e.g., save, load, train,
    replace, etc.) both the model and the embedding as necesary.

    The mixture_model and the combined_model are similar except that the
    mixture_model accepts an embedding vector in the second position of the
    tuple passed to forward() whereas the combined_model accpets an integer encoding.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        embedding: torch.nn.Embedding,
    ):
        """
        Arguments:
            model: torch.nn.Module - The model to which embeddings are to be
            applied to an encoding on one of the inputs.  The model.forward()
            method is assumed to accept a tuple (Tuple[torch.Tensor,
            torch.Tensor] with the first position of the tuple being predictors
            other than the embedding and the second position of the tuple being
            the embedding.  embedding: torch.nn.Embedding - The trainable
            embedding.

            embedding: torch.nn.Embedding - The embedding to apply.
        """

        super().__init__()

        self.model = model
        self.embedding = embedding

    @property
    def sampler(self):
        return self.model.sample

    def simulate_one(
        self,
        predictors_plus_encoding: Tuple[torch.Tensor, torch.Tensor],
        time_samples: int,
    ) -> torch.Tensor:
        return sample.simulate_one(
            self,
            predictors_plus_encoding,
            time_samples,
        )

    @property
    def is_mixture(self):
        return self.model.is_mixture

    def forward(
        self, predictors_plus_encoding: Tuple[torch.Tensor, torch.Tensor]
    ) -> Any:
        """
        Arguments:
            predictors_plus_encoding: Tuple[torch.Tensor, torch.Tensor] - The first position
            of the tuple could be any predictor vector (though here it's
            typically a time series window) intended to be passed to
            convolutional layers in `self.model.`  The second position is an
            encoding to be converted to an embedding.  The model is called again
            with a tuple (Tuple[torch.Tensor, torch.Tensor]).  The predictors
            are passed unmodified in the first position of thetuple.  The
            embedding that replaces the encoding is passed in the second position.

        """
        predictors, encoding = predictors_plus_encoding
        embeddings = self.embedding(encoding)
        return self.model((predictors, embeddings))
