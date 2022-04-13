import torch
import torch.nn

# Mixture tuning parametrers
DEFAULT_MIXTURE_FEATURES = 20  # May be overrideen by caller
DEFAULT_MIXTURE_COMPONENTS = 4  # May be overridden by caller

DEFAULT_GAUSSIAN_NOISE = 0.0025
DEFAULT_DROPOUT_P = 0.125
DEFAULT_BATCH_EPS = 1e-4

MIXTURE_MU_CLAMP = 0.10  # Clamp will be +/- this value
SIGMA_INV_CLAMP = 1000.0

logsoftmax = torch.nn.LogSoftmax(dim=1)

# Instances of various activation functions for convenience
relu = torch.nn.ReLU()
softplus = torch.nn.Softplus()
sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()


def batch_norm_factory_1d(feature_dimension, use_batch_norm):
    """Generate a batchnorm layer or generate a null layer as appropriate"""
    if use_batch_norm:
        return torch.nn.BatchNorm1d(feature_dimension, eps=DEFAULT_BATCH_EPS)
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

    def forward(self, input_tensor):
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

    def forward(self, input_tensor):
        if self.training:
            return input_tensor + self.sigma * torch.randn(input_tensor.shape).to(
                input_tensor.device
            )
        return input_tensor


class TimeSeriesFeatures(torch.nn.Module):
    """
    TODO
    """

    @property
    def window_size(self):
        return int(self._window_size)

    def __init__(
        self,
        input_channels: int,
        window_size: int,
        feature_dimension: int = DEFAULT_MIXTURE_FEATURES,
        exogenous_dimension: int = 0,
        gaussian_noise: float = DEFAULT_GAUSSIAN_NOISE,
        activation=relu,
        dropout: float = DEFAULT_DROPOUT_P,
        use_batch_norm: bool = True,
        extra_mixing_layers=0,
    ):
        super().__init__()

        if window_size == 0 and exogenous_dimension == 0:
            raise ValueError("window_size and exogenous_dimension cannot both be zero.")

        self._window_size = torch.nn.parameter.Parameter(
            torch.tensor(window_size), requires_grad=False
        )

        def conv_block(input_channels, width):
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

    def forward(self, window, exogenous=None):
        """
        Argument:
           context: (minibatch_size, channels, window_size)
        Returns:
           latents - (minibatch_size, feature_dimension)
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
    TODO
    """

    def __init__(
        self,
        input_channels,
        output_channels=None,
        feature_dimension=DEFAULT_MIXTURE_FEATURES,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
    ):
        super().__init__()

        if output_channels is None:
            output_channels = input_channels

        if input_channels != 1 or output_channels != 1:
            raise ValueError(
                "UnivariateHead requires input_channels == output_channels == 1"
            )

        self.p_head = torch.nn.Linear(feature_dimension, mixture_components)
        self.mu_head = torch.nn.Linear(feature_dimension, mixture_components)
        self.sigma_inv_head = torch.nn.Linear(feature_dimension, mixture_components)

    def forward(self, latents):
        """
        Argument:
           latents: (minibatch_size, feature_dimension)
        Returns:
           log_p: (minibatch_size, components)
           mu: (minibatch_size, components, output_symbols)
           sigma_inv: (minibatch_size, components, output_symbols, input_symbols)

        """
        # The unsqueeze() calls are required to maintain dimensions that comform
        # with the multivarate case.  In the multivate case, mu is a vector
        # (with dimension equal to the number of symbols) and sigma_inv is a
        # matrix (with row and colum dimensions equal to the number of symbols)
        log_p = logsoftmax(self.p_head(latents))
        mu = self.mu_head(latents).unsqueeze(2)
        sigma_inv = self.sigma_inv_head(latents).unsqueeze(2).unsqueeze(3)

        return log_p, mu, sigma_inv


class MultivariateHead(torch.nn.Module):
    """
    TODO
    """

    def __init__(
        self,
        input_channels,
        output_channels=None,
        feature_dimension=DEFAULT_MIXTURE_FEATURES,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
    ):
        super().__init__()

        if output_channels is None:
            output_channels = input_channels

        self.p_head = torch.nn.Linear(feature_dimension, mixture_components)

        self.mu_head = torch.nn.ConvTranspose1d(
            feature_dimension, mixture_components, output_channels
        )
        # It seems odd here to use "channels" as the matrix dimension,
        # but that's exactly what we want.  The number of input
        # channels is the number of time series.  Here we want a
        # square covariance matrix of the same dimension as the
        # output.
        self.sigma_inv_head = torch.nn.ConvTranspose2d(
            feature_dimension,
            mixture_components,
            (output_channels, input_channels),
        )

    def forward(self, latents):
        """
        Argument:
           latents: (minibatch_size, feature_dimension)
        Returns:
           log_p: (minibatch_size, components)
           mu: (minibatch_size, components, output_symbols)
           sigma_inv: (minibatch_size, components, output_symbols, input_symbols)

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
        output_channels, input_channels = sigma_inv.shape[2:]
        sigma_inv = torch.tril(sigma_inv, diagonal=(input_channels - output_channels))

        return log_p, mu, sigma_inv


class MixtureModel(torch.nn.Module):
    """ """

    @property
    def window_size(self):
        return self.time_series_features.window_size

    def __init__(
        self,
        window_size,
        input_channels,
        output_channels=None,
        output_head_type=UnivariateHead,
        feature_dimension=DEFAULT_MIXTURE_FEATURES,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
        exogenous_dimension=0,
        extra_mixing_layers=0,
        gaussian_noise=DEFAULT_GAUSSIAN_NOISE,
        activation=relu,
        dropout=DEFAULT_DROPOUT_P,
        use_batch_norm=True,
    ):
        super().__init__()

        self.time_series_features = TimeSeriesFeatures(
            input_channels,
            window_size,
            feature_dimension=feature_dimension,
            exogenous_dimension=exogenous_dimension,
            extra_mixing_layers=extra_mixing_layers,
            gaussian_noise=gaussian_noise,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

        self.head = output_head_type(
            input_channels,
            output_channels=output_channels,
            feature_dimension=feature_dimension,
            mixture_components=mixture_components,
        )

    def forward_unpacked(self, time_series, embedding=None):
        """
        Argument:
           context: (minibatch_size, channels, window_size)
        Returns:
           log_p_raw: (minibatch_size, components)
           mu: (minibatch_size, components, channels)
           sigma_inv: (minibatch_size, components, channels, channels)
        Notes:
           log_p_raw is "raw" in the sense in that the caller must apply a
           softmax to it to produce probabilities.

        """
        # Get a "flat" feature vector for the series.
        latents = self.time_series_features(time_series, embedding)

        log_p, mu, sigma_inv = self.head(latents)

        if not self.training:
            mu = torch.clamp(mu, -MIXTURE_MU_CLAMP, MIXTURE_MU_CLAMP)
            sigma_inv = torch.clamp(sigma_inv, -SIGMA_INV_CLAMP, SIGMA_INV_CLAMP)

        return log_p, mu, sigma_inv, latents

    def forward(self, predictors):
        """This is a wrapper for the forward_unpacked() method.  It assumes that
        data is a tuple of (time_series, embedding).  In the case that data is
        not a tuple, it is assumed to be the time_series portion.
        """
        # Allow this to work when passed a tensor. In that case, assume the
        # intent to be that there are no exogenous inputs.
        if not isinstance(predictors, tuple):
            predictors = (predictors, None)

        return self.forward_unpacked(*predictors)
