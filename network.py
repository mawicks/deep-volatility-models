from this import d
from matplotlib import use
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
        return self._window_size

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
    ):
        super().__init__()

        self._window_size = window_size

        layers = [
            MinMaxClamping(),
            GaussianNoise(gaussian_noise),
        ]

        def conv_block(input_channels, width):
            return [
                torch.nn.Conv1d(
                    input_channels, feature_dimension, width, stride=width, padding=0
                ),
                batch_norm_factory_1d(feature_dimension, use_batch_norm),
                activation,
                torch.nn.Dropout2d(p=dropout),
            ]

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

        self.blend_exogenous = torch.nn.Sequential(
            torch.nn.Linear(
                feature_dimension + exogenous_dimension,
                feature_dimension,
            ),
            activation,
        )

    def forward(self, window, exogenous=None):
        """
        Argument:
           context: (minibatch_size, channels, window_size)
        Returns:
           latents - (minibatch_size, feature_dimension)
        """

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

    def __dimensions(self):
        features_dimension = self.sigma_inv_head.in_channels
        components = self.sigma_inv_head.out_channels
        output_channels = input_channels = 1
        return features_dimension, components, output_channels, input_channels

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

        return log_p, mu, sigma_inv, latents


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

    def __dimensions(self):
        features_dimension = self.sigma_inv_output.in_channels
        components = self.sigma_inv_output.out_channels
        output_channels, input_channels = self.sigma_inv_output.kernel_size
        return features_dimension, components, output_channels, input_channels

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

        return log_p, mu, sigma_inv, latents


class MixtureModel(torch.nn.Module):
    """
    Arguments:
       context (tensor of dim 64)
       value (tensor of dim 1)

    Returns:
       log odds of prediction (tensor of dim 1)
    """

    @property
    def context_size(self):
        return 64

    def __init__(
        self,
        input_channels,
        output_channels=None,
        output_head_type=UnivariateHead,
        feature_dimension=DEFAULT_MIXTURE_FEATURES,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
        exogenous_dimension=0,
        gaussian_noise=DEFAULT_GAUSSIAN_NOISE,
        activation=relu,
        dropout=DEFAULT_DROPOUT_P,
        use_batch_norm=True,
    ):
        super().__init__()

        if output_channels is None:
            output_channels = input_channels

        self.time_series_features = TimeSeriesFeatures(
            input_channels,
            64,
            feature_dimension=feature_dimension,
            exogenous_dimension=exogenous_dimension,
            gaussian_noise=gaussian_noise,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        self.head = output_head_type(
            input_channels,
            output_channels=output_channels,
            feature_dimension=feature_dimension,
            mixture_components=mixture_components,
        )

    def __dimensions(self):
        features_dimension = self.sigma_inv_output.in_channels
        components = self.sigma_inv_output.out_channels
        output_channels, input_channels = self.sigma_inv_output.kernel_size
        return features_dimension, components, output_channels, input_channels

    def forward(self, context, embedding=None):
        """
        Argument:
           context: (minibatch_size, channels, 64)
        Returns:
           log_p_raw: (minibatch_size, components)
           mu: (minibatch_size, components, channels)
           sigma_inv: (minibatch_size, components, channels, channels)
        Notes:
           log_p_raw is "raw" in the sense in that the caller must apply a
           softmax to it to produce probabilities.

        """
        # Get a "flat" feature vector for the series.
        latents = self.time_series_features(context, embedding)

        log_p_raw, mu, sigma_inv, latents = self.head(latents)

        if not self.training:
            mu = torch.clamp(mu, -MIXTURE_MU_CLAMP, MIXTURE_MU_CLAMP)
            sigma_inv = torch.clamp(sigma_inv, -SIGMA_INV_CLAMP, SIGMA_INV_CLAMP)

        return log_p_raw, mu, sigma_inv, latents
