from this import d
from matplotlib import use
import torch.nn

# Mixture tuning parametrers
DEFAULT_MIXTURE_FEATURES = 20  # May be overrideen by caller
DEFAULT_MIXTURE_COMPONENTS = 4  # May be overridden by caller

DEFAULT_GAUSSIAN_NOISE = 0.0025  # 0.003  # 0.004
DEFAULT_DROPOUT_P = 0.125
DEFAULT_BATCH_EPS = 1e-4

MIXTURE_MU_CLAMP = 0.10  # Clamp will be +/- this value
SIGMA_INV_CLAMP = 1000.0
NEW_PROBABILITY_WEIGHT = 0.1

softmax = torch.nn.Softmax(dim=1)
relu = torch.nn.ReLU()
softplus = torch.nn.Softplus()
sigmoid = torch.nn.Sigmoid()
# logsoftmax = torch.nn.LogSoftmax(dim=1)
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
        return (self.window_size,)

    def __init__(
        self,
        input_channels: int,
        window_size: int,
        feature_dimension: int = DEFAULT_MIXTURE_FEATURES,
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

        def block(input_channels, width):
            return [
                torch.nn.Conv1d(
                    input_channels, feature_dimension, width, stride=width, padding=0
                ),
                batch_norm_factory_1d(feature_dimension, use_batch_norm),
                activation,
                torch.nn.Dropout2d(p=dropout),
            ]

        layers.extend(block(input_channels, 4))
        window_size //= 4

        while window_size > 1:
            layers.extend(block(feature_dimension, 4))
            window_size //= 4

        if window_size != 1:
            raise ValueError("window_size must be a power of 4")
        # Should have one "pixel" of depth feature_dimension

        # Do one more mixing layer.
        layers.extend(block(feature_dimension, 1))

        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, window):
        """
        Argument:
           context: (minibatch_size, channels, 64)
        Returns:
           latents - (minibatch_size, feature_dimension)
        """

        output = self.sequential(window)
        # The dimension of output is (batch, feature_dimensions, 1).  We'll
        # adopt the convention that this network produces a flattened feature
        # vector (not a series), so we remove the last dimension.  In some cases
        # caller may want to add it back if additional convolutional processing
        # is necessary.
        return output.squeeze(2)


class UnivariateMixture64(torch.nn.Module):
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
        feature_dimension=DEFAULT_MIXTURE_FEATURES,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
        embedding_dimension=0,
        gaussian_noise=DEFAULT_GAUSSIAN_NOISE,
        activation=relu,
        dropout=DEFAULT_DROPOUT_P,
        use_batch_norm=True,
    ):
        super().__init__()

        self.time_series_features = TimeSeriesFeatures(
            input_channels,
            64,
            feature_dimension=feature_dimension,
            gaussian_noise=gaussian_noise,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

        self.latent_pipeline = torch.nn.Sequential(
            torch.nn.Linear(
                feature_dimension + embedding_dimension,
                feature_dimension + embedding_dimension,
            ),
            activation,
        )

        self.mu_output = torch.nn.ConvTranspose1d(
            feature_dimension + embedding_dimension, mixture_components, input_channels
        )
        self.p_output = torch.nn.Conv1d(
            feature_dimension + embedding_dimension, mixture_components, 1
        )
        # It seems odd here to use "channels" as the matrix dimension,
        # but that's exactly what we want.  The number of input
        # channels is the number of time series.  Here we want a
        # square covariance matrix of the same dimension as the
        # output.
        self.sigma_inv_output = torch.nn.ConvTranspose2d(
            feature_dimension + embedding_dimension,
            mixture_components,
            (input_channels, input_channels),
        )

    def __dimensions(self):
        features_dimension = self.sigma_inv_output.in_channels
        components = self.sigma_inv_output.out_channels
        output_channels, input_channels = self.sigma_inv_output.kernel_size
        return features_dimension, components, output_channels, input_channels

    def forward(self, context, embedding=None, return_latents=False, debug=False):
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
        # Run through context pipeline squeeze to flatten
        latents = self.time_series_features(context)

        if embedding is not None:
            latents = torch.cat((latents, embedding), dim=1)

        latents = self.latent_pipeline(latents).unsqueeze(2)

        mu = self.mu_output(latents)
        log_p_raw = self.p_output(latents).squeeze(2)

        # Because we're using ConvTranspose2d to construct a
        # covariance matrix, here we convince ConvTranspose2d
        # that the input is 2D by adding a dimension using unsqueeze.
        sigma_inv = self.sigma_inv_output(latents.unsqueeze(3))

        # FIXME:  For compatibility with previously saved models
        # we get the shape from sigma_inv rather than from object state.
        output_channels, input_channels = sigma_inv.shape[2:]
        sigma_inv = torch.tril(sigma_inv, diagonal=(input_channels - output_channels))

        if not self.training:
            mu = torch.clamp(mu, -MIXTURE_MU_CLAMP, MIXTURE_MU_CLAMP)
            sigma_inv = torch.clamp(sigma_inv, -SIGMA_INV_CLAMP, SIGMA_INV_CLAMP)

        if debug:
            print("latents: ", latents)
            print("log_p_raw", log_p_raw)
            print("sigma_inv", sigma_inv)

        if return_latents:
            return log_p_raw, mu, sigma_inv, latents
        else:
            return log_p_raw, mu, sigma_inv


class MultivariateMixture64(torch.nn.Module):
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
        feature_dimension=DEFAULT_MIXTURE_FEATURES,
        mixture_components=DEFAULT_MIXTURE_COMPONENTS,
        embedding_dimension=0,
        gaussian_noise=DEFAULT_GAUSSIAN_NOISE,
        activation=relu,
        dropout=DEFAULT_DROPOUT_P,
        use_batch_norm=True,
    ):
        super().__init__()

        if output_channels is None:
            output_channels = input_channels

        self.limiter = MinMaxClamping()
        self.noise = GaussianNoise(gaussian_noise)
        self.context_pipeline = torch.nn.Sequential(
            # Assume content has 64 elements
            self.limiter,
            self.noise,
            torch.nn.Conv1d(input_channels, feature_dimension, 4, stride=4, padding=0),
            batch_norm_factory_1d(feature_dimension, use_batch_norm),
            activation,
            torch.nn.Dropout2d(p=dropout),
            # channel shape is (feature_dimension, 16)
            torch.nn.Conv1d(
                feature_dimension, feature_dimension, 4, stride=4, padding=0
            ),
            batch_norm_factory_1d(feature_dimension, use_batch_norm),
            activation,
            torch.nn.Dropout2d(p=dropout),
            # channel shape is (feature_dimension, 4)
            torch.nn.Conv1d(feature_dimension, feature_dimension, 4),
            batch_norm_factory_1d(feature_dimension, use_batch_norm),
            activation,
            torch.nn.Dropout2d(p=dropout),
            # Now have feature_dimension in a single slot
            # Do one more mixing layer.
            torch.nn.Conv1d(feature_dimension, feature_dimension, 1),
            batch_norm_factory_1d(feature_dimension, use_batch_norm),
            activation,
            torch.nn.Dropout2d(p=dropout)
            # Should have one channel of depth feature_dimension
        )

        self.latent_pipeline = torch.nn.Sequential(
            torch.nn.Linear(
                feature_dimension + embedding_dimension,
                feature_dimension + embedding_dimension,
            ),
            activation,
        )

        self.mu_output = torch.nn.ConvTranspose1d(
            feature_dimension + embedding_dimension, mixture_components, output_channels
        )
        self.p_output = torch.nn.Conv1d(
            feature_dimension + embedding_dimension, mixture_components, 1
        )
        # It seems odd here to use "channels" as the matrix dimension,
        # but that's exactly what we want.  The number of input
        # channels is the number of time series.  Here we want a
        # square covariance matrix of the same dimension as the
        # output.
        self.sigma_inv_output = torch.nn.ConvTranspose2d(
            feature_dimension + embedding_dimension,
            mixture_components,
            (output_channels, input_channels),
        )

    def dimensions(self):
        features_dimension = self.sigma_inv_output.in_channels
        components = self.sigma_inv_output.out_channels
        output_channels, input_channels = self.sigma_inv_output.kernel_size
        return features_dimension, components, output_channels, input_channels

    def just_latents(self, context):
        latents = self.context_pipeline(context)
        return latents

    def forward(self, context, embedding=None, return_latents=False, debug=False):
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
        # Run through context pipeline squeeze to flatten
        latents = self.just_latents(context)

        if embedding is not None:
            latents = torch.cat((latents, embedding.unsqueeze(2)), dim=1)

        # This check keeps this code backward compatible with
        # pre-existing model files that don't have embedding and don't have
        # latent_pipeline attributes.  We also need to presever
        # the channel dimension for backward compatibility where
        # the channel indicates the stock for multivariate portfolios.
        # Latents are only used for single symbols, so we remove the chennel,
        # apply another layer, and then restore the channel because the output
        # layers expect a channel.

        if hasattr(self, "latent_pipeline"):
            latents = self.latent_pipeline(latents.squeeze(2)).unsqueeze(2)

        mu = self.mu_output(latents)
        log_p_raw = self.p_output(latents).squeeze(2)

        # Because we're using ConvTranspose2d to construct a
        # covariance matrix, here we convince ConvTranspose2d
        # that the input is 2D by adding a dimension using unsqueeze.
        sigma_inv = self.sigma_inv_output(latents.unsqueeze(3))

        # FIXME:  For compatibility with previously saved models
        # we get the shape from sigma_inv rather than from object state.
        output_channels, input_channels = sigma_inv.shape[2:]
        sigma_inv = torch.tril(sigma_inv, diagonal=(input_channels - output_channels))

        if not self.training:
            mu = torch.clamp(mu, -MIXTURE_MU_CLAMP, MIXTURE_MU_CLAMP)
            sigma_inv = torch.clamp(sigma_inv, -SIGMA_INV_CLAMP, SIGMA_INV_CLAMP)

        if debug:
            print("latents: ", latents)
            print("log_p_raw", log_p_raw)
            print("sigma_inv", sigma_inv)

        if return_latents:
            return log_p_raw, mu, sigma_inv, latents
        else:
            return log_p_raw, mu, sigma_inv
