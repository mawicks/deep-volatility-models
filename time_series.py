from typing import Iterable, Any

import numpy as np
import torch
import torch.utils.data


def multivariate_stats(x):
    """
    Given a time series x, estimate the mean (mu) and the square root of
    the covariance (sigma) for that time series.
    Inputs:
      x: tensor of shape (tensor(mb_size, channels)) containing the sequence values
    Outputs:
      mu: tensor of shape: (channels,) containing the mean estimates
      sigma: tensor of shape: ((channels, channels) containing an estimate of
      the lower Cholesky factor of the covariance matrix.

      TODO: To improve numerical stability, Use an SVD to compute the Cholesky
      factor rather than the naive formula.
    """
    # Create tensor version of x in case it isn't already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    mb_size, channels = x.shape
    mu = torch.mean(x, dim=0)
    error = x - mu.unsqueeze(0).expand((mb_size, channels))
    # error is mb_size x channels
    error1 = error.unsqueeze(2)
    # error1 represents e (mb_size, channels, 1)
    error2 = error.unsqueeze(1)
    # error2 represents e^T (mb_size, 1, channels)
    cov = torch.mean(torch.matmul(error1, error2), dim=0)
    # cov is (channels, channels)

    # Return cholesky factor
    sigma = torch.linalg.cholesky(cov)
    return mu, sigma


class RollingWindowSeries(torch.utils.data.Dataset):
    """
    Given a time series, construct a sequence of rolling windows on the series.
    The resuling windows are compatible with the pytorch dataloader: the kth
    window is obtained by indexing the kth element of the output series.  Also,
    for compatibility with pytorch the output is represented by pytorch tensors
    and it follows pytorch conventions dimension order as explained below.

    Example usage:

    >>> import time_series

    This modules works with time seris of scalars or time series of vectors. The
    first example is a sequence of scalars:

    >>> series = list(range(5))
    >>> series
    [0, 1, 2, 3, 4]

    Construct a rolling sequence of windows for the series with a window size of
    3 and a default stride of 1.

    >>> windowed_series = time_series.RollingWindowSeries(series, 3)

    The first element (element 0) is a window with the first three values:

    >>> windowed_series[0]
    tensor([0., 1., 2.])

    The second element (element 1) is a window with the next three values:

    >>> windowed_series[1]
    tensor([1., 2., 3.])

    The third element (element 2) is a window with the next three values:

    >>> windowed_series[2]
    tensor([2., 3., 4.])

    For use with convolutional neworks, it's often necessary to create a channel dimension:

    >>> windowed_series = time_series.RollingWindowSeries(series, 3, create_channel_dim=True)
    >>> windowed_series[0]
    tensor([[0., 1., 2.]])

    RollingWindowSeries also works for vector-valued time series as long as you
    understand some conventions about the ordering of dimensions.  We assume
    that the first dimension of the input (dimension 0) represents time.  In
    other words, we assume the input is a sequence of vectors.  This is a
    natural convention for the input sequence.  However, we follow the pytorch
    convention on the output.  The pytorch convention is that the *last*
    dimension represents time.  In effect, the vector dimension becomes the
    channel dimension, so the `create_channel_dim` option is meaningless in this
    case.

    An example will clarify these ideas.

    >>> vector_series = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> windowed_vector_series = time_series.RollingWindowSeries(vector_series, 3)
    >>> windowed_vector_series[0]
    tensor([[1., 3., 5.],
            [2., 4., 6.]])

    The result may seem "transposed", but that's for consistency with pytorch
    conventions and necessary for use with a number of pytorch functions. Here's
    the rationale.  For a sequence of vectors, the vector dimension should be
    thought of as the "depth" dimension (e.g., RGB for images). The pytorch
    convention is for the depth to be the first dimension dimension 0) of the
    tensor and for the "time" (or space) dimension to be dimension 1 for 1d or
    dimensions 1 and 2 for 2d.  When these ecords get batched for machine
    learning, the index of the record is always dimension 0, so the depth
    becomes dimension 1, and "time" becomes dimension 2.  The convention for
    batched records is typically as follows:

       dimension 0 - index of record within a batch dimension 1 - "depth"
       dimension dimension 2 - "time" dimension for 1d or "x" dimensions for 2d
       dimension 3 - "y" dimension for 2d

    Since we're looking at records before they have been batched, the convention
    is

       dimension 0 - "depth" dimension dimension 1 - "time" dimension for 1d or
       "x" dimensions for 2d dimension 2 - "y" dimension for 2d

    More generally, the pytorch convention for time series (or any 1d signal) is
    that time (or whatever the 1d dimension represents) should always be the
    *last* dimension.  For images, "x" and "y" should be the last *two*
    dimensions. Continuing the exmaple, here's the next window:

    >>> windowed_vector_series[1]
    tensor([[3., 5., 7.],
            [4., 6., 8.]])

    Note: This code currently works for sequences of scalars and sequences of 1d
    vectors.

    TODO: Make this code work for sequences of tensors with two or more
    diemsions while following the above conventions that "time" should be the
    last dimension.
    """

    def __init__(
        self,
        series: Iterable[Any],
        sequence_length: int,
        stride: int = 1,
        create_channel_dim: bool = False,
        dtype: type = torch.float,
    ):
        if stride <= 0:
            raise ValueError("Stride cannot be negative")

        # Originally np.array() was simply suple(series)
        # pytorch issued a warning a recommended the use of np.array()
        self.__series = np.array(series)

        if (
            len(self.__series) > 0
            and hasattr(self.__series[0], "__len__")
            and len(self.__series[0]) > 0
            and create_channel_dim
        ):
            raise ValueError("create_channel_dim should be False for this series shape")

        self.__sequence_length = sequence_length
        self.__stride = stride
        self.__length = (len(self.__series) - sequence_length) // stride + 1
        self.__create_channel_dim = create_channel_dim
        self.__dtype = dtype

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        if index < 0:
            index = self.__length + index

        if index >= 0 and index < self.__length:
            start = index * self.__stride
            result = torch.tensor(
                self.__series[start : start + self.__sequence_length],
                dtype=self.__dtype,
            )
            if len(result.shape) == 1:
                if self.__create_channel_dim:
                    result = result.unsqueeze(0)
            else:
                result = result.t()

            return result
        else:
            raise IndexError()


class ContextAndTargetSeries(torch.utils.data.Dataset):
    """Split time series slices into covariates and target"""

    def __init__(self, rolling_window_series, target_dim: int = 1):
        """Generally, the stride used to construct Dataset should be equal to
        target_dim

        """
        self.__time_series_dataset = rolling_window_series
        self.__target_dim = target_dim

    def __len__(self):
        return len(self.__time_series_dataset)

    def __getitem__(self, index):
        t = self.__time_series_dataset[index]
        if len(t.shape) == 1:  # Sequence of scalars
            covariates = t[: -self.__target_dim]
            target = t[-self.__target_dim :]
            # Drop the last dimension when it's one.
            if self.__target_dim == 1:
                target = target.squeeze(-1)
        else:  # Sequence of vectors
            covariates = t[:, : -self.__target_dim]
            target = t[:, -self.__target_dim :]

        return covariates, target
