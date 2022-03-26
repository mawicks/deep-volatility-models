import torch
import torch.utils.data


class RollingWindowSeries(torch.utils.data.Dataset):
    """
    Given a time series, return a sequence of rolling windows on the series.
    This generates windows compatible for pytorch dataloader, so the kth window
    is obtained by indexing the kth element of the output series.  Also for compatibility
    with pytorch, the output is represented as pytorch tensors.

    Example usage:
    >>> import time_series

    This modules works with either scalar time series or vector time series.
    The first example is a sequence of scalars:

    >>> series = list(range(5))
    >>> series
    [0, 1, 2, 3, 4]

    Construct a rolling sequence of windows for the series with a window size of 3
    and a default stride of 1.

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

    RollingWindowSeries also works for vector-valued time series if you following
    some conventions about the ordering of dimensions.  We assume that the first
    dimension of the input (dimension 0) represents time since it's a sequence
    of vectors.  This is a natural convention for the input sequence, however,
    we follow the pytorch convention on the output.  The pytorch convention is
    that the *last* dimension represents time.

    An example will clarify these ideas.

    >>> vector_series = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> windowed_vector_series = time_series.RollingWindowSeries(vector_series, 3)
    >>> windowed_vector_series[0]
    tensor([[1., 3., 5.],
            [2., 4., 6.]])

    The result may seem "transposed", but that's for consistency with pytorch
    conventions used in a number of functions.  Here's the rationale.  For
    a sequence of vectors, the vector dimension should be thought of as the
    "depth" dimension (e.g., RGB for images). The pytorch convention is for
    the depth to be the first dimension (dimension 0) of the tensor and for
    the "time" (or space) dimension to be dimension 1 for 1d or dimensions 1
    and 2 for 2d.  When these records get batched for machine learning, the index
    of the record is always dimension 0, so the depth becomes dimension 1,
    and "time" becomes dimension 2.  The convention for batched records is
    typically as follows:

       dimension 0 - index of record within a batch
       dimension 1 - "depth" dimension
       dimension 2 - "time" dimension for 1d or "x" dimensions for 2d
       dimension 3 - "y" dimension for 2d

    Since we're looking at records before they have been batched, the convention is

       dimension 0 - "depth" dimension
       dimension 1 - "time" dimension for 1d or "x" dimensions for 2d
       dimension 2 - "y" dimension for 2d

    More generally, the pytorch convention for time series (or any 1d signal)
    is that time (or whatever the 1d dimension represents) should always be
    the *last* dimension.  For images, "x" and "y" should be the last
    *two* dimensions.
    Continuing the exmaple, here's the next window:

    >>> windowed_vector_series[1]
    tensor([[3., 5., 7.],
            [4., 6., 8.]])

    Note: This code currently works for sequences of scalars and sequences of 1d vectors.
    TODO: Make this code work for sequences of tensors with two or more diemsions while following
    the above conventions that "time" should be the last dimension.
    """

    def __init__(self, series, sequence_length, stride=1):
        if stride <= 0:
            raise ValueError()

        self.__series = tuple(series)
        self.__sequence_length = sequence_length
        self.__stride = stride
        self.__length = (len(self.__series) - sequence_length) // stride + 1

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        if index < 0:
            index = self.__length + index

        if index >= 0 and index < self.__length:
            start = index * self.__stride
            result = torch.tensor(
                self.__series[start : start + self.__sequence_length], dtype=torch.float
            )
            if len(result.shape) == 1:
                return result
            else:
                return result.t()
        else:
            raise IndexError()


class ContextAndTargetSeries(torch.utils.data.Dataset):
    """Split time series slices into covariates and target"""

    def __init__(self, rolling_window_series, target_dim=1, label=None):
        """Generally, the stride used to construct Dataset should be equal to
        target_dim

        """
        self.__time_series_dataset = rolling_window_series
        self.__target_dim = target_dim
        if label is not None:
            self.__label = torch.tensor(label)
        else:
            self.__label = None

    def __len__(self):
        return len(self.__time_series_dataset)

    def __getitem__(self, index):
        t = self.__time_series_dataset[index]
        if len(t.shape) == 1:
            covariates = t[: -self.__target_dim]
            target = t[-self.__target_dim :]
        else:
            covariates = t[:, : -self.__target_dim]
            target = t[:, -self.__target_dim :]

        if self.__label is None:
            return covariates, target
        else:
            return self.__label, covariates, target
