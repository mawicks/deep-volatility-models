import torch
import torch.utils.data


class RollingWindowSeries(torch.utils.data.Dataset):
    """DataSet subclass for time series"""

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
