import pytest

# Third party libraries
import torch

# Local modules
import time_series


def test_rolling_window_arg_check():
    with pytest.raises(ValueError):
        time_series.RollingWindowSeries(range(10), 3, stride=0)


@pytest.mark.parametrize(
    "series,window,stride,expected",
    [
        (
            range(10),
            3,
            1,
            [
                torch.tensor(range(0, 3)),
                torch.tensor(range(1, 4)),
                torch.tensor(range(2, 5)),
                torch.tensor(range(3, 6)),
                torch.tensor(range(4, 7)),
                torch.tensor(range(5, 8)),
                torch.tensor(range(6, 9)),
                torch.tensor(range(7, 10)),
            ],
        ),
        # Same case with a different stride
        (
            range(10),
            3,
            2,
            [
                torch.tensor(range(0, 3)),
                torch.tensor(range(2, 5)),
                torch.tensor(range(4, 7)),
                torch.tensor(range(6, 9)),
            ],
        ),
        # Check a sequence of vectors
        (
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            2,
            1,
            [
                torch.tensor([[1, 4], [2, 5], [3, 6]]),
                torch.tensor([[4, 7], [5, 8], [6, 9]]),
                torch.tensor([[7, 10], [8, 11], [9, 12]]),
            ],
        ),
    ],
)
def test_rolling_window_series(series, window, stride, expected):
    d = time_series.RollingWindowSeries(series, window, stride=stride)
    assert len(d) == len(expected)

    # We use indexes here rather than iterators because we're specifically
    # testing the implementation of __getitem__()
    for i in range(len(expected)):
        print(f"\nwindow returned:\n{d[i]}")
        print(f"window expected:\n{expected[i]}")
        assert d[i].shape == expected[i].shape
        assert (d[i] == expected[i]).all()

    # Make sure negative indexes work
    for i in range(-len(expected), 0):
        assert (d[i] == expected[i]).all()

    with pytest.raises(IndexError):
        d[len(expected)]
    with pytest.raises(IndexError):
        d[-len(expected) - 1]


@pytest.mark.parametrize(
    "series,window,stride,expected_cov,expected_target",
    [
        (
            range(10),
            3,
            2,
            [
                torch.tensor([0, 1]),
                torch.tensor([2, 3]),
                torch.tensor([4, 5]),
                torch.tensor([6, 7]),
            ],
            [
                torch.tensor(2),
                torch.tensor(4),
                torch.tensor(6),
                torch.tensor(8),
            ],
        ),
        # Check a sequence of vectors
        (
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
            ],
            2,
            1,
            [
                torch.tensor([[1], [2], [3]]),
                torch.tensor([[4], [5], [6]]),
                torch.tensor([[7], [8], [9]]),
            ],
            [
                torch.tensor([4, 5, 6]),
                torch.tensor([7, 8, 9]),
                torch.tensor([10, 11, 12]),
            ],
        ),
    ],
)
def test_target_selection(series, window, stride, expected_cov, expected_target):
    d = time_series.RollingWindowSeries(series, window, stride=stride)
    cts = time_series.ContextAndTargetSeries(d, target_dim=1)

    assert len(cts) == len(expected_target)

    # We use indexes here rather than iterators because we're specifically
    # testing the implementation of __getitem__()
    for i in range(len(expected_target)):
        cov, target = cts[i]
        print(f"cov returned:\n\n{cov}")
        print(f"cov expected:\n{expected_cov[i]}")
        assert cov.shape == expected_cov[i].shape
        assert (cov == expected_cov[i]).all()

        print(f"\ntarget returned:\n{target}")
        print(f"target expected:\n{expected_target[i]}")
        assert target.shape == expected_target[i].shape
        assert (target == expected_target[i]).all()

    # Make sure negatives indexes work
    for i in range(-len(expected_target), 0):
        cov, target = cts[i]
        assert (cov == expected_cov[i]).all()
        assert (target == expected_target[i]).all()

    with pytest.raises(IndexError):
        cts[len(expected_target)]

    with pytest.raises(IndexError):
        cts[-len(expected_target) - 1]
