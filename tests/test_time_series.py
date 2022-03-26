import pytest

# Third party libraries
import torch

# Local modules
import time_series


def test_rolling_window_series():
    d = time_series.RollingWindowSeries(range(10), 3)
    assert len(d) == 8
    assert tuple(d[0]) == tuple(range(3))
    assert tuple(d[1]) == tuple(range(1, 4))
    assert tuple(d[-1]) == tuple(range(7, 10))
    assert tuple(d[-2]) == tuple(range(6, 9))
    assert tuple(d[-8]) == tuple(range(3))
    with pytest.raises(IndexError):
        d[8]
    with pytest.raises(IndexError):
        d[-9]

    d = time_series.RollingWindowSeries(range(10), 3, stride=2)
    assert len(d) == 4
    assert tuple(d[0]) == tuple(range(3))
    assert tuple(d[1]) == tuple(range(2, 5))
    assert tuple(d[-1]) == tuple(range(6, 9))
    assert tuple(d[-2]) == tuple(range(4, 7))
    assert tuple(d[-4]) == tuple(range(3))
    with pytest.raises(IndexError):
        d[4]
    with pytest.raises(IndexError):
        d[-5]

    with pytest.raises(ValueError):
        time_series.RollingWindowSeries(range(10), 3, stride=0)


def test_target_selection():
    d = time_series.RollingWindowSeries(range(10), 3, stride=2)
    cts = time_series.ContextAndTargetSeries(d, target_dim=1)

    assert len(d) == 4
    assert len(cts) == 4

    expected_cov = [
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 7]),
    ]

    expected_target = [2, 4, 6, 8]

    for i in range(4):
        cov, target = cts[i]
        assert (cov == expected_cov[i]).all()
        assert target == expected_target[i]

    # Make sure negatives indexes work
    for i in range(-4, 0):
        cov, target = cts[i]
        assert (cov == expected_cov[i]).all()
        assert target == expected_target[i]

    with pytest.raises(IndexError):
        cts[4]

    with pytest.raises(IndexError):
        cts[-5]


def test_vector_series():
    series = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
    # Output sequence (represented as np arrays) should be
    expected = [
        torch.tensor([[1, 4], [2, 5], [3, 6]]),
        torch.tensor([[4, 7], [5, 8], [6, 9]]),
        torch.tensor([[7, 10], [8, 11], [9, 12]]),
    ]

    d = time_series.RollingWindowSeries(series, 2)
    for i in range(3):
        print(f"\nExpected:\n{expected[i]}")
        print(f"Got:\n{d[i]}")
        assert (d[i] == expected[i]).all().all()

    # Make sure negative indexes work
    for i in range(-3, 0):
        print(f"index: {i}")
        assert (d[i] == expected[i]).all().all()

    with pytest.raises(IndexError):
        d[3]

    with pytest.raises(IndexError):
        d[-4]

    cts = time_series.ContextAndTargetSeries(d, target_dim=1)

    expected_cov = [
        torch.tensor([[1], [2], [3]]),
        torch.tensor([[4], [5], [6]]),
        torch.tensor([[7], [8], [9]]),
    ]

    expected_target = [
        torch.tensor([[4], [5], [6]]),
        torch.tensor([[7], [8], [9]]),
        torch.tensor([[10], [11], [12]]),
    ]

    for i in range(3):
        cov, target = cts[i]
        print(f"\nExpected cov:\n{expected_cov[i]}")
        print(f"Got cov:\n {cov}")

        print(f"\nExpected target:\n{expected_target[i]}")
        print(f"Got target:\n{target}")

        assert (cov == expected_cov[i]).all().all()
        assert (target == expected_target[i]).all().all()

        # Make sure negative indexes work

    for i in range(-3, 0):
        print(f"index: {i}")
        cov, target = cts[i]
        assert (cov == expected_cov[i]).all().all()
        assert (target == expected_target[i]).all().all()

    with pytest.raises(IndexError):
        cts[3]

    with pytest.raises(IndexError):
        cts[-4]
