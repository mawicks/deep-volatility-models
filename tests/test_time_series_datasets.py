import pytest

# Third party libraries
import torch

# Local modules
import time_series_datasets


# Constants used in tests.
A_SYMBOL_ENCODING = 21


"""
The test cases for multivariate_stats() was generated as follows:

Assume x = L*z + b has zero mean and covariance = I

Now E[x] = b
C_x = E[xx’] = E[L (zz’) L’] = LL’
Where L is a lower triangular matrix.

We can use this to generate series of x with a specific L and b.
For example, let b = [1, 2]
let L = [1, 0],  [-1; 2]]

We need to choose values for z with zero mean and C = I

One possibility is
Z = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

Where each row is a (z1, z2) pair.

Because “time” is the row dimension of Z, we need to transpose the original 
equation to be:

x = ZL’ + b’

where Z is as above,  L=[[1, 0], [-1, 2]], 
and b’ = [[1, 2], [1, 2], [1, 2], [1, 2]]

>>> l = np.array([[1, 0], [-1, 2]])
>>> l
array([[ 1,  0],
       [-1,  2]])
>>> z=np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
>>> z
array([[ 1,  1],
       [ 1, -1],
       [-1,  1],
       [-1, -1]])
>>> b=np.array([[1, 2], [1, 2], [1, 2], [1, 2]])
>>> b
array([[1, 2],
       [1, 2],
       [1, 2],
       [1, 2]])

>>> np.matmul(z, l.T) + b
array([[ 2,  3],
            [ 2, -1],
            [ 0,  5],
            [ 0,  1]])
>>> 

"""


@pytest.mark.parametrize(
    "series, mu_expected, l_expected",
    [
        (
            [[2.0, 3.0], [2.0, -1.0], [0.0, 5.0], [0.0, 1.0]],
            torch.tensor([1, 2], dtype=torch.float),
            torch.tensor([[1.0, 0.0], [-1.0, 2.0]]),
        ),
    ],
)
def test_multivariate_stats(series, mu_expected, l_expected):
    mu, l = time_series_datasets.multivariate_stats(series)
    assert mu.shape == mu_expected.shape
    assert l.shape == l_expected.shape
    print(f"mu returned:\n{mu}")
    print(f"mu expected:\n{mu_expected}")
    print(f"\nl returned:\n{l}")
    print(f"l expected:\n{l_expected}")

    # Fortunately the test case is compute *exactly* so no approximate
    # comparisons are necessary.
    assert (mu == mu_expected).all()
    assert (l == l_expected).all()


def test_rolling_window_arg_check():
    with pytest.raises(ValueError):
        time_series_datasets.RollingWindow(range(10), 3, stride=0)

    with pytest.raises(ValueError):
        time_series_datasets.RollingWindow(
            [[1, 3], [3, 4], [5, 6]],
            2,
            create_channel_dim=True,
        )


@pytest.mark.parametrize(
    "series,window_size,stride,create_channel_dim,expected",
    [
        (
            range(10),
            3,
            1,
            False,
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
            False,
            [
                torch.tensor(range(0, 3)),
                torch.tensor(range(2, 5)),
                torch.tensor(range(4, 7)),
                torch.tensor(range(6, 9)),
            ],
        ),
        # Same case with create_channel_dim=True
        (
            range(10),
            3,
            1,
            True,
            [
                torch.tensor([list(range(0, 3))]),
                torch.tensor([list(range(1, 4))]),
                torch.tensor([list(range(2, 5))]),
                torch.tensor([list(range(3, 6))]),
                torch.tensor([list(range(4, 7))]),
                torch.tensor([list(range(5, 8))]),
                torch.tensor([list(range(6, 9))]),
                torch.tensor([list(range(7, 10))]),
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
            False,
            [
                torch.tensor([[1, 4], [2, 5], [3, 6]]),
                torch.tensor([[4, 7], [5, 8], [6, 9]]),
                torch.tensor([[7, 10], [8, 11], [9, 12]]),
            ],
        ),
    ],
)
def test_rolling_window_series(
    series, window_size, stride, create_channel_dim, expected
):
    d = time_series_datasets.RollingWindow(
        series, window_size, stride=stride, create_channel_dim=create_channel_dim
    )
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
    "series,window_size,stride,expected_window,expected_target",
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
                torch.tensor([[4], [5], [6]]),
                torch.tensor([[7], [8], [9]]),
                torch.tensor([[10], [11], [12]]),
            ],
        ),
    ],
)
def test_target_selection(
    series, window_size, stride, expected_window, expected_target
):
    raw_windows = time_series_datasets.RollingWindow(series, window_size, stride=stride)
    window_and_target = time_series_datasets.ContextWindowAndTarget(
        raw_windows, target_dim=1
    )
    encoding_window_and_target = time_series_datasets.EncodingContextWindowAndTarget(
        A_SYMBOL_ENCODING, window_and_target
    )

    assert len(window_and_target) == len(expected_target)
    assert len(encoding_window_and_target) == len(expected_target)

    # We use indexes here rather than iterators because we're specifically
    # testing the implementation of __getitem__()
    for i in range(len(expected_target)):
        window, target = window_and_target[i]
        print(f"window returned:\n\n{window}")
        print(f"window expected:\n{expected_window[i]}")
        assert window.shape == expected_window[i].shape
        assert (window == expected_window[i]).all()

        print(f"\ntarget returned:\n{target}")
        print(f"target expected:\n{expected_target[i]}")
        assert target.shape == expected_target[i].shape
        assert (target == expected_target[i]).all()

        (encoding, window), target = encoding_window_and_target[i]
        assert encoding == A_SYMBOL_ENCODING
        assert window.shape == expected_window[i].shape
        assert (window == expected_window[i]).all()
        assert target.shape == expected_target[i].shape
        assert (target == expected_target[i]).all()

    # Make sure negatives indexes work
    for i in range(-len(expected_target), 0):
        window, target = window_and_target[i]
        assert (window == expected_window[i]).all()
        assert (target == expected_target[i]).all()

        (encoding, window), target = encoding_window_and_target[i]
        assert encoding == A_SYMBOL_ENCODING
        assert (window == expected_window[i]).all()
        assert (target == expected_target[i]).all()

    with pytest.raises(IndexError):
        window_and_target[len(expected_target)]

    with pytest.raises(IndexError):
        window_and_target[-len(expected_target) - 1]
