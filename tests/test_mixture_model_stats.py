import pytest
import torch
import mixture_model_stats
import math
import numpy as np

BATCH_SIZE = 16

softmax = torch.nn.Softmax(dim=1)
logsoftmax = torch.nn.LogSoftmax(dim=1)


def test_multivariate_likelihood():
    """
    This only checks that multivariate_mixture_log_likelihood can be called
    with arguments of consistent dimension within the correct range.  It does
    not confirm the correctness of the values returned.
    """
    for batch_size in range(BATCH_SIZE, BATCH_SIZE + 2):
        for mixture in range(1, 3):
            for channels in range(1, 3):
                x = torch.randn(batch_size, channels)
                log_p = logsoftmax(torch.randn(batch_size, mixture))
                mu = torch.randn(batch_size, mixture, channels)
                sigma_inv = torch.tril(
                    torch.randn(batch_size, mixture, channels, channels)
                )

                log_loss = loss_functions.multivariate_mixture_log_likelihood(
                    x, log_p, mu, sigma_inv
                )

                assert log_loss.shape == (batch_size,)
                assert float(torch.sum(log_loss)) != 0.0


LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
# TODO: Why can't this be smaller?
EPS = 1e-7


@pytest.mark.parametrize(
    "x,log_p,mu,sigma_inv,expected",
    [
        # Case 0
        (
            torch.ones((1, 1)),
            0 * torch.ones((1, 1)),
            torch.ones((1, 1, 1)),
            torch.ones((1, 1, 1, 1)),
            -LOG_SQRT_TWO_PI,
        ),
        # Case 1
        (
            torch.ones((1, 1)),
            0 * torch.ones((1, 1)),
            torch.ones((1, 1, 1)),
            2.0 * torch.ones((1, 1, 1, 1)),
            math.log(2.0) - LOG_SQRT_TWO_PI,
        ),
        # Case 2
        (
            torch.ones((1, 1)),
            0 * torch.ones((1, 1)),
            0 * torch.ones((1, 1, 1)),
            2.0 * torch.ones((1, 1, 1, 1)),
            math.log(2.0) - LOG_SQRT_TWO_PI - 2.0,
        ),
        # Case 3
        (
            0 * torch.ones((1, 1)),
            0 * torch.ones((1, 1)),
            torch.ones((1, 1, 1)),
            2.0 * torch.ones((1, 1, 1, 1)),
            math.log(2.0) - LOG_SQRT_TWO_PI - 2.0,
        ),
    ],
)
def test_multivariate_likelihood_cases(x, log_p, mu, sigma_inv, expected):
    log_loss = loss_functions.multivariate_mixture_log_likelihood(
        x, log_p, mu, sigma_inv
    )
    assert float(log_loss) == pytest.approx(expected, EPS)

    log_loss = loss_functions.univariate_mixture_log_likelihood(x, log_p, mu, sigma_inv)
    assert float(log_loss) == pytest.approx(expected, EPS)


@pytest.mark.parametrize(
    "mb_size, mixture_components",
    [
        # Case 0
        (7, 3),
        (11, 10),
    ],
)
def test_combine_mixture_metrics_random(mb_size, mixture_components):
    """
    Check that combine_mixture_metrics runs without failure on
    random inputs
    """
    p = softmax(torch.randn((mb_size, mixture_components)))
    mu = 0.001 + 0.0001 * torch.randn((mb_size, mixture_components))
    sigma = 0.01 * torch.exp(torch.randn((mb_size, mixture_components)))

    gain, std_dev = loss_functions.combine_mixture_metrics(p, mu, sigma)

    print(gain)
    print(std_dev)

    assert gain.shape == (mb_size,)
    assert std_dev.shape == (mb_size,)


def test_combine_mixture_metrics_specific():
    """Check that combine_mixture_metrics() returns correct results on
    a specific test case.
    """
    p = torch.tensor([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])

    mu_base = np.array([0.001, 0.0005], dtype=np.float32)
    sigma_base = np.array([0.01, 0.005], dtype=np.float32)

    mu = torch.tensor(mu_base).unsqueeze(0).expand((len(p), len(mu_base)))
    sigma = torch.tensor(sigma_base).unsqueeze(0).expand((len(p), len(sigma_base)))

    print("p: ", p)
    print("mu: ", mu)
    print("sigma: ", sigma)

    gain_test, std_dev_test = loss_functions.combine_mixture_metrics(p, mu, sigma)

    print("gain_test: ", gain_test)
    print("std_dev_test: ", std_dev_test)

    # Compute the expected values using numpy functions
    component_means = np.exp(mu_base + 0.5 * sigma_base ** 2)
    component_variances = (np.exp(sigma_base ** 2) - 1.0) * component_means ** 2

    print("component means:", component_means)
    print("component variances: ", component_variances)

    mean_expected = []
    std_dev_expected = []

    for p_row in p:
        combined_mean = np.dot(p_row, component_means)
        combined_variance = np.dot(
            p_row, component_variances + (combined_mean - component_means) ** 2
        )
        combined_std_dev = np.sqrt(combined_variance)

        mean_expected.append(combined_mean)
        std_dev_expected.append(combined_std_dev)

    gain_expected = torch.tensor(mean_expected) - 1.0
    std_dev_expected = torch.tensor(std_dev_expected)

    print("gain_expected: ", gain_expected)
    print("gain_test: ", gain_test)
    print("std_dev_expected: ", std_dev_expected)
    print("std_dev_test: ", std_dev_test)

    assert torch.allclose(gain_test, gain_expected, rtol=1e-4)
    assert torch.allclose(std_dev_test, std_dev_expected, rtol=1e-4)


def test_ensure_nonsingular():
    """Check that ensure_nonsingular() on a specific test case."""
    base_s = torch.tensor(
        [
            [1, 0.0, 0.0, 0.0, 0.0],
            [0.75, -1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 1e-10, 0.0, 0.0],
            [0.5, 0.5, 0.7, -1e-8, 0.0],
            [0.5, 0.5, -1.0, 0.0, 0.0],
        ]
    )
    s = base_s.unsqueeze(0).unsqueeze(0)
    modified_s = loss_functions.ensure_nonsingular(s)
    new_diagonal = torch.diagonal(modified_s, dim1=2, dim2=3).squeeze(0).squeeze(0)
    delta = (torch.diagonal(modified_s - s, dim1=2, dim2=3)).squeeze(0).squeeze(0)
    print(new_diagonal)

    # Check both the direction of the change and the magnitude of any modified values:

    # rows 0 and 1 should not change
    assert delta[0] == 0.0
    assert delta[1] == 0.0

    # The diagonal entries in rows 2 and 3 should move in the same
    # direction as their original sign
    assert delta[2] > 0.0
    assert delta[3] < 0.0

    # The zero entry in row 4 should end up positive
    assert delta[4] > 0.0

    # The new values should have the same magnitude
    assert abs(new_diagonal[2]) == abs(new_diagonal[3])
    assert abs(new_diagonal[2]) == abs(new_diagonal[4])
