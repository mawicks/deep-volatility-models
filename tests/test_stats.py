import pytest

import math

import torch

import deep_volatility_models.stats as stats

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
# TODO: Why can't this be smaller?
EPS = 1e-7


@pytest.mark.parametrize(
    "x,mu,sigma_inv,expected",
    [
        # Case 0
        (
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.ones((1, 1, 1)),
            -LOG_SQRT_TWO_PI,
        ),
        # Case 1
        (
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            2.0 * torch.ones((1, 1, 1)),
            math.log(2.0) - LOG_SQRT_TWO_PI,
        ),
        # Case 2
        (
            torch.ones((1, 1)),
            0 * torch.ones((1, 1)),
            2.0 * torch.ones((1, 1, 1)),
            math.log(2.0) - LOG_SQRT_TWO_PI - 2.0,
        ),
        # Case 3
        (
            0 * torch.ones((1, 1)),
            torch.ones((1, 1)),
            2.0 * torch.ones((1, 1, 1)),
            math.log(2.0) - LOG_SQRT_TWO_PI - 2.0,
        ),
    ],
)
def test_likelihood_cases(x, mu, sigma_inv, expected):
    log_loss = stats.univariate_log_likelihood(x, mu, sigma_inv)
    assert float(log_loss) == pytest.approx(expected, EPS)
