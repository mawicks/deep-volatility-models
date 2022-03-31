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

                log_loss = mixture_model_stats.multivariate_log_likelihood(
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
    log_loss = mixture_model_stats.multivariate_log_likelihood(x, log_p, mu, sigma_inv)
    assert float(log_loss) == pytest.approx(expected, EPS)

    log_loss = mixture_model_stats.univariate_log_likelihood(x, log_p, mu, sigma_inv)
    assert float(log_loss) == pytest.approx(expected, EPS)


def test_univeriate_fails_on_multivariate_input():
    mb_size, mixture_components, symbols = (5, 3, 2)

    x = torch.randn(mb_size, symbols)
    log_p = torch.randn(mb_size, mixture_components)
    mu = torch.randn(mb_size, mixture_components, symbols)
    sigma_inv = torch.randn(mb_size, mixture_components, symbols, symbols)

    # As a sanity check that the dimensions are correct except for being
    # multi-variate, a ccall to
    # mixture_model_state.multivariate_log_likelihood() should return
    # *something*:

    mixture_model_stats.multivariate_log_likelihood(x, log_p, mu, sigma_inv)

    with pytest.raises(ValueError):
        log_loss = mixture_model_stats.univariate_log_likelihood(
            x, log_p, mu, sigma_inv
        )


def test_univeriate_fails_on_inconsistent_dimensions():

    mb_size, mixture_components, symbols = (5, 3, 1)

    x = torch.randn(mb_size, symbols)
    log_p = torch.randn(mb_size, mixture_components)
    # Introduce an incompatible dimension
    mu = torch.randn(mb_size, mixture_components + 1, symbols)
    sigma_inv = torch.randn(mb_size, mixture_components, symbols, symbols)

    with pytest.raises(ValueError):
        log_loss = mixture_model_stats.univariate_log_likelihood(
            x, log_p, mu, sigma_inv
        )
