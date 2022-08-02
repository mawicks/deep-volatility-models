# Standard Python
import math

# Common packages
import torch

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
EPS_FOR_LOG = 1e-8

# TODO: Write a test for this


def univariate_log_likelihood(
    x: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor
):
    """Inputs:
       x: tensor of shape tensor(mb_size, symbols=1) containing the observed values

       mu: tensor of shape (mb_size, symbols=1) containing the mu
       estimate for each component

       sigma_inv: tensor of shape (mb_size, input_symbols=1,
       output_symbols=1) containing the estimate of the reciprocal of
       the sqrt of the variance for each component

    Output:
       tensor of shape (mb_size,) containing the log likelihood for each sample
       in the batch

    Note:
       The symbol dimension may seem superfluous, but the
       dimensions of the input tensors have been chosen for
       compatability with a multivarate version of this function,
       which requires the number of symbols.  The dimensions
       associated with the number of symbols are required to be 1.

    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float)
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, dtype=torch.float)
    if not isinstance(sigma_inv, torch.Tensor):
        sigma_inv = torch.tensor(sigma_inv, dtype=torch.float)

    mb_size, symbols = sigma_inv.shape[:2]
    if (
        x.shape != (mb_size, symbols)
        or mu.shape != (mb_size, symbols)
        or sigma_inv.shape != (mb_size, symbols, symbols)
    ):
        raise ValueError(
            f"Dimensions of x {x.shape}, mu {mu.shape}, and sigma_inv {sigma_inv.shape} are inconsistent"
        )

    if symbols != 1:
        raise ValueError(
            f"This function requires the number of symbols to be 1 and not {symbols}"
        )

    # Drop the dimensions that were just confirmed to be one.
    x = x.squeeze(1)
    mu = mu.squeeze(1)
    sigma_inv = sigma_inv.squeeze(2).squeeze(1)

    z_squared = (sigma_inv * (x - mu)) ** 2

    # Inclusion of EPS is to ensure argument remains bounded away from zero.
    log_sigma_inv = torch.log(
        torch.maximum(torch.tensor(EPS_FOR_LOG), torch.abs(sigma_inv))
    )

    # log_p, z_squared, and log_sigma_inv have the same shape: (mb_size, mixture_components)

    ll = -0.5 * z_squared + log_sigma_inv - LOG_SQRT_TWO_PI

    return ll
