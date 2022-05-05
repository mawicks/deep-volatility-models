import math
import torch


def gcd(list_like):
    """Find the GCD of a list-like of integers"""
    il = iter(list_like)
    result = next(il)
    for i in il:
        result = math.gcd(result, i)
    return result


def combine_mixture_metrics(p, mu, sigma_inv):
    """
    Inputs:
        p (tensor(mb_size, mixture_componente)):
           probability of each component
        mu (tensor(mb_size, mixture_components, symbols): mu for each component
        sigma_inv (tensor(mb_size, mixture_components, symbols, symbols): sigma for each component

    Outputs:
        return (tensor(mb_size)): expected return
        std_dev (tensor(mb_size, symbols, symbols)): std deviations (sqrt of covariance matrix)
    """

    mb_size, mixture_components, symbols = mu.shape
    if symbols != 1:
        raise Exception(
            "FIXME:  This code currently requires number of symbols to be 1"
        )

    # Drop symbols dimension on mu and sigma_inv which is known to be 1
    sigma_inv = sigma_inv.squeeze(3).squeeze(2)
    mu = mu.squeeze(2)

    variance = (1.0 / sigma_inv) ** 2
    composite_mean = torch.sum(p * mu, dim=1)
    print(f"composite_mean: {composite_mean}")

    # TODO: Verify this is correct implementation of parallel axis theorem
    # E[(x-mu)**2] = sum p_i E[(x_i-mu)**2]
    # E[(x_i-mu)**2] = E[((x_i-mu_i) + (mu_i-mu))**2]
    #                = sigma_i**2 + (mu_i-mu)**2
    shifted_component_variances = variance + (mu - composite_mean.unsqueeze(1)) ** 2
    composite_std_dev = torch.sqrt(torch.sum(p * shifted_component_variances, dim=1))
    return composite_mean, composite_std_dev
