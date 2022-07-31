# Standard Python
import math

# Common packages
import torch

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
EPS_FOR_LOG = 1e-8
EPS_FOR_SINGULARITY = 1e-4

# TODO: Write a test for this


def univariate_log_likelihood(
    x: torch.Tensor, log_p: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor
):
    """
    Inputs:
       x: tensor of shape tensor(mb_size,1) containing the observed values

       log_p: tensor of shape (mb_size, mixture_componente) containing the log
       probability of each component.

       mu: tensor of shape (mb_size, mixture_components, 1) containing the mu
       estimate for each component

       sigma_inv: tensor of shape (mb_size, mixture_components, 1, 1) containing the
       estimate of the reciprocal of the sqrt of the variance for each component

    Output:
       tensor of shape (mb_size,) containing the log likelihood for each sample
       in the batch

    Note: The dimensions of the input tensors have been chosen for compatability
    with a multivarate version of this function.  The dimensions associated with
    the number of symbols are required to be 1.

    """
    mb_size, mixture_components, symbols = sigma_inv.shape[:3]
    if (
        x.shape != (mb_size, symbols)
        or log_p.shape != (mb_size, mixture_components)
        or mu.shape != (mb_size, mixture_components, symbols)
        or sigma_inv.shape != (mb_size, mixture_components, symbols, symbols)
    ):
        raise ValueError(
            f"Dimensions of x ({x.shape}), log_p ({log_p.shape}), mu ({mu.shape}), and sigma_inv ({sigma_inv.shape}) are inconsistent"
        )

    if symbols != 1:
        raise ValueError(
            f"Symbol dim is {symbols}. This function requires the number of symbols to be 1"
        )

    # Drop the dimensions that were just confirmed to be one.
    x = x.squeeze(1)
    mu = mu.squeeze(2)
    sigma_inv = sigma_inv.squeeze(3).squeeze(2)

    # Subtract mu from x in each component.
    # Be explicit rather than relying on broadcasting
    e = x.unsqueeze(1).expand(mu.shape) - mu

    z_squared = (sigma_inv * e) ** 2

    # Inclusion of EPS is to ensure argument remains bounded away from zero.
    log_sigma_inv = torch.log(EPS_FOR_LOG + torch.abs(sigma_inv))

    # log_p, z_squared, and log_sigma_inv have the same shape: (mb_size, mixture_components)

    ll_components = log_p - 0.5 * z_squared + log_sigma_inv - LOG_SQRT_TWO_PI

    # Now sum over the mixture components with logsumexp to get the liklihoods
    # for each batch sample
    ll = torch.logsumexp(ll_components, dim=1)
    return ll


def multivariate_log_likelihood(
    x: torch.Tensor, log_p: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor
):
    """Inputs:
       x (tensor(mb_size, channels)): values
       log_p (tensor(mb_size, mixture_componente)):
                      log probability of each component (this code assumes
                      these have been normalized with logsumexp!!)
       mu (tensor(mb_size, mixture_components, channels): mu for each component
       sigma_inv (tensor(mb_size, mixture_components, channels, channels)):
                     - sqrt of inverse of covariance matrix
                       (More specifically, the inverse of the lower triangular
                       Cholesky factor of the channel covariances so that
                       C^{-1} = L^T L)


    Output:
       tensor(mb_size): log likelihood for each sample in batch

    """
    mb_size, mixture_components, channels = sigma_inv.shape[:3]
    if (
        x.shape != (mb_size, channels)
        or log_p.shape != (mb_size, mixture_components)
        or mu.shape != (mb_size, mixture_components, channels)
        or sigma_inv.shape != (mb_size, mixture_components, channels, channels)
    ):
        raise ValueError(
            f"Dimensions of x ({x.shape}), log_p ({log_p.shape}), mu ({mu.shape}), and sigma_inv ({sigma_inv.shape}) are inconsistent"
        )

    # Ensure the sigma_inv matrix is lower triangular
    # Values in the upper triangle part get ignored
    sigma_inv = torch.tril(sigma_inv)

    e = x.unsqueeze(1).expand(mu.shape) - mu
    # e is (mb_size, mixture_components, channels)

    e = e.unsqueeze(3)
    # e is now (mb_size, mixture_components, channels, 1)

    z = torch.matmul(sigma_inv, e)
    # z is (mb_size, mixture_components, channels, 1)

    z_squared = torch.sum((z**2).squeeze(3), dim=2)
    # z_squared is (mb_size, mixture_components)

    # print('x: ', x)
    # print('mu: ', mu)
    # print('e: ', e)
    # print('z_squared: ', z_squared)

    # Compute the log of the diagonal entries of the inverse covariance matrix
    # Inclusion of EPS is to ensure argument stays well above zero.
    log_diag_sigma_inv = torch.log(
        EPS_FOR_LOG + torch.abs(torch.diagonal(sigma_inv, 0, -2, -1))
    )
    # log_diag_sigma_inv is (mb_size, mixture_components, channels)

    # Compute the log of the determinant of the inverse covariance
    # matrix by summing the above
    log_det_sigma_inv = torch.sum(log_diag_sigma_inv, dim=2)
    # print('log_det_sigma_inv', log_det_sigma_inv)
    # log_det_sigma_inv is (mb_size, mixture_components)

    ll_components = (
        log_p - 0.5 * z_squared + log_det_sigma_inv - channels * LOG_SQRT_TWO_PI
    )

    # Now sum over the components with logsumexp to get the liklihoods
    # for each batch sample
    ll = torch.logsumexp(ll_components, dim=1)
    return ll


# TODO: Generalize the following function for the multivariate case.


def new_univariate_combine_metrics(p, mu, sigma_inv):
    """
    Given a mixture model of normal distributions charaterized by probabilities
    (p), components-wise mean (mu) and component-wise inverse standard deviation
    (sigma_inv), compute the overall mean and inverse standard deviation for the
    mixture.

    Note:  This assumes a univariate mu and sigma_inv.  It's simpler than the multivariate version.

    Inputs:
        p: tensor of shape (mb_size, mixture_componente): probability of each component
        mu: tensor of shape (mb_size, mixture_components): mu for each
            component.
        sigma_inv: tensor of shape (mb_size, mixture_components) containing
        the inverse of the standard deviation of each component.

    Outputs:
        mu: tensor of shape (mb_size,) containing the expected mean
        variance: tensor of shape (mb_size,) containing the
            variance of the mixture.

        Note that the return value is the variance (i.e., the standard deviation squared) and *not* the inverse
        of the standard deviation that's often used elsewhere in this code.

    """
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float)
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, dtype=torch.float)
    if not isinstance(sigma_inv, torch.Tensor):
        sigma_inv = torch.tensor(sigma_inv, dtype=torch.float)

    if p.shape != mu.shape or p.shape != sigma_inv.shape:
        raise ValueError(
            f"Dimensions of p ({p.shape}), mu ({mu.shape}), and sigma_inv ({sigma_inv.shape}) are inconsistent"
        )

    variance = (1.0 / sigma_inv) ** 2
    composite_mean = torch.sum(p * mu, dim=1)

    # Composite variance comes from the shifted component means and
    # shifted component covariances.  Here's a derivation:

    # E[(x-mu)**2] = sum p_i E[(x_i-mu)**2]
    # E[(x_i-mu)**2] = E[((x_i-mu_i) + (mu_i-mu))**2]
    #                = sigma_i**2 + (mu_i-mu)**2

    shifted_component_means = mu - composite_mean.unsqueeze(1).expand(mu.shape)
    shifted_component_variances = variance + shifted_component_means**2
    composite_variance = torch.sum(p * shifted_component_variances, dim=1)
    return composite_mean, composite_variance


def univariate_combine_metrics(p, mu, sigma_inv):
    """
    Given a mixture model of normal distributions charaterized by probabilities
    (p), components-wise mean (mu) and component-wise inverse standard deviation
    (sigma_inv), compute the overall mean and inverse standard deviation for the
    mixture.

    Note:  This assumes a univariate mu and sigma_inv.  It's simpler than the multivariate version.

    Inputs:
        p: tensor of shape (mb_size, mixture_componente): probability of each component
        mu: tensor of shape (mb_size, mixture_components, 1): mu for each
            component.
        sigma_inv: tensor of shape (mb_size, mixture_components, 1, 1) containing
        the inverse of the standard deviation of each component.

    Outputs:
        mu: tensor of shape (mb_size,) containing the expected mean
        variance: tensor of shape (mb_size,) containing the
            variance of the mixture.

        Note that the return value is the variance (i.e., the standard deviation squared) and *not* the inverse
        of the standard deviation that's often used elsewhere in this code.

    """
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float)
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, dtype=torch.float)
    if not isinstance(sigma_inv, torch.Tensor):
        sigma_inv = torch.tensor(sigma_inv, dtype=torch.float)

    mb_size, mixture_components, symbols = sigma_inv.shape[:3]
    if (
        p.shape != (mb_size, mixture_components)
        or mu.shape != (mb_size, mixture_components, symbols)
        or sigma_inv.shape != (mb_size, mixture_components, symbols, symbols)
    ):
        raise ValueError(
            f"Dimensions of p ({p.shape}), mu ({mu.shape}), and sigma_inv ({sigma_inv.shape}) are inconsistent"
        )

    if symbols != 1:
        raise ValueError(
            f"Symbol dim is {symbol}. This code requires the number of symbols to be 1"
        )

    # Drop the symbol dimension on mu and sigma_inv which is known to be 1
    # for this special case.

    sigma_inv = sigma_inv.squeeze(3).squeeze(2)
    mu = mu.squeeze(2)

    variance = (1.0 / sigma_inv) ** 2
    composite_mean = torch.sum(p * mu, dim=1)

    # Composite variance comes from the shifted component means and
    # shifted component covariances.  Here's a derivation:

    # E[(x-mu)**2] = sum p_i E[(x_i-mu)**2]
    # E[(x_i-mu)**2] = E[((x_i-mu_i) + (mu_i-mu))**2]
    #                = sigma_i**2 + (mu_i-mu)**2

    shifted_component_means = mu - composite_mean.unsqueeze(1).expand(mu.shape)
    shifted_component_variances = variance + shifted_component_means**2
    composite_variance = torch.sum(p * shifted_component_variances, dim=1)
    return composite_mean, composite_variance


def multivariate_combine_metrics(p, mu, sigma_inv):
    """Given a mixture model of normal distributions charaterized by
    probabilities (p), components-wise mean (mu) and component-wise
    inverse standard deviation (sigma_inv), compute the overall mean
    and inverse standard deviation for the mixture.

    Note:  This is the multivariate version of univariate_combine_metrics.

    Inputs:
        p: tensor of shape (mb_size, mixture_componente) - probability of each component
        mu: tensor of shape (mb_size, mixture_components, symbols) - mu for each
            component.
        sigma_inv: tensor of shape (mb_size, mixture_components, symbols, symbols) -
            the inverse of the standard deviation of each component.

    Outputs:
        mu: tensor of shape (mb_size, symbols) - the mean of the mixture.
        covariance: tensor of shape (mb_size, symbols, symbols) - the covariance of the mixture.

        Note that the return value is the covariance matrix.  This is
        different from elsewhere in the code where we often use the
        Cholesky factor of the inverse of the covariance matrix to
        represent the variance.

    """
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float)
    if not isinstance(mu, torch.Tensor):
        mu = torch.tensor(mu, dtype=torch.float)
    if not isinstance(sigma_inv, torch.Tensor):
        sigma_inv = torch.tensor(sigma_inv, dtype=torch.float)

    if (
        p.shape != sigma_inv.shape[:2]
        or mu.shape != sigma_inv.shape[:3]
        or sigma_inv.shape[2] > sigma_inv.shape[3]
    ):
        raise ValueError(
            f"Dimensions of p ({p.shape}), mu ({mu.shape}), and sigma_inv ({sigma_inv.shape}) are inconsistent"
        )

    #  Note that sigma_inv may not be square but the number of rows
    #  should be no more than the number of columns

    inverse_covariance = torch.matmul(sigma_inv, torch.transpose(sigma_inv, 2, 3))
    covariance = torch.inverse(inverse_covariance)
    composite_mean = torch.sum(p.unsqueeze(2).expand(mu.shape) * mu, dim=1)

    # Composite covariance comes from the shifted component means and
    # shifted component covariances.  Here's a derivation:

    # E[(x-mu)(x-mu)'] = sum p_i E[(x_i-mu)(x_i-mu)']
    # But E[(x_i-mu)(x_i-mu)'] = E[((x_i-mu_i)(x_i-mu_i)' + (mu_i-mu)(mu_i-mu)']
    #                          = cov_i+ (mu_i-mu)(mu_i-mu)'

    shifted_means = (mu - composite_mean.unsqueeze(1).expand(mu.shape)).unsqueeze(3)
    shifted_component_variances = covariance + torch.matmul(
        shifted_means, torch.transpose(shifted_means, 2, 3)
    )
    composite_covariance = torch.sum(
        p.unsqueeze(2).unsqueeze(3).expand(shifted_component_variances.shape)
        * shifted_component_variances,
        dim=1,
    )
    return composite_mean, composite_covariance
