# Standard Python
import math
# Common packages
import torch

LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
EPS = 1e-8
SINGULAR_EPS = 1e-4

MAX_SIGMA_ENTRY = torch.tensor([1.0])

# TODO: Write a test for this


def ensure_nonsingular(x):
    """
    Ensure that a lower triangular matrix is non-singular by adding a small perturbation
    when necessary.  The goal here is not to touch entries that don't need a perturbation and
    to make the minimum change to reach the threshhold.
    """
    d = torch.diagonal(x, dim1=2, dim2=3)
    pos = ((d < SINGULAR_EPS) * (d >= 0.0)).type(torch.float32)
    neg = ((d > -SINGULAR_EPS) * (d < 0.0)).type(torch.float32)
    perturbation = pos * (SINGULAR_EPS - d) + neg * (-SINGULAR_EPS - d)
    return x + torch.diag_embed(perturbation)


def combine_mixture_metrics(p, mu, sigma):
    """
    Inputs:
        p (tensor(mb_size, mixture_componente)):
           probability of each component
        mu (tensor(mb_size, mixture_components): mu for each component
        sigma (tensor(mb_size, mixture_components): sigma for each component

    Outputs:
        gain (tensor(mb_size)): expected gain (in log-normal space)
        std_dev (tensor(mb_size)): std deviations (in log-normal sape)
    """
    sigma_squared = sigma**2
    component_means = torch.exp(mu + 0.5 * sigma_squared)
    component_variances = (torch.exp(sigma_squared) - 1.0) * component_means**2
    mean = torch.sum(p * component_means, dim=1)
    # TODO: Verify this is correct implementation of parallel axis theorem
    shifted_component_variances = (component_variances +
                                   (mean.unsqueeze(1) - component_means)**2)
    std_dev = torch.sqrt(torch.sum(p * shifted_component_variances, dim=1))
    return mean - 1.0, std_dev


def get_portfolio_truth(allocation, true_returns):
    x = torch.log(torch.sum(allocation * torch.exp(true_returns), dim=1))
    return x.unsqueeze(1)


def check(t, name):
    if torch.isnan(t).any():
        print('got nan')
        print(name, t)
        return True
    if torch.isinf(t).any():
        print('got inf')
        print(name, t)
        return True
    return False


def get_portfolio_distribution(allocation, mu, sigma_inv):
    """
    Given the assumed allocation, reduce the multivariate log-normal
    distributions characterized by mu and sigma_inv to a univariate
    log-normal mixture representing the entire portfolio

    Inputs:
       allocation (tensor(mb_size, channels): Allocation among channels
       mu (tensor(mb_size, mixture_components, channels): mu for each component
       sigma_inv (tensor(mb_size, mixture_components, channels, channels)):
           - sqrt of inverse of covariance matrix for each component
             (More specifically, the inverse of the lower triangular
             Cholesky factor of the channel covariances so that
             C^{-1} = L^T L)

    Outputs:
       mu (tensor(mb_size, mixture_components, 1))
       sigma_inv (tensor(mb_size, mixture_components, 1, 1))

    """
    mb_size, mixture_components, channels = mu.shape
    check(mu, 'mu')
    check(sigma_inv, 'sigma_inv')

    # Duplicate the allocation across all of the components by
    #    1) Creating a component dimension with unsqueeze(1)
    #    2) Duplicating the contents with expand
    # Discard the originally passed in allocation

    allocation = allocation.unsqueeze(1).expand(
        (mb_size, mixture_components, channels))

    sigma = torch.inverse(ensure_nonsingular(sigma_inv))
    check(sigma, 'sigma')
    # Limit sigma before computing covariance since limiting
    # entries of covariance might not preserve positive definiteness
    sigma = torch.max(torch.min(sigma, MAX_SIGMA_ENTRY), -MAX_SIGMA_ENTRY)

    # covariance is (mb_size, mixture_components, channels, channels)
    # and has the full covariance matrix for each batch element and
    # mixture component
    covariance = torch.matmul(sigma, torch.transpose(sigma, 2, 3))
    check(covariance, 'covariance')

    # marginal_variances is (mb_size, mixture_components, channels) and
    # has the channels maginal_variances for each batch element and
    # mixture component
    marginal_variances = torch.diagonal(covariance, dim1=2, dim2=3)
    check(marginal_variances, 'marginal_variances')

    # mu_z is the mean for the log-normal distribution (whereas mu
    # is the mean for the underlying normal distribution)
    mu_z = torch.exp(mu + 0.5 * marginal_variances)
    if check(mu_z, 'mu_z'):
        print('mu', mu)
        print('marginal_variances', marginal_variances)
        print('sigma', sigma)
        print('sigma_inv', sigma_inv)

    # diag_mu_z contains matrices with mu_z along the diagonal
    diag_mu_z = torch.diag_embed(mu_z)
    check(diag_mu_z, 'diag_mu_z')

    # covariance_z is (mb_size, mixture_components, channels,
    # channels) and represents the covariance of the log-normal
    # distribution (whereas covariance is the covariance for the
    # underlying normal distribution)
    tec = torch.exp(covariance)
    check(tec, 'tec')

    tec1 = tec - 1.0
    check(tec1, 'tec1')

    t1 = torch.matmul(diag_mu_z, tec1)
    if check(t1, 't1'):
        print('diag_mu_z', diag_mu_z)
        print('tec1', tec1)

    covariance_z = torch.matmul(t1, diag_mu_z)
    check(covariance_z, 'covariance_z')

    # mu_p_z is (mb_size, mixture_components, 1) and contains the
    # expected value of the porfolio's log-normal distribution for
    # each batch element and mixture component
    mu_p_z = torch.sum(allocation * mu_z, dim=2).unsqueeze(2)
    check(mu_p_z, 'mu_p_z')

    # variance_p_z is (mb_size, mixture_components, 1, 1) and contains the
    # variance of the porfolio's log-normal distribution for
    # each batch element and mixture component
    variance_p_z = torch.matmul(
        allocation.unsqueeze(2),
        torch.matmul(covariance_z, allocation.unsqueeze(3)))
    check(variance_p_z, 'variance_p_z')

    # mu_p_x is (mb_size, mixture_components, 1) and contains the
    # expected value of the porfolio's log-normal distribution for
    # each batch element and mixture component
    variance_p_x = torch.log(1.0 + variance_p_z.squeeze(3) / mu_p_z**2)
    check(variance_p_x, 'variance_p_x')

    mu_p_x = torch.log(mu_p_z + EPS) - 0.5 * variance_p_x
    check(mu_p_x, 'mu_p_x')

    sigma_inv_p_x = 1 / torch.sqrt(variance_p_x.unsqueeze(3) + EPS)
    if check(sigma_inv_p_x, 'sigma_inv_p_x'):
        print('variance_p_x', variance_p_x)
        print('tec1', tec1)
        print('diag_mu_z', diag_mu_z)
        print('covariance_z', covariance_z)
        print('covariance', covariance)
        print('variance_p_z', variance_p_z)

    return mu_p_x, sigma_inv_p_x


def multivariate_stats(x):
    """Inputs:
         x (tensor(mb_size, channels)): values
       Outputs:
         mu (tensor(channels)): channel means
         sigma (tensor(channels, channels)): cholesky factor of cov matrix
    """
    mb_size, channels = x.shape
    mu = torch.mean(x, dim=0)
    error = x - mu.unsqueeze(0).expand((mb_size, channels))
    # error is mb_size x channels
    error1 = error.unsqueeze(2)
    # error1 represents e (mb_size, channels, 1)
    error2 = error.unsqueeze(1)
    # error2 represents e^T (mb_size, 1, channels)
    cov = torch.mean(torch.matmul(error1, error2), dim=0)
    # cov is (channels, channels)

    # Return cholesky factor
    sigma = torch.cholesky(cov)
    return mu, sigma


def multivariate_mixture_log_likelihood(x, log_p, mu, sigma_inv):
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
    mb_size, channels = x.shape
    mixture_components = mu.shape[1]

    # Ensure the sigma_inv matrix is lower triangular
    # Values in the upper triangle are ignored
    sigma_inv = torch.tril(sigma_inv)

    e = x.unsqueeze(1).expand(mb_size, mixture_components, channels) - mu
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
        EPS + torch.abs(torch.diagonal(sigma_inv, 0, -2, -1)))
    # log_diag_sigma_inv is (mb_size, mixture_components, channels)

    # Compute the log of the determinant of the inverse covariance
    # matrix by summing the above
    log_det_sigma_inv = torch.sum(log_diag_sigma_inv, dim=2)
    # print('log_det_sigma_inv', log_det_sigma_inv)
    # log_det_sigma_inv is (mb_size, mixture_components)

    ll_components = (log_p - 0.5 * z_squared + log_det_sigma_inv -
                     channels * LOG_SQRT_TWO_PI)

    # Now sum over the components with logsumexp to get the liklihoods
    # for each batch sample
    ll = torch.logsumexp(ll_components, dim=1)
    return ll
