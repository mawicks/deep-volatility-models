# Common packages
import torch


def multivariate_mixture_sample(
    mixture_model: torch.nn.Module,
    window: torch.Tensor,
    sample_size: int,
    normalize: bool = False,
    n_sigma=1,
):
    """
    Draw samples from a mixture model
    Parameters:
        mixture_model: torch.nn.Module - The model to evaluate_model
        window: torch.Tensor of shape (batch_size, symbols, window_size)
        sample_size: int - The number of samples to draw
        normalize: bool - Draw samples that are a fixed number of standard
        deviations away (useful for generating multivariate contours of points that are
        n-sigma from the mean, but not useful for univariate distributions).
        n_sigma: int - The number of standard deviations away to generate
        samples (only applicable when `normalize` is True)

    Returns:
        torch.Tensor of shape (batch_size, symbols, sample_size) - Log returns
        sampled from the model's distribution

    """
    log_p, mu, sigma_inv = mixture_model(window)[:3]
    p = torch.exp(log_p)

    batch_size, _, symbols = mu.shape

    samples = torch.tensor([])
    for _ in range(sample_size):
        selections = torch.multinomial(p, 1)
        mu_selector = selections.unsqueeze(2).expand(batch_size, 1, symbols)
        selected_mu = torch.gather(mu, 1, mu_selector).squeeze(1).unsqueeze(2)
        # selected_mu is (nb_size x channels x 1)
        assert selected_mu.shape == (batch_size, symbols, 1)

        sigma_selector = (
            selections.unsqueeze(2).unsqueeze(3).expand(batch_size, 1, symbols, symbols)
        )
        selected_sigma_inv = torch.gather(sigma_inv, 1, sigma_selector)
        selected_sigma = torch.inverse(selected_sigma_inv).squeeze(1)
        # selected_sigma is (nb_size x channels x channels)
        assert selected_sigma.shape == (batch_size, symbols, symbols)

        z = torch.randn(batch_size, symbols, 1)
        if normalize:
            norm_z = (
                torch.norm(z, p=2, dim=1).unsqueeze(1).expand(batch_size, symbols, 1)
            )
            z = n_sigma * z / norm_z
            assert z.shape == (batch_size, symbols, 1)

        next_values = selected_mu + torch.matmul(selected_sigma, z)
        # next_values is (mb_size x channels)
        assert next_values.shape == (batch_size, symbols, 1)

        samples = torch.cat((samples, next_values), dim=2)

    return samples.detach()


def multivariate_mixture_simulate(
    mixture_model: torch.nn.Module, window: torch.Tensor, time_samples: int
):
    """
    For each row of `window`, generate simulated log returns for `time_samples` intervals

    Parameters:
        mixture_model: torch.nn.Module - model to evaluate
        window: torch.Tensor of shape (minibatch, symbols, window_size)
        time_samples: number of time intervals to simulate.
    Returns:
        torch.Tensor of shape (batch_size, symbols, samples) - Time series
        containing the simulated log returns.
    """
    # Create an initial simulation day having returns of zero for day
    # 0.  By day 0, we mean "right now" so the returns are zero compared
    # relative to the current stock price.  It may seem unnecessary to
    # explicitly add these zeros, but it's really convenient to be able to index
    # into the simulation with day index==0 meaning the current stock price.
    # The simulation results are typically evaluated by cumsum, exponentiated,
    # and multiplied by the current stock price.  Using this approach, the 0th
    # entry (the price on day 0) will be the current price because a log return of
    # zero has been applied.  This avoids having to do some awkward indexing
    # elsewhere.

    batches, symbols = window.shape[:2]
    simulation = torch.zeros(batches, symbols, 1)

    for _ in range(time_samples):
        next_values = multivariate_mixture_sample(mixture_model, window, 1)
        print("next_values: ", next_values)
        window = torch.cat([window[:, :, 1:], next_values], dim=2)
        simulation = torch.cat((simulation, next_values), dim=2)

    return simulation


def multivariate_mixture_simulate_many(
    mixture_model: torch.nn.Module,
    window: torch.Tensor,
    time_samples: int,
    simulation_count: int,
):
    """
    This is a wrapper that calls multiariate_mixture_simulate `simulation_count` times.
    """

    simulations = torch.stack(
        tuple(
            multivariate_mixture_simulate(mixture_model, window, time_samples)
            for _ in range(simulation_count)
        )
    )
    return simulations


def multivariate_mixture_simulate_extremes(
    mixture_model: torch.nn.Module,
    window: torch.Tensor,
    time_samples: int,
    simulation_count: int,
):

    simulations = multivariate_mixture_simulate_many(
        mixture_model, window, time_samples, simulation_count
    )

    cumsums = torch.cumsum(simulations, dim=3)
    max_outcomes = torch.max(cumsums, dim=0)[0]
    min_outcomes = torch.min(cumsums, dim=0)[0]
    median_outcomes = torch.median(cumsums, dim=0)[0]
    return min_outcomes, median_outcomes, max_outcomes
