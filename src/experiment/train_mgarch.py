# Standard Python
import datetime as dt
import logging

import os
from typing import Callable, Dict, Iterable, Iterator, Union, Tuple

# Common packages
import click
import numpy as np
import pandas as pd

import torch

# Local imports
from deep_volatility_models import data_sources
from deep_volatility_models import stock_data
from deep_volatility_models import time_series_datasets

DEBUG = False

PROGRESS_ITERATIONS = 20
DIAGONAL_MODEL = False
DECAY_FOR_INITIALIZATION = 0.30
LEARNING_RATE = 0.25

MAX_CLAMP = 1e10
MIN_CLAMP = -MAX_CLAMP

DEFAULT_SEED = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    force=True,
)

if torch.cuda.is_available():
    dev = "cuda:0"
# elif torch.has_mps:
#    dev = "mps"
else:
    dev = "cpu"

device = torch.device(dev)


def prepare_data(
    history_loader: Callable[
        [Union[str, Iterable[str]]], Iterator[Tuple[str, pd.DataFrame]]
    ],
    symbol_list: Iterable[str],
    start_date: Union[dt.date, None] = None,
    end_date: Union[dt.date, None] = None,
    eval_start_date: Union[dt.date, None] = None,
    eval_end_date: Union[dt.date, None] = None,
):
    # Refresh historical data
    logging.info("Reading historical data")

    combiner = stock_data.PriceHistoryConcatenator()

    symbol_list = sorted(symbol_list)
    full_history = combiner(history_loader(symbol_list))
    training_data = full_history.loc[start_date:end_date, (symbol_list, "log_return")]
    print(training_data.columns)

    return training_data


def make_diagonal_nonnegative(m):
    """Given a single lower triangular matrices m, return an `equivalent` matrix
    having non-negative diagonal entries.  Here `equivalent` means m@m.T is unchanged.
    """
    diag = torch.diag(m)
    diag_signs = torch.ones(diag.shape)
    diag_signs[diag < 0.0] = -1
    return m * diag_signs


def random_lower_triangular(n, scale=1.0, requires_grad=True):
    scale = torch.tensor(scale, device=device)
    m = torch.rand(n, n, device=device) - 0.5
    m = scale * make_diagonal_nonnegative(torch.tril(m))
    m.requires_grad = requires_grad
    return m


def initial_univariate_parameters(n, device=None):
    a = (1.0 - DECAY_FOR_INITIALIZATION) * torch.ones(n, device=device)
    b = DECAY_FOR_INITIALIZATION * torch.ones(n, device=device)
    c = torch.ones(n, device=device)
    d = torch.ones(n, device=device)

    return a, b, c, d


def initial_multivariate_parameters(n, device=None):
    # Initialize a and b as simple multiples of the identity
    a = (1.0 - DECAY_FOR_INITIALIZATION) * torch.eye(n, device=device)
    b = DECAY_FOR_INITIALIZATION * torch.eye(n, device=device)

    # c is triangular
    c = random_lower_triangular(n, 0.01, False)
    d = torch.eye(n, device=device)

    if DIAGONAL_MODEL:
        a = torch.diag(torch.diag(a))
        b = torch.diag(torch.diag(b))
        c = torch.diag(torch.diag(c))
        d = torch.diag(torch.diag(d))

    # We set requires_grad here so that the preceeding diags and tril calls
    # aren't subject to differentiation

    for t in [a, b, c, d]:
        t.requires_grad = True

    return a, b, c, d


normal_distribution = torch.distributions.normal.Normal(
    loc=torch.tensor(0.0), scale=torch.tensor(1.0)
)


def conditional_log_likelihoods(observations, transformations, distribution):
    """
    Arguments:
       observations: torch.Tensor of shape (n_obs, n_symbols)
       transformations: torch.Tensor of shape (n_symbols, n_symbols)
           transformation is a lower-triangular matrix and the outcome is presumed
           to be z = transformation @ e where the elements of e are iid from distrbution.
       distribution: torch.distributions.distribution.Distribution instance which should have a log_prob() method.
           Note we assume distrubution was constructed with center=0 and shape=1.  Any normalizing and recentering
           is achieved by explicit`transformations` here.

    Returns vector of log_likelihoods"""

    # Make a tensor containing stacked identity matrices so that we can
    # use solve_triangular() rather than inverting some lower-trangular matirces.
    # solve_triangular() should be more efficient.
    identities = (
        torch.eye(observations.shape[1])
        .unsqueeze(0)
        .expand(observations.shape[0], observations.shape[1], observations.shape[1])
    )

    # inv_t = torch.inverse(transformations)
    inv_t = torch.linalg.solve_triangular(
        transformations,
        identities,
        upper=False,
    )

    # The unsqueeze is necessary because the matmul is on the 'batch'
    # of observations.  The shape of `inv_t` is (n_obs, n, n) while
    # the shape of `observations` is (n_obs, n) Without adding the
    # extract dimension to observations, matmul doesn't understand
    # what we're trying to multiply.  We remove the ambiguity, we make
    # `observations` have shape (n_obj, n, 1)
    observations = observations.unsqueeze(2)

    # Do the multiplication, then drop the extra dimension
    e = (inv_t @ observations).squeeze(2)

    logging.debug(f"e: \n{e}")

    # Compute the log likelihoods on the innovations
    log_pdf = torch.sum(distribution.log_prob(e), dim=1)

    # Divide by the determinant by subtracting its log to get the the log
    # likelihood of the observations.  The `transformations` are lower-triangular, so
    # only the diagonal entries need to be used in the determinant calculation.
    # This should be faster than calling log_det().

    log_det = torch.sum(
        torch.log(torch.abs(torch.diagonal(transformations, dim1=1, dim2=2))), dim=1
    )

    ll = log_pdf - log_det

    return torch.mean(ll)


def multivariate_simulate(observations, a, b, c, d, h_bar):
    ht = d @ h_bar
    h_sequence = []

    for k, o in enumerate(observations):
        # Store the current ht before predicting next one
        h_sequence.append(ht)

        # While searching over the parameter space an unstable value for `a` may be tested.
        # Clamp ht to prevent it from overflowing.

        a_ht = torch.clamp(a @ ht, min=MIN_CLAMP, max=MAX_CLAMP)
        b_o = (b @ o).unsqueeze(1)

        # The covariance is a_ht @ a_ht.T + b_o @ b_o.T + c @ c.Tp
        # Unnecessary squaring is discouraged for nunerical stability.
        # Instead, we use only square roots and never explicity
        # compute the covariance.  This is a common 'trick' achieved
        # by concatenating the square roots in a larger array and
        # computing the QR factoriation, which computes the square
        # root of the sum of squares.  The covariance matrix isn't
        # formed explicitly in this code except at the very end when
        # it's time to return the covariance matrices to the user.

        m = torch.cat((a_ht, b_o, c @ h_bar), axis=1)

        # Unfortunately there's no QL factorization in PyTorch so we
        # transpose m and use the QR.  We only need the 'R' return
        # value, so the Q return value is dropped.

        ht_t = torch.linalg.qr(m.T, mode="reduced")[1]

        # Transpose ht to get the lower triangular version.

        ht = make_diagonal_nonnegative(ht_t.T)

    h = torch.stack(h_sequence)
    return h


def univariate_log_likelihood(observations, a, b, c, d, sample_var, distribution):
    logging.debug(f"uv likelihood: a:\n{a}")
    logging.debug(f"uv likelihood: b:\n{b}")
    logging.debug(f"uv likelihood: c:\n{c}")
    logging.debug(f"uv likelihood: d:\n{c}")

    var = univariate_simulate(observations, a, b, c, d, sample_var)

    if h is not None:
        mean_ll = conditional_log_likelihoods(observations, var, distribution)
    else:
        mean_ll = torch.tensor(float("-inf"), requires_grad=True)

    return mean_ll


def multivariate_log_likelihood(observations, a, b, c, d, h_bar, distribution):
    # Running through a tril() will force autograd to compute a zero gradient
    # for the upper triangular portion so that those entries are ignored.
    a = torch.tril(a)
    b = torch.tril(b)
    c = torch.tril(c)
    d = torch.tril(d)

    if DIAGONAL_MODEL:
        a = torch.diag(torch.diag(a))
        b = torch.diag(torch.diag(b))
        c = torch.diag(torch.diag(c))
        d = torch.diag(torch.diag(d))

    logging.debug(f"likelihood: a:\n{a}")
    logging.debug(f"likelihood: b:\n{b}")
    logging.debug(f"likelihood: c:\n{c}")
    logging.debug(f"likelihood: d:\n{c}")

    h = multivariate_simulate(observations, a, b, c, d, h_bar)

    if h is not None:
        mean_ll = conditional_log_likelihoods(observations, h, distribution)
    else:
        mean_ll = torch.tensor(float("-inf"), requires_grad=True)

    return mean_ll


def run(
    use_hsmd,
    symbols,
    refresh,
    seed=DEFAULT_SEED,
    start_date=None,
    end_date=None,
    eval_start_date=None,
    eval_end_date=None,
):
    # Rewrite symbols with deduped, uppercase versions
    symbols = list(map(str.upper, set(symbols)))

    logging.info(f"device: {device}")
    logging.info(f"symbols: {symbols}")
    logging.info(f"refresh: {refresh}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Start date: {start_date}")
    logging.info(f"End date: {end_date}")
    logging.info(f"Evaluation/termination start date: {eval_start_date}")
    logging.info(f"Evaluation/termination end date: {eval_end_date}")

    distribution = normal_distribution

    data_store = stock_data.FileSystemStore("training_data")
    if use_hsmd:
        data_source = data_sources.HugeStockMarketDatasetSource(use_hsmd)
    else:
        data_source = data_sources.YFinanceSource()

    history_loader = stock_data.CachingSymbolHistoryLoader(
        data_source, data_store, refresh
    )

    torch.random.manual_seed(seed)

    training_data = prepare_data(
        history_loader,
        symbols,
        start_date=start_date,
        end_date=end_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
    )
    logging.info(f"training_data:\n {training_data}")

    observations = torch.tensor(training_data.values, dtype=torch.float, device=device)
    logging.info(f"observations:\n {observations}")

    a, b, c, d = initial_multivariate_parameters(len(symbols), device=device)
    logging.debug(f"a:\n{a}")
    logging.debug(f"b:\n{b}")
    logging.debug(f"c:\n{c}")
    logging.debug(f"d:\n{d}")

    au, bu, cu, du = initial_univariate_parameters(len(symbols), device=device)
    logging.debug(f"au:\n{au}")
    logging.debug(f"bu:\n{bu}")
    logging.debug(f"cu:\n{cu}")
    logging.debug(f"du:\n{du}")

    h_bar = (torch.linalg.qr(observations, mode="reduced")[1]).T / torch.sqrt(
        torch.tensor(observations.shape[0])
    )
    h_bar = make_diagonal_nonnegative(h_bar)
    logging.info(f"h_bar:\n{h_bar}")

    sample_var = torch.var(observations, dim=0)
    logging.info(f"sample_var:\n{sample_var}")

    def loss_closure():
        optim.zero_grad()
        loss = -multivariate_log_likelihood(
            observations, a, b, c, d, h_bar, distribution
        )
        loss.backward()
        return loss

    optim = torch.optim.LBFGS(
        [a, b, c, d],
        max_iter=PROGRESS_ITERATIONS,
        lr=LEARNING_RATE,
        line_search_fn="strong_wolfe",
    )

    logging.info("Starting optimization.")
    best_loss = float("inf")
    done = False

    while not done:
        optim.step(loss_closure)
        with torch.no_grad():
            current_loss = -multivariate_log_likelihood(
                observations, a, b, c, d, h_bar, distribution
            )
            logging.info(
                f"\tcurrent loss: {current_loss:.4f}   previous best loss: {best_loss:.4f}"
            )

        if float(current_loss) < best_loss:
            best_loss = current_loss
        else:
            logging.info("Stopping")
            done = True

    logging.info("Finished")

    # Simulate one more time with optimal parameters.
    h = multivariate_simulate(observations, a, b, c, d, h_bar)

    # Compute some useful quantities to display and to record
    covariance = h @ torch.transpose(h, 1, 2)
    variance = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2))
    inverse_variance = torch.diag_embed(variance ** (-1), dim1=1, dim2=2)
    correlation = inverse_variance @ covariance @ inverse_variance

    result = {
        "transformation": h.detach().numpy(),
        "variance": variance.detach().numpy(),
        "covariance": covariance.detach().numpy(),
        "correlation": correlation.detach().numpy(),
        "training_data": training_data,
    }

    torch.save(result, "mgarch_output.pt")

    print("variances:\n", variance)
    print("correlations:\n", correlation)
    print("transformations:\n", h)

    # Compute the final loss
    ll = conditional_log_likelihoods(observations, h, distribution)

    logging.info(f"Transformation estimates:\n{h}")

    logging.info(f"AR matrix a:\n{a}")
    logging.info(f"MA matrix b:\n{b}")
    logging.info(f"h_bar multiplier for constant term c:\n{c}")
    logging.info(f"h_bar multiplier for initial estimate d:\n{d}")

    logging.info(f"**** Final likelihood (per sample): {ll:.4f} ****")

    logging.debug("Gradients: ")
    logging.debug(f"a.grad:\n{a.grad}")
    logging.debug(f"b.grad:\n{b.grad}")
    logging.debug(f"c.grad:\n{c.grad}")
    logging.debug(f"d.grad:\n{d.grad}")


@click.command()
@click.option(
    "--use-hsmd",
    default=None,
    show_default=True,
    help="Use huge stock market dataset if specified zip file (else use yfinance)",
)
@click.option("--symbol", "-s", multiple=True, show_default=True)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    show_default=True,
    help="Refresh stock data",
)
@click.option("--seed", default=DEFAULT_SEED, show_default=True, type=int)
@click.option(
    "--start-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date of data used for training",
)
@click.option(
    "--end-date",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Final date of data used for training",
)
@click.option(
    "--eval-start-date",
    default=None,
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First date of data used for evaluation/termination",
)
@click.option(
    "--eval-end-date",
    show_default=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last date of data used for evaluation/termination",
)
def main_cli(
    use_hsmd,
    symbol,
    refresh,
    seed,
    start_date,
    end_date,
    eval_start_date,
    eval_end_date,
):

    if start_date:
        start_date = start_date.date()

    if end_date:
        end_date = end_date.date()

    run(
        use_hsmd,
        symbols=symbol,
        refresh=refresh,
        seed=seed,
        start_date=start_date,
        end_date=end_date,
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,
    )


if __name__ == "__main__":
    main_cli()
