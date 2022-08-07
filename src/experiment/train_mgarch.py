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

EPS = 1e-10
LEARNING_RATE = 0.75
DEFAULT_SEED = 42
MAX_ITERATIONS = 10000
DEBUG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    force=True,
)

if torch.cuda.is_available():
    dev = "cuda:0"
# elif torch.has_mps:
#     dev = "mps"
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

    return training_data


# Model is
# h_k = w + h_{k-1} * f + |e_{k-1}| * g
# where h is intended to be the covariance estimate
# f and g are lower triangular.

DECAY = 0.30


def initialize_parameters(n, observations):
    std = torch.std(observations, dim=0)

    def make_one(scale=1.0, requires_grad=True):
        scale = torch.tensor(scale)
        m = torch.randn(n, n) - 0.5
        diag_signs = torch.diag(torch.sign(torch.diag(m)))
        m = scale * torch.tril(diag_signs @ m)
        m.requires_grad = requires_grad
        return m

    w = make_one(0.01)

    f = (1.0 - DECAY) * torch.eye(n)
    g = DECAY * torch.eye(n)
    h0 = torch.diag(std)

    print(h0)

    for t in [f, g, w, h0]:
        t.requires_grad = True

    return f, g, w, h0


distribution = torch.distributions.normal.Normal(
    loc=torch.tensor(0.0), scale=torch.tensor(1.0)
)

print("lll@1.0: ", distribution.log_prob(torch.tensor(1.0)))


def fix_negative_determinant(m):
    """The log loss of transformed variables depends on the absolute value of the determinant.
    This function makes sure the determinant is >= 0"""
    diag_signs = torch.sign(torch.diagonal(m, dim1=1, dim2=2))
    diag_signs_as_matrix = torch.diag_embed(diag_signs, dim1=1, dim2=2)
    return diag_signs_as_matrix @ m


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

    inv_t = torch.inverse(transformations)
    # The unsqueeze is necessary because the matmul is on the 'batch'
    # of observations.  The shape of `inv_t` is (n_obs, n, n) while
    # the shape of `observations` is (n_obs, n) Without adding the
    # extract dimension to observations, matmul doesn't understand
    # what we're trying to multiply.  We remove the ambiguity, we make
    # `observations` have shape (n_obj, n, 1)
    observations = observations.unsqueeze(2)
    # Do the multiplication, then drop the extra dimension
    e = (inv_t @ observations).squeeze(2)

    print("e: \n", e)

    # Compute the log likelihoods on the innovations
    log_pdf = torch.sum(distribution.log_prob(e), dim=1)
    # Divide by the determinant by subtracting its log to get the the log
    # likelihood of the observations.  The `transformations` are lower-triangular, so
    # this calculation should be resonably fast.
    log_det = torch.logdet(fix_negative_determinant(transformations))

    ll = log_pdf - log_det
    return ll


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
    print(f"training_data:\n {training_data}")

    observations = torch.tensor(training_data.values, dtype=torch.float)
    print(f"observations:\n {observations}")

    f, g, w, h0 = initialize_parameters(len(symbols), observations)
    print(f"h0:\n{h0}")
    print(f"f:\n{f}")
    print(f"g:\n{g}")
    print(f"w:\n{w}")

    parameters = [f, g, w, h0]

    optim = torch.optim.LBFGS(
        parameters,
        max_iter=MAX_ITERATIONS,
        lr=LEARNING_RATE,
        line_search_fn="strong_wolfe",
    )

    def simulate(observations, f, g, w, h0):
        hk = h0
        h_sequence = []
        for k, o in enumerate(observations):
            # Store the current hk before predicting next one
            h_sequence.append(hk)
            t1 = hk @ f
            t2 = torch.diag(o) @ g
            h_squared = t1 @ t1.T + t2 @ t2.T + w @ w.T
            try:
                hk = torch.linalg.cholesky(
                    h_squared + EPS * torch.eye(h_squared.shape[0])
                )
            except Exception as e:
                print(e)
                return None

        h = torch.stack(h_sequence)
        return h

    def mean_log_likelihood(observations, f, g, w, h0):
        # Running through a tri will force autograd to ignore any upper entries
        f = torch.tril(f)
        g = torch.tril(g)
        w = torch.tril(w)
        h0 = torch.tril(h0)

        print("likelihood: f:\n", f)
        print("likelihood: g:\n", g)
        print("likelihood: w:\n", w)
        print("likelihood: h0:\n", h0)

        h = simulate(observations, f, g, w, h0)

        if h is not None:
            ll = conditional_log_likelihoods(observations, h, distribution)
            mean_ll = torch.mean(ll)
        else:
            mean_ll = torch.tensor(float("-inf"), requires_grad=True)

        print(f"mean_ll: {mean_ll}")

        return mean_ll

    def loss_closure():
        optim.zero_grad()
        loss = -mean_log_likelihood(observations, f, g, w, h0)
        loss.backward()
        return loss

    logging.info("Starting optimization.")

    optim.step(loss_closure)

    logging.info("Finished")

    # Simulate one more time with optimal parameters.
    h = simulate(observations, f, g, w, h0)

    # Compute the final loss
    ll = torch.mean(conditional_log_likelihoods(observations, h, distribution))

    logging.info(f"Transformation estimates:\n{h}")

    logging.info(f"Initial estimate h0:\n{h0}")
    logging.info(f"AR matrix f:\n{f}")
    logging.info(f"MA matrix g:\n{g}")
    logging.info(f"Constant matrix w:\n{w}")

    logging.info(f"**** Final likelihood (per sample): {ll:.4f} ****")

    logging.info("Gradients: ")
    logging.info(f"f.grad:\n{f.grad}")
    logging.info(f"g.grad:\n{g.grad}")
    logging.info(f"h0.grad:\n{h0.grad}")
    logging.info(f"w.grad:\n{w.grad}")


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
