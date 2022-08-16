# Standard Python
import datetime as dt
import logging

from typing import Callable, Dict, Iterable, Iterator, Union, Tuple

# Common packages
import click
import pandas as pd

import torch

from deep_volatility_models import data_sources
from deep_volatility_models import stock_data
from deep_volatility_models import time_series_datasets

# Local imports
from mgarch_models import (
    ZeroMeanModel,
    ARMAMeanModel,
    UnivariateARCHModel,
    MultivariateARCHModel,
    ParameterConstraint,
)

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


def run(
    use_hsmd,
    symbols,
    refresh,
    seed=DEFAULT_SEED,
    start_date=None,
    end_date=None,
    eval_start_date=None,
    eval_end_date=None,
    use_univariate=False,
    use_mean=False,
    constraint=ParameterConstraint.FULL,
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
    logging.info(f"Use univariate: {use_univariate}")
    logging.info(f"Use mean: {use_mean}")

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

    if use_mean:
        mean_model = ARMAMeanModel(device=device)
    else:
        mean_model = ZeroMeanModel(device=device)

    if use_univariate:
        univariate_model = UnivariateARCHModel(mean_model=mean_model, device=device)
    else:
        univariate_model = None

    multivariate_model = MultivariateARCHModel(
        univariate_model=univariate_model,
        constraint=constraint,
        device=device,
    )
    multivariate_model.fit(observations)

    # Predict one more time with optimal parameters.
    h, h_next, sigma, sigma_next = multivariate_model.predict(observations)
    print("h: ", h.shape)

    # Compute some useful quantities to display and to record
    covariance = h @ torch.transpose(h, 1, 2)
    predicted_covariance = h_next @ h_next.T

    sigma = torch.sqrt(torch.diagonal(covariance, dim1=1, dim2=2))
    predicted_sigma = torch.sqrt(torch.diag(predicted_covariance))

    inverse_sigma = torch.diag_embed(sigma ** (-1), dim1=1, dim2=2)
    inverse_predicted_sigma = torch.diag(predicted_sigma ** (-1))
    correlation = inverse_sigma @ covariance @ inverse_sigma
    predicted_correlation = (
        inverse_predicted_sigma @ predicted_covariance @ inverse_predicted_sigma
    )

    result = {
        "transformation": h.numpy(),
        "predicted_transformation": h_next.numpy(),
        "sigma": sigma.numpy(),
        "predicted_sigma": predicted_sigma.numpy(),
        "covariance": covariance.numpy(),
        "predicted_covariance": predicted_covariance.numpy(),
        "correlation": correlation.numpy(),
        "predicted_correlation": predicted_correlation.numpy(),
        "training_data": training_data,
    }

    torch.save(result, "mgarch_output.pt")

    logging.info(f"Transformation estimates:\n{h}")
    logging.info(f"sigma:\n{sigma}")
    logging.info(f"correlations:\n{correlation}")

    # Compute the final loss
    ll = multivariate_model.mean_log_likelihood(observations)

    logging.info(f"**** Final likelihood (per sample): {ll:.4f} ****")


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
@click.option(
    "--use-univariate",
    is_flag=True,
    default=None,
    show_default=True,
    help="Use a univariate 'garchs' as a prescalar",
)
@click.option(
    "--use-mean",
    is_flag=True,
    default=None,
    show_default=True,
    help="Use an ARMA mean model rather than just zero.",
)
@click.option(
    "--constraint",
    "-c",
    type=click.Choice(
        ["full", "triangular", "diagonal", "scalar"], case_sensitive=False
    ),
    default="full",
    help="Type of constraint to be applied to multivariate parameters.",
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
    use_univariate,
    use_mean,
    constraint,
):

    constraints = {
        p.value: p
        for p in [
            ParameterConstraint.FULL,
            ParameterConstraint.TRIANGULAR,
            ParameterConstraint.DIAGONAL,
            ParameterConstraint.SCALAR,
        ]
    }

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
        use_univariate=use_univariate,
        use_mean=use_mean,
        constraint=constraints[constraint],
    )


if __name__ == "__main__":
    main_cli()
