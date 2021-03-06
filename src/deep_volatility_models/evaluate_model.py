#  Standard Python

import datetime as dt
import logging
import os
import pickle
import sys
import traceback

# Common packages
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

plt.style.use("ggplot")

# import cufflinks as cf
# from IPython.display import display,HTML

# Local imports
from deep_volatility_models import data_sources
from deep_volatility_models import embedding_models
from deep_volatility_models import sample
from deep_volatility_models import mixture_model_stats
from deep_volatility_models import stock_data
from deep_volatility_models import time_series_datasets


pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.min_rows", None)
pd.set_option("display.max_rows", 1500)

# Configure external packages and run()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO, force=True)

# Torch configuration
torch.set_printoptions(
    precision=4, threshold=None, edgeitems=None, linewidth=None, profile="short"
)

ANNUAL_TRADING_DAYS = 252.0
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))

TIME_SAMPLES = 98


def simulate(model, symbol, window, current_price, simulations):
    """
    Arguments:
        model: torch.nn.Module
        symbol: str
        window: single input row as a torch.Tensor of shape (symbols, window_size)
    """
    # Create a batch dimension (we'll doing a single row, so the batch dimension is one):
    window = window.unsqueeze(0)

    logging.info(f"{symbol} window: {window.shape}")
    logging.info(f"{symbol} window]: {window}")

    simulated_returns = model.simulate_one(window, TIME_SAMPLES)
    simulated_returns_many = sample.simulate_many(
        model, window, TIME_SAMPLES, simulations
    )

    logging.info(f"{symbol} simulated_returns]: {simulated_returns}")

    historic_returns = np.exp(np.cumsum(window.squeeze(1).squeeze(0).numpy()))
    simulated_returns_many = simulated_returns_many.squeeze(1).squeeze(0).numpy()
    logging.info(f"mean return: {np.mean(simulated_returns_many)}")
    sample_index = list(
        range(
            len(historic_returns) - 1,
            len(historic_returns) + len(simulated_returns_many) - 1,
        )
    )
    plt.plot(
        current_price * historic_returns / historic_returns[-1],
        color="k",
        alpha=0.5,
        label=f"Time Series Input ({symbol})",
    )
    colors = ["c", "m"]
    for _ in range(2):
        plt.plot(
            sample_index,
            current_price * simulated_returns_many[:, _],
            f"{colors[_]}",
            alpha=0.5,
            label=f"Sampled Prediction #{_+1}",
        )
    plt.xlabel("Time (days)")
    plt.ylabel("Price ($)")

    max_return = np.percentile(simulated_returns_many, 95.0, axis=1)
    min_return = np.percentile(simulated_returns_many, 5.0, axis=1)

    plt.plot(
        sample_index,
        current_price * max_return,
        "b-",
        alpha=0.3,
        label="95th Percentile Price (Est)",
    )
    plt.plot(
        sample_index,
        current_price * min_return,
        "r-",
        alpha=0.3,
        label="5th Percentile Price (Est)",
    )
    plt.legend(loc="lower left")
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    current_aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    print(xlim)
    print(ylim)
    print("current_aspect: ", current_aspect)
    ax.set_aspect(0.5 * current_aspect)
    plt.savefig(f"model_evaluation_{symbol}@2x.png", dpi=200)
    plt.show()


def do_one_symbol(
    symbol,
    model,
    refresh,
    simulations,
):
    logging.info(f"symbol: {symbol.upper()}")
    logging.info(f"model: {model}")
    logging.info(f"refresh: {refresh}")

    window_size = model.network.window_size

    # FIXME when we're certain the model file was saved in eval mode.
    model.network.eval()

    logging.info(f"model epochs:\t{model.epochs}")
    logging.info(f"model loss:\t{model.loss:.4f}")
    logging.info(f"model symbols:\t{model.symbols}")

    # Refresh historical data
    logging.info("Reading historical data")

    data_store = stock_data.FileSystemStore(os.path.join(ROOT_PATH, "current_data"))
    data_source = data_sources.YFinanceSource()
    history_loader = stock_data.CachingSymbolHistoryLoader(
        data_source, data_store, overwrite_existing=True
    )
    # The Cachingloader returns a sequence of (symbol, data).
    # Since we pass just one symbol rather than a list, use
    # next to grab the first (symbol, dataframe) pair, then [1] to grab the data.
    symbol_history = next(history_loader(symbol))[1]
    current_price = symbol_history.close[-1]
    windowed_returns = time_series_datasets.RollingWindow(
        symbol_history.log_return,
        window_size,
        create_channel_dim=True,
    )
    logging.debug(f"{symbol} windowed_returns[0]: {windowed_returns[0].shape}")
    logging.debug(f"{symbol} windowed_returns[0]: {windowed_returns[0]}")

    simulate(model.network, symbol, windowed_returns[-1], current_price, simulations)

    with torch.no_grad():

        windows = torch.stack(tuple(windowed_returns), dim=0)
        logging.debug(f"{symbol} windows: {windows.shape}")

        dates = symbol_history.index[window_size - 1 :]
        logging.info(f"last date is {dates[-1]}")

        if model.network.is_mixture:
            log_p, mu, sigma_inv = model.network(windows)[:3]
            p = torch.exp(log_p)

            logging.info(f"p: {p}")
            logging.debug(f"mu: {mu}")
            logging.debug(f"sigma_inv: {sigma_inv}")

            mean, variance = mixture_model_stats.univariate_combine_metrics(
                p, mu, sigma_inv
            )
        else:
            mu, sigma_inv = model.network(windows)[:2]

            logging.debug(f"mu: {mu}")
            logging.debug(f"sigma_inv: {sigma_inv}")

            sigma = torch.inverse(sigma_inv)
            mean = mu.squeeze(1)
            variance = (sigma.squeeze(2).squeeze(1)) ** 2
            p = torch.ones((mean.shape[0],))

        annual_return = ANNUAL_TRADING_DAYS * mean
        daily_std_dev = np.sqrt(variance)
        volatility = np.sqrt(ANNUAL_TRADING_DAYS) * daily_std_dev

        logging.debug(f"daily mean: {mean}")
        logging.debug(f"daily std_dev: {daily_std_dev}")

        logging.debug(f"annual return: {annual_return}")
        logging.debug(f"annual volatility: {volatility}")

        df = pd.DataFrame(
            {
                "pred_volatility": volatility,
                "pred_return": map(
                    lambda x: x.numpy(), mean
                ),  # Hack so it will print but won't plot
                "pred_sigma": daily_std_dev,
                "p": map(lambda x: x.numpy(), p),
                "mu": map(lambda x: x.numpy(), mu),
                "sigma_inv": map(lambda x: x.numpy(), sigma_inv),
            },
            index=dates,
        )

        df = df.merge(
            symbol_history,
            how="left",
            left_index=True,
            right_index=True,
        )

        df = df[
            [
                "pred_volatility",
                "log_return",
                "close",
                "pred_return",
                "pred_sigma",
                "p",
                "mu",
                "sigma_inv",
            ]
        ]

        return_df = df[
            ["log_return", "pred_return", "pred_volatility", "p", "mu", "sigma_inv"]
        ]
        return return_df


def run(model, symbol, simulations):
    wrapped_model = torch.load(model)
    single_symbol_model_factory = embedding_models.SingleSymbolModelFactory(
        wrapped_model.encoding, wrapped_model
    )

    # symbols_to_process = list(set(symbol).difference(exclude_symbols))
    symbols_to_process = sorted(list(set(symbol)))
    logging.info(f"symbols_to_process: {symbols_to_process}")

    dataframes = {}
    for s in symbols_to_process:
        df = do_one_symbol(s, single_symbol_model_factory(s.upper()), True, simulations)
        dataframes[s] = df

    combined_df = pd.concat(
        dataframes.values(), keys=dataframes.keys(), axis=1
    ).dropna()

    return combined_df


@click.command()
@click.option(
    "--model",
    show_default=True,
    help="Model file to use.",
)
@click.option(
    "--symbol",
    multiple=True,
    show_default=True,
    help="Load model for this symbol.",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    show_default=True,
    help="Start date",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=None,
    show_default=True,
    help="End date",
)
@click.option(
    "--simulations",
    type=int,
    show_default=True,
    default=10,
    help="Number of simulations to run",
)
def run_cli(
    model,
    symbol,
    start_date,
    end_date,
    simulations,
):
    logging.info(f"model: {model}")
    logging.info(f"symbol: {symbol}")
    logging.info(f"start_date: {start_date}")
    logging.info(f"simulations: {simulations}")

    df = run(model, symbol, simulations)

    if start_date:
        start_date = start_date.date()

    if end_date:
        end_date = end_date.date()

    df = df.loc[start_date:end_date]

    logging.info(df)
    df.plot(subplots=True)
    plt.savefig("volatility_over_time.png")
    plt.show()


if __name__ == "__main__":
    # Run everything
    run_cli()
