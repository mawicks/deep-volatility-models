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

# import cufflinks as cf
# from IPython.display import display,HTML

# Local imports
import data_sources
import embedding_models
import stock_data
import time_series_datasets
import stats_utils


pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50000)
pd.set_option("display.min_rows", 1000)

# Configure external packages and run()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO, force=True)

# Torch configuration
torch.set_printoptions(
    precision=4, threshold=None, edgeitems=None, linewidth=None, profile="short"
)

ANNUAL_TRADING_DAYS = 252.0
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))


def do_one_symbol(
    symbol,
    model,
    refresh,
):
    logging.info(f"symbol: {symbol}")
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
    symbol_history = next(history_loader(symbol))[1]
    windowed_returns = time_series_datasets.RollingWindow(
        symbol_history.log_return,
        window_size,
        create_channel_dim=True,
    )
    logging.debug(f"{symbol} windowed_returns[0]: {windowed_returns[0].shape}")
    logging.debug(f"{symbol} windowed_returns[0]: {windowed_returns[0]}")

    with torch.no_grad():

        windows = torch.stack(tuple(windowed_returns), dim=0)
        logging.debug(f"{symbol} windows: {windows.shape}")

        dates = symbol_history.index[window_size - 1 :]
        logging.info(f"last date is {dates[-1]}")

        log_p, mu, sigma_inv = model.network(windows)[:3]
        p = torch.exp(log_p)

        logging.debug(f"p: {p}")
        logging.debug(f"mu: {mu}")
        logging.debug(f"sigma_inv: {sigma_inv}")

        mean, std_dev = stats_utils.combine_mixture_metrics(p, mu, sigma_inv)
        annual_return = ANNUAL_TRADING_DAYS * mean
        volatility = np.sqrt(ANNUAL_TRADING_DAYS) * std_dev

        logging.debug(f"daily mean: {mean}")
        logging.debug(f"daily std_dev: {std_dev}")

        logging.debug(f"annual return: {annual_return}")
        logging.debug(f"annual volatility: {volatility}")

        dominant_component_sigma = 1 / torch.max(sigma_inv, dim=1)[0]
        dominant_component_sigma = dominant_component_sigma.squeeze(2).squeeze(1)

        df = pd.DataFrame(
            {
                "volatility": volatility,
                "mean": mean,
                "p_non_base": 1.0 - torch.max(p, dim=1)[0],
                "sigma": std_dev,
                "base_sigma": dominant_component_sigma,
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
                "volatility",
                "log_return",
                "close",
                "mean",
                "p_non_base",
                "sigma",
                "base_sigma",
            ]
        ]
        logging.info(df[["log_return", "mean"]])

        df.plot(subplots=True)
        plt.show()


@click.command()
@click.option(
    "--model_dir",
    show_default=True,
    help="Use specified model directory.",
)
@click.option(
    "--symbol",
    multiple=True,
    show_default=True,
    help="Load model for this symbol.",
)
def run(
    model_dir,
    symbol,
):
    logging.info(f"model_dir: {model_dir}")
    logging.info(f"symbol: {symbol}")

    wrapped_model = torch.load(os.path.join(model_dir, "model.pt"))
    single_symbol_model_factory = embedding_models.SingleSymbolModelFactory(
        wrapped_model.encoding, wrapped_model
    )

    # symbols_to_process = list(set(symbol).difference(exclude_symbols))
    symbols_to_process = sorted(list(set(symbol)))
    logging.info(f"symbols_to_process: {symbols_to_process}")

    for s in symbols_to_process:
        do_one_symbol(s, single_symbol_model_factory(s.upper()), True)


if __name__ == "__main__":
    # Run everything
    run()
