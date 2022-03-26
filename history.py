import logging
import os
from typing import Dict, Iterable, Union

# Third party libraries
import pandas as pd
import yfinance as yf

# Local imports
import data_sources
import utils

# Initialization
logging.basicConfig(level=logging.INFO)


class FileSystemHistoryCache(object):
    def __init__(self, cache_dir="."):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, symbol: str) -> str:
        symbol_path = os.path.join(self.cache_dir, f"{symbol.lower()}.csv")
        return symbol_path

    def exists(self, symbol: str) -> bool:
        return os.path.exists(self._path(symbol))

    def save(self, symbol: str, df: pd.DataFrame):
        df = df.reset_index().set_index("date")
        df.to_csv(self._path(symbol), index=True)

    def load(self, symbol: str) -> pd.DataFrame:
        return pd.read_csv(
            self._path(symbol),
            index_col="date",
            parse_dates=["date"],
        )


class History(object):
    def __init__(self, data_source, cache_mgr):
        self.data_source = data_source
        self.cache_mgr = cache_mgr

    def download(
        self, symbols: Union[Iterable[str], str], overwrite_existing: bool = False
    ) -> Dict[str, pd.DataFrame]:
        # Handle the case where `symbol`is a single symbol
        symbols = utils.to_symbol_list(symbols)

        if not overwrite_existing:
            missing = []
            for symbol in symbols:
                if not self.cache_mgr.exists(symbol):
                    missing.append(symbol)
            symbols = missing

        if len(symbols) > 0:
            df = self.data_source.price_history(symbols)

            # Write the results to the cache
            for symbol in symbols:
                self.cache_mgr.save(symbol, df[symbol])
        else:
            df = {}

        return df

    def load(
        self, symbols: Union[Iterable[str], str], overwrite_existing=False
    ) -> None:
        symbols = utils.to_symbol_list(symbols)
        """
        Return a dataframe containing all historic values for the given set of symbosl.
        The dates are inner joined so there is one row for each date where all symbols
        have a value for that date.  The row index for the returned dataframe is the
        date.  The column is a muli-level index where the first position is the symbol
        and the second position is the value of interest (e.g., "close", "log_return", etc.)

        The expected use case is to get the log returns for a portfolio of stocks.  For example,
        the following returns a datafram of log returns for a portfolio on the dates where every
        item in the portfolio has a return:

        df.loc[:, (symbol_list, 'log_return')]

        This is intended for a portfolio, but you can specify just one stock if that's all that's required:

        df.loc[:, (symbol, 'log_return')]

        Arguments:
           symbols:  Union[Iterable[str], str] - a list of symbols of interst
           overwrite_existing: bool - whether to overwrite previously downloaded data (default False)

        Returns
           pd.DataFrame - The column is a muli-level index where the first position is the symbol
        and the second position is the value of interest (e.g., "close", "log_return", etc.)

        """

        self.download(symbols, overwrite_existing)

        dataframes = []
        for symbol in symbols:
            df = self.cache_mgr.load(symbol)
            df["symbol"] = symbol
            dataframes.append(df)

        print(dataframes)
        return pd.concat(dataframes, axis=1, join="inner", keys=symbols)


if __name__ == "__main__":  # pragma: no cover
    cache = FileSystemHistoryCache("training_data")
    data_source = data_sources.YFinanceSource()
    history = History(data_source, cache)
    symbols = ["QQQ", "SPY", "BND", "EDV"]
    df = history.load(symbols, overwrite_existing=False)

    selection = df.loc[:, (symbols, "log_return")]
    print(selection)
    print(selection.values.shape)

    selection = df.loc[:, (symbols[0], "log_return")]
    print(selection)
    print(selection.values.shape)
