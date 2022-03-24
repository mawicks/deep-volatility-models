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
        df = df.reset_index().set_index("Date")
        df.to_csv(self._path(symbol), index=True)

    def load(self, symbol: str) -> pd.DataFrame:
        return pd.read_csv(
            self._path(symbol),
            index_col="Date",
            parse_dates=["Date"],
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

        self.download(symbols, overwrite_existing)

        dataframes = []
        for symbol in symbols:
            df = self.cache_mgr.load(symbol)
            df["Symbol"] = symbol
            dataframes.append(df)

        return pd.concat(dataframes, axis=0)


if __name__ == "__main__":  # pragma: no cover
    cache = FileSystemHistoryCache("training_data")
    data_source = data_sources.YFinanceSource()
    history = History(data_source, cache)
    df = history.load(["qqq", "spy", "bnd"], overwrite_existing=False)
    print(df.head())
