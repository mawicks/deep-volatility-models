import logging

from typing import Dict, Iterable, Union

# Third party modules
import numpy as np
import pandas as pd
import yfinance as yf

# Local modules
import utils

logging.basicConfig(level=logging.INFO)


def YFinanceSource():
    """
    xxx
    >>> import data_sources
    >>> symbols = ["SPY", "QQQ"]
    >>> ds = data_sources.YFinanceSource()
    >>> response = ds(symbols)
    >>> response["SPY"][:4][['open', 'close']]  # doctest: +NORMALIZE_WHITESPACE
                    open     close
    date
    1993-02-01  43.96875  44.25000
    1993-02-02  44.21875  44.34375
    1993-02-03  44.40625  44.81250
    1993-02-04  44.96875  45.00000
        >>>
    """

    def _add_columns(df):
        new_df = df.dropna().reset_index()
        rename_dict = {c: utils.rename_column(c) for c in new_df.columns}
        log_return = np.log(new_df["Adj Close"] / new_df["Adj Close"].shift(1))
        new_df = new_df.assign(log_return=log_return)
        new_df.rename(columns=rename_dict, inplace=True)
        new_df.set_index("date", inplace=True)
        return new_df

    def price_history(symbol_set: Union[Iterable[str], str]) -> Dict[str, pd.DataFrame]:

        # Convert symbol_set to a list
        symbols = utils.to_symbol_list(symbol_set)

        # Do the download
        df = yf.download(
            symbols, period="max", group_by="ticker", actions=True, progress=False
        )
        response = {}

        for symbol in symbols:
            # The `group_by` option for yf.download() behaves differently when there's only one symbol.
            # Always return a dictionary of dataframes, even for one symbol.
            if len(symbols) > 1:
                symbol_df = df[symbol]
            else:
                symbol_df = df

            response[symbol] = (
                _add_columns(symbol_df).dropna().applymap(lambda x: round(x, 6))
            )

        return response

    return price_history


if __name__ == "__main__":  # pragma: no cover
    symbols = ["spy", "qqq"]
    ds = YFinanceSourceFactory()
    response = ds(symbols)

    for k, v in response.items():
        print(f"{k}:\n{v.head(3)}")
