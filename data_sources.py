import logging

from typing import Dict, Iterable, Union

# Third party modules
import numpy as np
import pandas as pd
import yfinance as yf

# Local modules
import utils

logging.basicConfig(level=logging.INFO)


class YFinanceSource(object):
    @staticmethod
    def _add_columns(df):
        rename_dict = {c: utils.rename_column(c) for c in df.columns}
        log_return = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
        new_df = df.assign(log_return=log_return)
        print(rename_dict)
        new_df.rename(columns=rename_dict, inplace=True)
        return new_df

    def price_history(
        self, symbol_set: Union[Iterable[str], str]
    ) -> Dict[str, pd.DataFrame]:

        # Convert symbol_set to a list
        symbols = utils.to_symbol_list(symbol_set)

        # Do the download
        df = yf.download(symbols, period="max", group_by="ticker", actions=True)
        response = {}

        for symbol in symbols:
            # The `group_by` option for yf.download() behaves differently when there's only one symbol.
            # Always return a dictionary of dataframes, even for one symbol.
            if len(symbols) > 1:
                symbol_df = df[symbol]
            else:
                symbol_df = df

            response[symbol] = (
                self._add_columns(symbol_df).dropna().applymap(lambda x: round(x, 6))
            )

        return response


if __name__ == "__main__":  # pragma: no cover
    symbols = ["spy", "qqq"]
    ds = YFinanceSource()
    response = ds.price_history(symbols)

    for k, v in response.items():
        print(f"{k}:\n{v.head(3)}")
