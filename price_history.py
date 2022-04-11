import io
import logging
import os
from typing import Any, Callable, Dict, BinaryIO, Iterable, Union

# Third party libraries
import pandas as pd

# Local imports
import data_sources
import lutils

# Initialization
logging.basicConfig(level=logging.INFO)


def SymbolHistoryReader() -> Callable[[io.BufferedReader], pd.DataFrame]:
    """
    Constructs a reader() function that will read symbol history from an open
    file-like object.

    Returns:
        Callable[BinaryIO, pd.DataFrame] - Reader that whenn called on an open
        file returns a history dataframe.
    """

    def read_symbol_history(f: io.BufferedReader) -> pd.DataFrame:
        df = pd.read_csv(
            f,
            index_col="date",
            parse_dates=["date"],
        )

        # Be 100% certain it's in ascending order, even though it should have
        # been stored that way.
        df.sort_index(inplace=True)
        return df

    return read_symbol_history


def SymbolHistoryWriter(df: pd.DataFrame) -> Callable[[io.BufferedWriter], None]:
    def write_symbol_history(f: io.BufferedWriter) -> None:
        # Create an index on date and write to CSV in ascending order by date
        # with index=True
        indexed_df = df.copy()

        if indexed_df.index.name != "date":
            indexed_df.set_index("date", inplace=True)

        indexed_df.sort_index(inplace=True)
        indexed_df.to_csv(f, index=True)

    return write_symbol_history


class FileSystemStore(object):
    """
    This clsss implements an abstract interface for data storage.  It
    implements three methods:
        exists() to determine whether an object has beenstored
        write() to store an object
        load() to load an object.
    This particular implementation is specific to writing and loading
    dataframes.  It does some additional housekeeping and sanity checking on the dataframe.

    Abstracting this interface allows the file system to be replaced or
    mocked out more esily for testing.
    """

    def __init__(self, cache_dir="."):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, symbol: str) -> str:
        """Construct a filesystem path to store and retrieve the data for the
        associated givwn key
        Arguments:
            symbol: str
        Returns:
            str - The filesystem path to be used for the key
        """
        # TODO: Return a path object rather than a string to increase porability.
        symbol_path = os.path.join(self.cache_dir, f"{symbol.lower()}.csv")
        return symbol_path

    def exists(self, symbol: str) -> bool:
        """Return whether the symbol exists in the data store
        Arguments:
            symbol: str - the symbol or key to retrieve
        Returns:
            True if the key exists in the data store.
        """
        return os.path.exists(self._path(symbol))

    def write(self, symbol: str, writer: Callable[[io.BufferedWriter], None]):
        """
        Write a key and data (must be a dataframe) to the data store
        Arguments:
            symbol: str - The symbol or "key" for the data.
            df: pd.DataFrame - The dataframe to store for that symbol.
        Returns:
            None
        """
        with open(self._path(symbol), "wb") as f:
            writer(f)

    def read(self, symbol: str, reader: Callable[[io.BufferedReader], Any]) -> Any:
        """
        Read a dataframe given its symbol.
        Arguments:
            symbol: str
        Returns:
            pd.DataFrame - The associated dataframe.
        """
        with open(self._path(symbol), "rb") as f:
            result = reader(f)
        return result


def CachingDownloader(
    data_source: Callable[[Union[str, Iterable[str]]], Dict[str, pd.DataFrame]],
    data_store: FileSystemStore,
    writer_factory,
):
    """
    Construct and return a download function that will download and write the
    results to the data store as necessary.
    Arguments:
        data_source: Callable[[Union[str, Iterable[str]]], Dict[str,
        pd.DataFrame]] -  A datasource function which given a list of symbols
        returns a dictionary keyed by the symbol with values that are dataframe with history data for
        that symbol.

        data_store: FilesystemStore (or similar) - An implementation of a data_store class (see
        FileSystemStore above)

    """

    def download(
        symbols: Union[Iterable[str], str], overwrite_existing: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Arguments:
            symbols: Union[Iterable[str], str] - A symbol of list of symbols to populate
        in the cache.
            overwrite_existing: bool - Forces all symbols to be downloaded whether or not
            they already exist in the cache.
        """
        # Handle the case where `symbol`is a single symbol
        symbols = lutils.to_symbol_list(symbols)

        if not overwrite_existing:
            # Determine what's missing
            missing = []
            for symbol in symbols:
                if not data_store.exists(symbol):
                    missing.append(symbol)

            # Replace full list with missing list
            symbols = missing

        if len(symbols) > 0:
            ds = data_source(symbols)

            # Write the results to the cache
            for symbol in symbols:
                writer = writer_factory(ds[symbol])
                data_store.write(symbol, writer)
        else:
            ds = {}

        return ds

    return download


def CachingLoader(data_source, data_store, reader_factory, writer_factory):
    caching_download = CachingDownloader(data_source, data_store, writer_factory)

    def load(symbols: Union[Iterable[str], str], overwrite_existing=False) -> Any:
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

        <"""
        symbols = lutils.to_symbol_list(symbols)
        caching_download(symbols, overwrite_existing)

        dataframes = []
        reader = reader_factory()
        for symbol in symbols:
            df = data_store.read(symbol, reader)
            df["symbol"] = symbol
            dataframes.append(df)

        print(dataframes)
        return pd.concat(dataframes, axis=1, join="inner", keys=symbols)

    return load


def CachingSymbolHistoryLoader(data_source, data_store):
    return CachingLoader(
        data_source, data_store, SymbolHistoryReader, SymbolHistoryWriter
    )


if __name__ == "__main__":  # pragma: no cover
    data_store = FileSystemStore("training_data")
    data_source = data_sources.YFinanceSource()
    load = CachingSymbolHistoryLoader(data_source, data_store)
    symbols = ["QQQ", "SPY", "BND", "EDV"]
    df = load(symbols, overwrite_existing=False)

    selection = df.loc[:, (symbols, "log_return")]
    print(selection)
    print(selection.values.shape)

    selection = df.loc[:, (symbols[0], "log_return")]
    print(selection)
    print(selection.values.shape)
