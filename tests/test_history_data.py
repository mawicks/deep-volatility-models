import pytest

# Standard Python modules
import collections
import os
from unittest.mock import create_autospec, patch

# Third party modules
import pandas as pd
from uritemplate import partial

# Local imports
import data_sources
import history_data


SAMPLE_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "open": [1.0, 2.0, 3.0],
        "close": [0.5, 2.5, 3.1],
    }
).set_index("date")

SAMPLE_PATH = "any_path"


@pytest.fixture
def dataframe():
    """
    Create a mock dataframe fixture with the same API as a real dataframe
    """
    mock_df = create_autospec(SAMPLE_DF)
    mock_df.to_csv = create_autospec(SAMPLE_DF.to_csv)
    mock_df.reset_index = create_autospec(SAMPLE_DF.reset_index)
    return mock_df


@pytest.fixture
def cache_tmp_path(tmp_path):
    """
    Create an instance of CSVFileSystemCache for testing
    """
    return history_data.FileSystemHistoryCache(os.fspath(tmp_path))


@pytest.fixture
def data_source():
    """
    Create an instance of a data source for testing
    """
    mock_data_source = create_autospec(data_sources.YFinanceSource)
    mock_data_source.price_history = lambda symbols: {
        s.upper(): SAMPLE_DF for s in symbols
    }

    return mock_data_source


def test_cache_normal_use_sequence(cache_tmp_path):
    symbol = "xyz"
    assert not cache_tmp_path.exists(symbol)

    cache_tmp_path.save(symbol, SAMPLE_DF)

    assert cache_tmp_path.exists(symbol)

    reloaded = cache_tmp_path.load(symbol)

    print(reloaded.head(3))
    print(SAMPLE_DF.head(3))
    assert (reloaded == SAMPLE_DF).all().all()


def test_check_cache_exists_path(cache_tmp_path):
    """
    Check that the os.path.exists() gets called with the correct path
    and check that exists is not case sensitive.
    """
    with patch("history_data.os.path.exists") as os_path_exists:
        cache_tmp_path.exists("symbol1")
        os_path_exists.assert_called_with(
            os.path.join(cache_tmp_path.cache_dir, "symbol1.csv")
        )

        cache_tmp_path.exists("SyMbOL2")
        os_path_exists.assert_called_with(
            os.path.join(cache_tmp_path.cache_dir, "symbol2.csv")
        )


def test_history(data_source, cache_tmp_path):
    partial_symbol_set = set(["ABC", "DEF"])
    missing_symbol_set = set(["GHI", "JKL"])
    full_symbol_set = partial_symbol_set.union(missing_symbol_set)

    history = history_data.History(data_source, cache_tmp_path)

    response = history.download(partial_symbol_set)
    assert len(response) == len(partial_symbol_set)
    for symbol in partial_symbol_set:
        assert cache_tmp_path.exists(symbol)

    for symbol in missing_symbol_set:
        assert not cache_tmp_path.exists(symbol)

    response = history.download(full_symbol_set)
    # Check that only the missing symbols were downloaded
    # This is true if all missing symbols are in the response
    # and if the length of the response is equal to the number
    # of missing symbols
    assert len(response) == len(missing_symbol_set)
    for symbol in missing_symbol_set:
        assert symbol in response

    for symbol in full_symbol_set:
        assert cache_tmp_path.exists(symbol)

    # Try downloading again, which should be a no-op
    response = history.download(full_symbol_set)
    assert len(response) == 0

    # Try loading one of the downloaded files
    history.load("pqr")
