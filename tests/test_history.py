import pytest

# Standard Python modules
import os
from unittest.mock import create_autospec, patch

# Third party modules
import pandas as pd
from uritemplate import partial

# Local imports
import data_sources
import history


SAMPLE_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "open": [1.0, 2.0, 3.0],
        "close": [0.5, 2.5, 3.1],
    }
).set_index("date")

SAMPLE_PATH = "any_path"


@pytest.fixture
def tmp_path_store(tmp_path):
    """
    Create an instance of CSVFileSystemCache for testing
    """
    return history.FileSystemStore(os.fspath(tmp_path))


@pytest.fixture
def data_source():
    """
    Create an instance of a data source for testing
    """
    mock_data_source = lambda symbols: {s.upper(): SAMPLE_DF for s in symbols}
    return mock_data_source


def test_cache_normal_use_sequence(tmp_path_store):
    symbol = "xyz"
    assert not tmp_path_store.exists(symbol)

    tmp_path_store.write(symbol, SAMPLE_DF)

    assert tmp_path_store.exists(symbol)

    reloaded = tmp_path_store.read(symbol)

    print(reloaded.head(3))
    print(SAMPLE_DF.head(3))
    assert (reloaded == SAMPLE_DF).all().all()


def test_check_cache_exists_path(tmp_path_store):
    """
    Check that the os.path.exists() gets called with the correct path
    and check that exists is not case sensitive.
    """
    with patch("history.os.path.exists") as os_path_exists:
        tmp_path_store.exists("symbol1")
        os_path_exists.assert_called_with(
            os.path.join(tmp_path_store.cache_dir, "symbol1.csv")
        )

        tmp_path_store.exists("SyMbOL2")
        os_path_exists.assert_called_with(
            os.path.join(tmp_path_store.cache_dir, "symbol2.csv")
        )


def test_history(data_source, tmp_path_store):
    partial_symbol_set = set(["ABC", "DEF"])
    missing_symbol_set = set(["GHI", "JKL"])
    full_symbol_set = partial_symbol_set.union(missing_symbol_set)

    caching_download = history.CachingDownloader(data_source, tmp_path_store)

    response = caching_download(partial_symbol_set)
    assert len(response) == len(partial_symbol_set)
    for symbol in partial_symbol_set:
        assert tmp_path_store.exists(symbol)

    for symbol in missing_symbol_set:
        assert not tmp_path_store.exists(symbol)

    response = caching_download(full_symbol_set)
    # Check that only the missing symbols were downloaded
    # This is true if all missing symbols are in the response
    # and if the length of the response is equal to the number
    # of missing symbols
    assert len(response) == len(missing_symbol_set)
    for symbol in missing_symbol_set:
        assert symbol in response

    for symbol in full_symbol_set:
        assert tmp_path_store.exists(symbol)

    # Try downloading again, which should be a no-op
    response = caching_download(full_symbol_set)
    assert len(response) == 0

    # Try loading one of the downloaded files
    load = history.CachingLoader(data_source, tmp_path_store)
    load("pqr")
