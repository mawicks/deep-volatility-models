from typing import List, Iterable, Union


def to_symbol_list(symbols: Union[Iterable[str], str]) -> List[str]:
    """
    This function converts its `symbols` argument to a list of strings.
    It's used as a convenence function so that a caller can provide a
    single symbol to a function rather than in Iterable.
    We also normalize symbols lists by converting the symbols to upper case.

    Arguments:
       symbols: Union[Iterable[str], str]: The collection of symbols to convert or a single symbol
    Returns:
       List[str]: An instantiated list of symbols, hich may the same list passed in
    """

    if isinstance(symbols, str):
        symbols = (symbols,)

    return list(map(str.upper, symbols))


def rename_column(c):
    """
    Standardize column naming.  No spaces and no caps.
    """
    return c.lower().replace(" ", "_")