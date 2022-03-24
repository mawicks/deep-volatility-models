import pytest

# Local modules
import utils


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        ("foo", ["FOO"]),
        ([], []),
        (["foo"], ["FOO"]),
        (("a", "b"), ["A", "B"]),
        (iter(("x", "y", "z")), ["X", "Y", "Z"]),
    ],
)
def test_to_symbol_list(test_input, expected_output):
    print(f"test input: {test_input}")
    print(f"expected output: {expected_output}")
    assert utils.to_symbol_list(test_input) == expected_output
