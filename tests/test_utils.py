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


@pytest.mark.parametrize(
    "test_input, expected_output",
    [
        ("foo", "foo"),
        ("Foo", "foo"),
        ("a b c", "a_b_c"),
        ("A b C", "a_b_c"),
    ],
)
def test_rename_column(test_input, expected_output):
    print(f"test input: {test_input}")
    print(f"expected output: {expected_output}")
    assert utils.rename_column(test_input) == expected_output
