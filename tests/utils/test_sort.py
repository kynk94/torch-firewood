from firewood.utils import sort


def test_str_to_float():
    assert sort.str_to_float("1.2") == 1.2
    assert sort.str_to_float("1.2.3") == "1.2.3"


def test_numeric_string_sort():
    strings = [
        "as3ab",
        "as1.24gf",
        "b2cd",
        "a0.9gg",
    ]
    sorted_strings = [
        "a0.9gg",
        "as1.24gf",
        "as3ab",
        "b2cd",
    ]
    assert sorted(strings, key=sort.numeric_string_sort) == sorted_strings

    tupled_strings = tuple((string, i) for i, string in enumerate(strings))
    sorted_tupled_strings = [
        ("a0.9gg", 3),
        ("as1.24gf", 1),
        ("as3ab", 0),
        ("b2cd", 2),
    ]
    assert (
        sorted(tupled_strings, key=sort.numeric_string_sort)
        == sorted_tupled_strings
    )


def test_digit_first_sort():
    strings = [
        "as3ab",
        "as1.24gf",
        "b2cd",
        "a0.9gg",
    ]
    sorted_strings = [
        "a0.9gg",
        "as1.24gf",
        "b2cd",
        "as3ab",
    ]
    assert sorted(strings, key=sort.digit_first_sort) == sorted_strings

    tupled_strings = tuple((string, i) for i, string in enumerate(strings))
    sorted_tupled_strings = [
        ("a0.9gg", 3),
        ("as1.24gf", 1),
        ("b2cd", 2),
        ("as3ab", 0),
    ]
    assert (
        sorted(tupled_strings, key=sort.digit_first_sort)
        == sorted_tupled_strings
    )
