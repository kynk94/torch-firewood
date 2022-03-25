import re
from typing import List, Tuple, Union, overload


def str_to_float(text: str) -> Union[str, float]:
    try:
        return float(text)
    except ValueError:
        return text


@overload
def numeric_string_sort(obj: str) -> List[Union[int, float, str]]:
    ...


@overload
def numeric_string_sort(
    obj: Union[Tuple[Union[int, float, str], ...], List[Union[int, float, str]]]
) -> List[List[Union[int, float, str]]]:
    ...


def numeric_string_sort(
    obj: Union[
        str,
        Tuple[Union[int, float, str], ...],
        List[Union[int, float, str]],
    ],
) -> Union[List[Union[int, float, str]], List[List[Union[int, float, str]]]]:
    """
    Implements a numeric string sort.

    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """
    if isinstance(obj, str):
        return [
            str_to_float(s)
            for s in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", obj)
        ]
    results: List[List[Union[int, float, str]]] = []
    for item in obj:
        if isinstance(item, str):
            results.append(numeric_string_sort(item))
        else:
            results.append([item])
    return results


@overload
def digit_first_sort(obj: str) -> List[Union[int, float, str]]:
    ...


@overload
def digit_first_sort(
    obj: Union[Tuple[Union[int, float, str], ...], List[Union[int, float, str]]]
) -> List[List[Union[int, float, str]]]:
    ...


def digit_first_sort(
    obj: Union[
        str,
        Tuple[Union[int, float, str], ...],
        List[Union[int, float, str]],
    ],
) -> Union[List[Union[int, float, str]], List[List[Union[int, float, str]]]]:
    if isinstance(obj, str):
        digits: List[Union[int, float, str]] = []
        strings: List[str] = []
        for s in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", obj):
            result = str_to_float(s)
            if isinstance(result, str):
                strings.append(result)
            else:
                digits.append(result)
        digits.extend(strings)
        return digits
    results = []
    for item in obj:
        if isinstance(item, str):
            results.append(digit_first_sort(item))
        else:
            results.append([item])
    return results
