import os
from typing import Any, Callable, Dict, Optional, Union

from firewood.utils.common import get_last_file


def find_checkpoint(
    path: str,
    regex: str = "*.pt*",
    key: Optional[Union[Callable, str]] = None,
) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    if os.path.isdir(path):
        checkpoint = get_last_file(directory=path, regex=regex, key=key)
    else:
        checkpoint = path
    print(f"Found checkpoint: {checkpoint}")
    return checkpoint


def extract_state_dict(
    dictionary: Dict[str, Any], key: str, pop_key: bool = True
) -> Dict[str, Any]:
    extracted_state_dict = dict()
    if not key.endswith("."):
        key += "."
    for k, v in dictionary.items():
        if not k.startswith(key):
            continue
        if pop_key:
            k = k[len(key) :]
        extracted_state_dict[k] = v
    return extracted_state_dict
