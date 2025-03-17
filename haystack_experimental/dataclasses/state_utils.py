# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, List, TypeVar, Union, get_origin

T = TypeVar("T")


def _is_valid_type(obj: Any) -> bool:
    """
    Check if an object is a valid type annotation.

    Valid types include:
    - Normal classes (str, dict, CustomClass)
    - Generic types (List[str], Dict[str, int])
    - Union types (Union[str, int], Optional[str])

    :param obj: The object to check
    :return: True if the object is a valid type annotation, False otherwise

    Example usage:
        >>> _is_valid_type(str)
        True
        >>> _is_valid_type(List[int])
        True
        >>> _is_valid_type(Union[str, int])
        True
        >>> _is_valid_type(42)
        False
    """
    # Handle Union types (including Optional)
    if hasattr(obj, "__origin__") and obj.__origin__ is Union:
        return True

    # Handle normal classes and generic types
    return inspect.isclass(obj) or type(obj).__name__ in {"_GenericAlias", "GenericAlias"}


def _is_list_type(type_hint: Any) -> bool:
    """
    Check if a type hint represents a list type.

    :param type_hint: The type hint to check
    :return: True if the type hint represents a list, False otherwise
    """
    return type_hint is list or (hasattr(type_hint, "__origin__") and get_origin(type_hint) is list)


def merge_lists(current: Union[List[T], T], new: Union[List[T], T]) -> List[T]:
    """
    Merge two values according to list merging rules.

    :param current: The current value, which may or may not be a list
    :param new: The new value to merge, which may or may not be a list
    :return: A new list containing the merged values
    """
    # If the current value is None, return the new value as a list
    if current is None:
        return new if isinstance(new, list) else [new]

    # If the current value is not none, then merge
    current_list = current if isinstance(current, list) else [current]
    new_list = new if isinstance(new, list) else [new]
    return current_list + new_list


def replace_values(current: Any, new: Any) -> Any:
    """
    Replace the current value with the new value.

    :param current: The existing value
    :param new: The new value to replace
    :return: The new value
    """
    return new
