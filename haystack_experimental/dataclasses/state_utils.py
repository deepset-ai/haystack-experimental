# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict, List, TypeVar, Union, get_origin


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
    return (inspect.isclass(obj) or
            type(obj).__name__ in {"_GenericAlias", "GenericAlias"})



T = TypeVar("T")


def is_list_type(type_hint: Any) -> bool:
    """
    Check if a type hint represents a list type.

    :param type_hint: The type hint to check
    :return: True if the type hint represents a list, False otherwise
    """
    return type_hint is list or (hasattr(type_hint, "__origin__") and get_origin(type_hint) is list)


def is_dict_type(type_hint: Any) -> bool:
    """
    Check if a type hint represents a dict type.

    :param type_hint: The type hint to check
    :return: True if the type hint represents a dict, False otherwise
    """
    return type_hint is dict or (hasattr(type_hint, "__origin__") and get_origin(type_hint) is dict)


def merge_lists(current: Union[List[T], Any], new: Union[List[T], T]) -> List[T]:
    """
    Merge two values according to list merging rules.

    :param current: The current value, which may or may not be a list
    :param new: The new value to merge, which may or may not be a list
    :return: A new list containing the merged values
    """
    if isinstance(current, list):
        if isinstance(new, list):
            return [*current, *new]  # Extend
        return [*current, new]  # Append
    if isinstance(new, list):
        return new  # Replace
    return [current, new]  # Create new list


def merge_dicts(current: Union[Dict[str, T], T], new: Union[Dict[str, T], T]) -> Union[Dict[str, T], T]:
    """
    Merge two values according to dict merging rules.

    :param current: The current value, which may or may not be a dict
    :param new: The new value to merge, which may or may not be a dict
    :return: A new dict containing the merged values if both inputs are dicts, otherwise the new value
    """
    if isinstance(current, dict) and isinstance(new, dict):
        return {**current, **new}  # Update
    return new  # Replace


def merge_values(current_value: T, new_value: T, declared_type: Any) -> T:
    """
    Merge values based on their types and declared type hint.

    Rules:
    - Lists: extend if new value is also a list, otherwise append
    - Dicts: shallow update if both are dicts, otherwise replace
    - Others: replace entirely

    :param current_value: The existing value
    :param new_value: The new value to merge
    :param declared_type: The type hint for the value
    :return: The merged value according to the merging rules
    """
    if current_value is None:
        return new_value

    if is_list_type(declared_type):
        return merge_lists(current_value, new_value)

    if is_dict_type(declared_type):
        return merge_dicts(current_value, new_value)

    return new_value  # Default: replace
