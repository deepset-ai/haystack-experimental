import pytest
from typing import Any, List, Dict, Optional, Union, TypeVar, Generic
from dataclasses import dataclass

from haystack_experimental.dataclasses.state_utils import merge_values, is_list_type, is_dict_type, merge_lists, merge_dicts, _is_valid_type

import inspect

def test_is_list_type():
    """Test the list type detection function."""
    assert is_list_type(list) is True
    assert is_list_type(List[int]) is True
    assert is_list_type(List[str]) is True
    assert is_list_type(dict) is False
    assert is_list_type(int) is False
    assert is_list_type(Union[List[int], None]) is False


def test_is_dict_type():
    """Test the dict type detection function."""
    assert is_dict_type(dict) is True
    assert is_dict_type(Dict[str, int]) is True
    assert is_dict_type(Dict[str, Any]) is True
    assert is_dict_type(list) is False
    assert is_dict_type(int) is False
    assert is_dict_type(Union[Dict[str, int], None]) is False


class TestMergeLists:
    """Test cases for list merging functionality."""

    def test_merge_two_lists(self):
        """Test merging two lists."""
        current = [1, 2, 3]
        new = [4, 5, 6]
        result = merge_lists(current, new)
        assert result == [1, 2, 3, 4, 5, 6]
        # Ensure original lists weren't modified
        assert current == [1, 2, 3]
        assert new == [4, 5, 6]

    def test_append_to_list(self):
        """Test appending a non-list value to a list."""
        current = [1, 2, 3]
        new = 4
        result = merge_lists(current, new)
        assert result == [1, 2, 3, 4]
        assert current == [1, 2, 3]  # Ensure original wasn't modified

    def test_create_new_list(self):
        """Test creating a new list from two non-list values."""
        current = 1
        new = 2
        result = merge_lists(current, new)
        assert result == [1, 2]

    def test_replace_with_list(self):
        """Test replacing a non-list value with a list."""
        current = 1
        new = [2, 3]
        result = merge_lists(current, new)
        assert result == [2, 3]


class TestMergeDicts:
    """Test cases for dictionary merging functionality."""

    def test_merge_two_dicts(self):
        """Test merging two dictionaries."""
        current = {'a': 1, 'b': 2}
        new = {'c': 3, 'd': 4}
        result = merge_dicts(current, new)
        assert result == {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        # Ensure original dicts weren't modified
        assert current == {'a': 1, 'b': 2}
        assert new == {'c': 3, 'd': 4}

    def test_dict_key_override(self):
        """Test that new dict values override existing ones."""
        current = {'a': 1, 'b': 2}
        new = {'b': 3, 'c': 4}
        result = merge_dicts(current, new)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_replace_non_dict(self):
        """Test replacing when either value is not a dict."""
        assert merge_dicts(42, {'a': 1}) == {'a': 1}
        assert merge_dicts({'a': 1}, 42) == 42
        assert merge_dicts(42, 43) == 43


class TestMergeValues:
    """Test cases for the main merge_values function."""

    def test_none_handling(self):
        """Test handling of None values."""
        assert merge_values(None, 42, int) == 42
        assert merge_values(None, [1, 2], List[int]) == [1, 2]
        assert merge_values(None, {'a': 1}, Dict[str, int]) == {'a': 1}

    def test_list_merging(self):
        """Test list merging with different types."""
        # List + List
        assert merge_values([1, 2], [3, 4], List[int]) == [1, 2, 3, 4]
        # List + Value
        assert merge_values([1, 2], 3, List[int]) == [1, 2, 3]
        # Value + List
        assert merge_values(1, [2, 3], List[int]) == [2, 3]
        # Value + Value
        assert merge_values(1, 2, List[int]) == [1, 2]

    def test_dict_merging(self):
        """Test dictionary merging with different types."""
        # Dict + Dict
        assert merge_values({'a': 1}, {'b': 2}, Dict[str, int]) == {'a': 1, 'b': 2}
        # Dict + Non-Dict
        assert merge_values({'a': 1}, 42, Dict[str, int]) == 42
        # Non-Dict + Dict
        assert merge_values(42, {'a': 1}, Dict[str, int]) == {'a': 1}

    def test_primitive_replacement(self):
        """Test primitive value replacement."""
        assert merge_values(1, 2, int) == 2
        assert merge_values("old", "new", str) == "new"
        assert merge_values(1.0, 2.0, float) == 2.0

    @pytest.mark.parametrize(
        "current,new,type_hint,expected",
        [
            ([1, 2], [3, 4], List[int], [1, 2, 3, 4]),
            ([1, 2], 3, List[int], [1, 2, 3]),
            ({'a': 1}, {'b': 2}, Dict[str, int], {'a': 1, 'b': 2}),
            (1, 2, int, 2),
            (None, 42, int, 42),
        ]
    )
    def test_parametrized_cases(self, current, new, type_hint, expected):
        """Parametrized test cases for different scenarios."""
        assert merge_values(current, new, type_hint) == expected

    def test_nested_structures(self):
        """Test merging nested structures."""
        current = {'list': [1, 2], 'dict': {'a': 1}}
        new = {'list': [3, 4], 'dict': {'b': 2}}
        result = merge_values(current, new, Dict[str, Any])
        assert result == {'list': [3, 4], 'dict': {'b': 2}}


class TestIsValidType:
    """Test cases for type validation function."""

    def test_builtin_types(self):
        """Test with built-in Python types."""
        assert _is_valid_type(str) is True
        assert _is_valid_type(int) is True
        assert _is_valid_type(dict) is True
        assert _is_valid_type(list) is True
        assert _is_valid_type(tuple) is True
        assert _is_valid_type(set) is True
        assert _is_valid_type(bool) is True
        assert _is_valid_type(float) is True

    def test_generic_types(self):
        """Test with generic type hints."""
        assert _is_valid_type(List[str]) is True
        assert _is_valid_type(Dict[str, int]) is True
        assert _is_valid_type(List[Dict[str, int]]) is True
        assert _is_valid_type(Dict[str, List[int]]) is True

    def test_custom_classes(self):
        """Test with custom classes and generic custom classes."""

        @dataclass
        class CustomClass:
            value: int

        T = TypeVar('T')

        class GenericCustomClass(Generic[T]):
            pass

        # Test regular and generic custom classes
        assert _is_valid_type(CustomClass) is True
        assert _is_valid_type(GenericCustomClass) is True
        assert _is_valid_type(GenericCustomClass[int]) is True

        # Test generic types with custom classes
        assert _is_valid_type(List[CustomClass]) is True
        assert _is_valid_type(Dict[str, CustomClass]) is True
        assert _is_valid_type(Dict[str, GenericCustomClass[int]]) is True

    def test_invalid_types(self):
        """Test with invalid type inputs."""
        # Test regular values
        assert _is_valid_type(42) is False
        assert _is_valid_type("string") is False
        assert _is_valid_type([1, 2, 3]) is False
        assert _is_valid_type({"a": 1}) is False
        assert _is_valid_type(True) is False

        # Test class instances
        @dataclass
        class SampleClass:
            value: int

        instance = SampleClass(42)
        assert _is_valid_type(instance) is False

        # Test callable objects
        assert _is_valid_type(len) is False
        assert _is_valid_type(lambda x: x) is False
        assert _is_valid_type(print) is False

    def test_union_and_optional_types(self):
        """Test with Union and Optional types."""
        # Test basic Union types
        assert _is_valid_type(Union[str, int]) is True
        assert _is_valid_type(Union[str, None]) is True
        assert _is_valid_type(Union[List[int], Dict[str, str]]) is True

        # Test Optional types (which are Union[T, None])
        assert _is_valid_type(Optional[str]) is True
        assert _is_valid_type(Optional[List[int]]) is True
        assert _is_valid_type(Optional[Dict[str, list]]) is True

        # Test that Union itself is not a valid type (only instantiated Unions are)
        assert _is_valid_type(Union) is False

    def test_nested_generic_types(self):
        """Test with deeply nested generic types."""
        assert _is_valid_type(List[List[Dict[str, List[int]]]]) is True
        assert _is_valid_type(Dict[str, List[Dict[str, set]]]) is True
        assert _is_valid_type(Dict[str, Optional[List[int]]]) is True
        assert _is_valid_type(List[Union[str, Dict[str, List[int]]]]) is True

    def test_edge_cases(self):
        """Test edge cases and potential corner cases."""
        # Test None and NoneType
        assert _is_valid_type(None) is False
        assert _is_valid_type(type(None)) is True

        # Test functions and methods
        def sample_func(): pass

        assert _is_valid_type(sample_func) is False
        assert _is_valid_type(type(sample_func)) is True

        # Test modules
        assert _is_valid_type(inspect) is False

        # Test type itself
        assert _is_valid_type(type) is True

    @pytest.mark.parametrize("test_input,expected", [
        (str, True),
        (int, True),
        (List[int], True),
        (Dict[str, int], True),
        (Union[str, int], True),
        (Optional[str], True),
        (42, False),
        ("string", False),
        ([1, 2, 3], False),
        (lambda x: x, False),
    ])
    def test_parametrized_cases(self, test_input, expected):
        """Parametrized test cases for different scenarios."""
        assert _is_valid_type(test_input) is expected
