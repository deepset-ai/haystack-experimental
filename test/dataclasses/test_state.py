import pytest
from typing import List, Dict

from haystack_experimental.dataclasses.state import State, _validate_schema

# Test fixtures
@pytest.fixture
def basic_schema():
    return {
        "numbers": {"type": list},
        "metadata": {"type": dict},
        "name": {"type": str},
    }


@pytest.fixture
def complex_schema():
    return {
        "numbers": {
            "type": list,
            "handler": lambda current, new: sorted(set(current + new)) if current else sorted(set(new))
        },
        "metadata": {"type": dict},
        "name": {"type": str},
    }


# Test _validate_schema
def test_validate_schema_valid(basic_schema):
    # Should not raise any exceptions
    _validate_schema(basic_schema)


def test_validate_schema_invalid_type():
    invalid_schema = {
        "test": {"type": "not_a_type"}
    }
    with pytest.raises(ValueError, match="must be a Python type"):
        _validate_schema(invalid_schema)


def test_validate_schema_missing_type():
    invalid_schema = {
        "test": {"handler": lambda x, y: x + y}
    }
    with pytest.raises(ValueError, match="missing a 'type' entry"):
        _validate_schema(invalid_schema)


def test_validate_schema_invalid_handler():
    invalid_schema = {
        "test": {"type": list, "handler": "not_callable"}
    }
    with pytest.raises(ValueError, match="must be callable or None"):
        _validate_schema(invalid_schema)


# Test State class
def test_state_initialization(basic_schema):
    # Test empty initialization
    state = State(basic_schema)
    assert state.data == {}

    # Test initialization with data
    initial_data = {"numbers": [1, 2, 3], "name": "test"}
    state = State(basic_schema, initial_data)
    assert state.data["numbers"] == [1, 2, 3]
    assert state.data["name"] == "test"


def test_state_get(basic_schema):
    state = State(basic_schema, {"name": "test"})
    assert state.get("name") == "test"
    assert state.get("non_existent") is None
    assert state.get("non_existent", "default") == "default"


def test_state_set_basic(basic_schema):
    state = State(basic_schema)

    # Test setting new values
    state.set("numbers", [1, 2])
    assert state.get("numbers") == [1, 2]

    # Test updating existing values
    state.set("numbers", [3, 4])
    assert state.get("numbers") == [1, 2, 3, 4]


def test_state_set_with_handler(complex_schema):
    state = State(complex_schema)

    # Test custom handler for numbers
    state.set("numbers", [3, 2, 1])
    assert state.get("numbers") == [1, 2, 3]

    state.set("numbers", [6, 5, 4])
    assert state.get("numbers") == [1, 2, 3, 4, 5, 6]


def test_state_set_with_force(basic_schema):
    state = State(basic_schema)

    # Set initial value
    state.set("numbers", [1, 2])

    # Force update should replace instead of merge
    state.set("numbers", [3, 4], force=True)
    assert state.get("numbers") == [3, 4]


def test_state_set_with_handler_override(basic_schema):
    state = State(basic_schema)

    # Custom handler that concatenates strings
    custom_handler = lambda current, new: f"{current}-{new}" if current else new

    state.set("name", "first")
    state.set("name", "second", handler_override=custom_handler)
    assert state.get("name") == "first-second"


def test_state_has(basic_schema):
    state = State(basic_schema, {"name": "test"})
    assert state.has("name") is True
    assert state.has("non_existent") is False


def test_state_to_dict(basic_schema):
    state = State(basic_schema)
    serialized = state.to_dict()
    assert isinstance(serialized, dict)
    assert "numbers" in serialized
    assert "metadata" in serialized
    assert "name" in serialized
    assert serialized["numbers"]["type"] == "list"
    assert serialized["metadata"]["type"] == "dict"
    assert serialized["name"]["type"] == "str"


def test_state_from_dict():
    data = {
        "numbers": {"type": "list"},
        "metadata": {"type": "dict"},
        "name": {"type": "str"},
    }
    state = State.from_dict(data)
    assert isinstance(state, State)
    assert "numbers" in state.schema
    assert "metadata" in state.schema
    assert "name" in state.schema


# Test edge cases
def test_state_empty_schema():
    state = State({})
    assert state.data == {}
    state.set("any_key", "value")  # Should work with empty schema
    assert state.get("any_key") == "value"


def test_state_none_values(basic_schema):
    state = State(basic_schema)
    state.set("name", None)
    assert state.get("name") is None
    state.set("name", "value")
    assert state.get("name") == "value"


def test_state_invalid_types(basic_schema):
    state = State(basic_schema)
    # Setting wrong type should still work (Python's dynamic typing)
    state.set("numbers", "not_a_list")
    assert state.get("numbers") == "not_a_list"

    # Next set with correct type should handle the invalid previous value
    state.set("numbers", [1, 2])
    assert state.get("numbers") == [1, 2]


# Test complex nested structures
def test_state_nested_structures():
    schema = {
        "complex": {
            "type": Dict[str, List[int]],
            "handler": lambda current, new: {
                k: current.get(k, []) + new.get(k, [])
                for k in set(current.keys()) | set(new.keys())
            } if current else new
        }
    }

    state = State(schema)
    state.set("complex", {"a": [1, 2], "b": [3, 4]})
    state.set("complex", {"b": [5, 6], "c": [7, 8]})

    expected = {
        "a": [1, 2],
        "b": [3, 4, 5, 6],
        "c": [7, 8]
    }
    assert state.get("complex") == expected
