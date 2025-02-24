# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional

from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.type_serialization import deserialize_type, serialize_type

from haystack_experimental.dataclasses.state_utils import _is_valid_type, merge_values


def _schema_to_dict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a schema dictionary to a serializable format.

    Converts each parameter's type and optional handler function into a serializable
    format using type and callable serialization utilities.

    :param schema: Dictionary mapping parameter names to their type and handler configs
    :returns: Dictionary with serialized type and handler information
    """
    serialized_schema = {}
    for param, config in schema.items():
        serialized_schema[param] = {"type": serialize_type(config["type"])}
        if config.get("handler"):
            serialized_schema[param]["handler"] = serialize_callable(config["handler"])

    return serialized_schema

def _schema_from_dict(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a serialized schema dictionary back to its original format.

    Deserializes the type and optional handler function for each parameter from their
    serialized format back into Python types and callables.

    :param schema: Dictionary containing serialized schema information
    :returns: Dictionary with deserialized type and handler configurations
    """
    deserialized_schema = {}
    for param, config in schema.items():
        deserialized_schema[param] = {"type": deserialize_type(config["type"])}

        if config.get("handler"):
            deserialized_schema[param]["handler"] = deserialize_callable(config["handler"])

    return deserialized_schema

def _validate_schema(schema: Dict[str, Any]) -> None:
    """
    Validate that a schema dictionary meets all required constraints.

    Checks that each parameter definition has a valid type field and that any handler
    specified is a callable function.

    :param schema: Dictionary mapping parameter names to their type and handler configs
    :raises ValueError: If schema validation fails due to missing or invalid fields
    """
    for param, definition in schema.items():
        if "type" not in definition:
            raise ValueError(f"StateSchema: Key '{param}' is missing a 'type' entry.")
        if not _is_valid_type(definition["type"]):
            raise ValueError(
                f"StateSchema: 'type' for key '{param}' must be a Python type, "
                f"got {definition['type']}"
            )
        if definition.get("handler") is not None and not callable(definition["handler"]):
            raise ValueError(
                f"StateSchema: 'handler' for key '{param}' must be callable or None"
            )


class State:
    """
    A dataclass that wraps a StateSchema and maintains an internal _data dictionary.

    Each schema entry has:
      {
        "type": SomeType,
        "handler": Optional[Callable[[Any, Any], Any]]
      }
    """
    def __init__(
        self,
        schema: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None
    ):
        _validate_schema(schema)
        self.schema = schema
        self._data = data or {}

        if data:
            for key, val in data.items():
                self.set(key, val, force=True)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the state by key.

        :param key: Key to look up in the state
        :param default: Value to return if key is not found
        :returns: Value associated with key or default if not found
        """
        return self._data.get(key, default)

    def set(
        self,
        key: str,
        value: Any,
        handler_override: Optional[Callable[[Any, Any], Any]] = None,
        force: bool = False
    ) -> None:
        """
        Set or merge a value in the state according to schema rules.

        Value is merged or overwritten according to these rules:
          - If force=True, just overwrite
          - else if handler_override is given, use that
          - else if the schema for 'key' has a custom handler, use it
          - else use default merge logic

        :param key: Key to store the value under
        :param value: Value to store or merge
        :param handler_override: Optional function to override default merge behavior
        :param force: If True, overwrites existing value without merging
        """
        # If key not in schema, we consider no special merging => just store
        definition = self.schema.get(key, {})
        declared_type = definition.get("type")
        declared_handler = definition.get("handler")

        if force or (not declared_type and not declared_handler and not handler_override):
            self._data[key] = value
            return

        current_value = self._data.get(key, None)

        # if the current value was None and no merging needed, just store
        if current_value is None and not declared_handler and not handler_override:
            self._data[key] = value
            return

        # pick the handler (override > declared > default)
        handler = handler_override or declared_handler
        if handler:
            self._data[key] = handler(current_value, value)
        else:
            # default merging
            if declared_type is None:
                # no known type => just replace
                self._data[key] = value
            else:
                self._data[key] = merge_values(current_value, value, declared_type)

    @property
    def data(self):
        """
        All current data of the state.
        """
        return self._data

    def has(self, key: str) -> bool:
        """
        Check if a key exists in the state.

        :param key: Key to check for existence
        :returns: True if key exists in state, False otherwise
        """
        return key in self._data
