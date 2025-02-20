# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Dict

from haystack.utils.callable_serialization import serialize_callable, deserialize_callable
from haystack.utils.type_serialization import serialize_type, deserialize_type

from haystack_experimental.dataclasses.state_utils import _is_valid_type, merge_values


def _schema_to_dict(schema: Dict[str, Any]) -> Dict[str, Any]:
    serialized_schema = {}
    for param, config in schema.items():
        serialized_schema[param] = serialize_type(config["type"])
        if config.get("handler"):
            serialized_schema[param] = serialize_callable(config["handler"])

    return serialized_schema

def _schema_from_dict(schema: Dict[str, Any]) -> Dict[str, Any]:
    deserialized_schema = {}
    for param, config in schema.items():
        deserialized_schema[param] = {"type": deserialize_type(config["type"])}

        if config.get("handler"):
            deserialized_schema[param]["handler"] = deserialize_callable(config["handler"])

    return deserialized_schema

def _validate_schema(schema: Dict[str, Any]) -> None:
    for param, definition in schema.items():
        if "type" not in definition:
            raise ValueError(f"StateSchema: Key '{param}' is missing a 'type' entry.")
        if not _is_valid_type(definition["type"]):
            raise ValueError(
                f"StateSchema: 'type' for key '{param}' must be a Python type, "
                f"got {definition['type']}"
            )
        if "handler" in definition and definition["handler"] is not None:
            if not callable(definition["handler"]):
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
        return self._data.get(key, default)

    def set(
        self,
        key: str,
        value: Any,
        handler_override: Optional[Callable[[Any, Any], Any]] = None,
        force: bool = False
    ) -> None:
        """
        Merge or overwrite 'value' into _data[key] according to:
          - If force=True, just overwrite
          - else if the schema for 'key' has a custom handler, use it
          - else if handler_override is given, use that
          - else use default merge logic
        """
        # If key not in schema, we consider no special merging => just store
        # or you can choose to raise an error if it's not in schema
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
        return self._data

    def has(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the entire internal state as a dictionary (shallow copy).
        """
        # TODO: think about full state serialization
        return _schema_to_dict(self.schema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        # TODO: think about full state deserialization
        return cls(_schema_from_dict(data))
