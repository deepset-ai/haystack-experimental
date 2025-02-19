# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Optional

from haystack.utils.callable_serialization import serialize_callable, deserialize_callable
from haystack.utils.type_serialization import serialize_type, deserialize_type

import inspect

def _is_valid_type(obj: Any) -> bool:
    # True if it's a normal class (e.g. str, dict, MyCustomClass)
    # or a generic type (List[str], Dict[str, CustomClass], etc.)
    return inspect.isclass(obj) or type(obj).__name__ in {"_GenericAlias", "GenericAlias"}


def _default_merge_handler(current_value: Any, new_value: Any, declared_type: Any) -> Any:
    """
    Default merging logic for different types:
      - Lists: extend if new_value is also a list, otherwise append
      - Dicts: shallow update
      - Primitives/other: replace entirely
    """
    if current_value is None:
        return new_value

    # Check if declared_type is a list (or List[...] annotation)
    if declared_type is list or (hasattr(declared_type, "__origin__") and declared_type.__origin__ is list):
        if isinstance(new_value, list):
            if isinstance(current_value, list):
                current_value.extend(new_value)
                return current_value
            else:
                # current wasn't a list, just replace
                return new_value
        else:
            # new_value isn't a list => append
            if isinstance(current_value, list):
                current_value.append(new_value)
                return current_value
            else:
                return [current_value, new_value]

    # Check if declared_type is a dict (or Dict[...] annotation)
    if declared_type is dict or (hasattr(declared_type, "__origin__") and declared_type.__origin__ is dict):
        if isinstance(new_value, dict) and isinstance(current_value, dict):
            current_value.update(new_value)
            return current_value
        else:
            # fallback: just replace
            return new_value

    # Otherwise, just replace
    return new_value

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
        deserialized_schema[param] = deserialize_type(config["type"])

        if config.get("handler"):
            deserialized_schema[param] = deserialize_callable(config["handler"])

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
                self._data[key] = _default_merge_handler(current_value, value, declared_type)

    @property
    def data(self):
        return self._data

    def has(self, key: str) -> bool:
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the entire internal state as a dictionary (shallow copy).
        """
        return _schema_to_dict(self.schema)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        return cls(_schema_from_dict(data))
