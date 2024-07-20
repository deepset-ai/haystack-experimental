# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from typing import Any, Callable, Dict, List, Optional, Union, cast


def create_function_payload_extractor(
    arguments_field_name: str,
) -> Callable[[Any], Dict[str, Any]]:
    """
    Extracts invocation payload from a given LLM completion containing function invocation.

    :param arguments_field_name: The name of the field containing the function arguments.
    :return: A function that extracts the function invocation details from the LLM payload.
    """

    def _extract_function_invocation(payload: Any) -> Dict[str, Any]:
        """
        Extract the function invocation details from the payload.

        :param payload: The LLM fc payload to extract the function invocation details from.
        """
        fields_and_values = _search(payload, arguments_field_name)
        if fields_and_values:
            arguments = fields_and_values.get(arguments_field_name)
            if not isinstance(arguments, (str, dict)):
                raise ValueError(
                    f"Invalid {arguments_field_name} type {type(arguments)} for function call, expected str/dict"
                )
            return {
                "name": fields_and_values.get("name"),
                "arguments": (
                    json.loads(arguments) if isinstance(arguments, str) else arguments
                ),
            }
        return {}

    return _extract_function_invocation


def _get_dict_converter(
    obj: Any, method_names: Optional[List[str]] = None
) -> Union[Callable[[], Dict[str, Any]], None]:
    method_names = method_names or [
        "model_dump",
        "dict",
    ]  # search for pydantic v2 then v1
    for attr in method_names:
        if hasattr(obj, attr) and callable(getattr(obj, attr)):
            return getattr(obj, attr)
    return None


def _is_primitive(obj) -> bool:
    return isinstance(obj, (int, float, str, bool, type(None)))


def _required_fields(arguments_field_name: str) -> List[str]:
    return ["name", arguments_field_name]


def _search(payload: Any, arguments_field_name: str) -> Dict[str, Any]:
    if _is_primitive(payload):
        return {}
    if dict_converter := _get_dict_converter(payload):
        payload = dict_converter()
    elif dataclasses.is_dataclass(payload):
        # Cast payload to Any to satisfy mypy 1.11.0
        payload = dataclasses.asdict(cast(Any, payload))
    if isinstance(payload, dict):
        if all(field in payload for field in _required_fields(arguments_field_name)):
            # this is the payload we are looking for
            return payload
        for value in payload.values():
            result = _search(value, arguments_field_name)
            if result:
                return result
    elif isinstance(payload, list):
        for item in payload:
            result = _search(item, arguments_field_name)
            if result:
                return result
    return {}
