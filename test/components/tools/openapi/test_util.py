from haystack_experimental.components.tools.utils import normalize_tool_definition, normalize_function_name


def test_normalize_function_name():
    assert normalize_function_name("test-function") == "test_function"
    assert normalize_function_name("missing-operation-id_get") == "missing_operation_id_get"
    assert normalize_function_name("/test/function/with/slashes-and-dashes") == "test_function_with_slashes_and_dashes"
    assert normalize_function_name("test\\function\\with\\backslashes") == "test_function_with_backslashes"
    assert normalize_function_name("-test-function-with-dashes-") == "test_function_with_dashes"


def test_normalize_tool_definition():
    function = {
        "name": "_test-function/with/slashes-and-dashes_",
        "description": "Test function description at least 50 characters long " * 100,
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA" * 100,
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        }
    }
    sanitized_function = normalize_tool_definition(function)
    assert sanitized_function["name"] == "test_function_with_slashes_and_dashes"
    assert len(sanitized_function["description"]) <= 1024
    assert len(sanitized_function["parameters"]["properties"]["location"]["description"]) <= 1024
    assert len(sanitized_function["parameters"]["properties"]["format"]["description"]) <= 1024
