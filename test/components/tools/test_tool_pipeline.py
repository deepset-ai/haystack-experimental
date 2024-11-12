import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from haystack import Pipeline, component
from pydantic import BaseModel
import pytest
from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator
from haystack_experimental.components.tools.tool_invoker import ToolInvoker
from haystack_experimental.dataclasses.chat_message import ChatMessage
from haystack_experimental.dataclasses.tool import Tool
import json


### Component and Model Definitions, helper classes used in the tests

### Some classes used in the tests have comments and some don't.
### This is intentional to test that the tool schema generation works for both commented and uncommented dataclasses.
@component
class SimpleComponent:
    """A simple component that generates text."""

    @component.output_types(reply=str)
    def run(self, text: str) -> Dict[str, str]:
        """
        A simple component that generates text.

        :param text: The text to generate.
        :return: A dictionary with the generated text.
        """
        return {"reply": f"Hello, {text}!"}

class Product(BaseModel):
    """A product model."""
    name: str
    price: float

@dataclass
class User:
    """A simple user dataclass."""
    name: str = "Anonymous"
    age: int = 0

@component
class UserGreeter:
    """A simple component that processes a User."""

    @component.output_types(message=str)
    def run(self, user: User) -> Dict[str, str]:
        """
        A simple component that processes a User.

        :param user: The User object to process.
        :return: A dictionary with a message about the user.
        """
        return {"message": f"User {user.name} is {user.age} years old"}

@component
class ListProcessor:
    """A component that processes a list of strings."""

    @component.output_types(concatenated=str)
    def run(self, texts: List[str]) -> Dict[str, str]:
        """
        Concatenates a list of strings into a single string.

        :param texts: The list of strings to concatenate.
        :return: A dictionary with the concatenated string.
        """
        return {"concatenated": ' '.join(texts)}

@component
class UsersProcessor:
    """A component that processes a list of Users."""

    @component.output_types(summary=str)
    def run(self, users: List[User]) -> Dict[str, str]:
        """
        Processes a list of users and returns a summary.

        :param users: The list of User objects to process.
        :return: A dictionary with a summary of the users.
        """
        names = [user.name for user in users]
        return {"summary": f"Processing users: {', '.join(names)}"}

@component
class MixedInputComponent:
    """A component that processes both a string and a list of Users."""

    @component.output_types(result=str)
    def run(self, greeting: str, users: List[User]) -> Dict[str, str]:
        """
        Greets a list of users with the provided greeting.

        :param greeting: The greeting to use.
        :param users: The list of User objects to greet.
        :return: A dictionary with the greeting result.
        """
        names = [user.name for user in users]
        return {"result": f"{greeting}, {', '.join(names)}!"}

@component
class MultiTypeInputComponent:
    """A component that processes a string, a User, and a list of strings."""

    @component.output_types(summary=str)
    def run(self, message: str, user: User, tags: List[str]) -> Dict[str, str]:
        """
        Creates a summary using the provided message, user, and tags.

        :param message: A message string.
        :param user: A User object.
        :param tags: A list of tags.
        :return: A dictionary with the summary.
        """
        tags_str = ', '.join(tags)
        return {"summary": f"{message} by {user.name} (age {user.age}) with tags: {tags_str}"}

@component
class DictInputComponent:
    """A component that processes a dictionary of string keys to integer values."""

    @component.output_types(total=int)
    def run(self, data: Dict[str, int]) -> Dict[str, int]:
        """
        Sums the values in the dictionary.

        :param data: A dictionary of integer values.
        :return: A dictionary with the total sum.
        """
        total = sum(data.values())
        return {"total": total}

@component
class ProductProcessor:
    """A component that processes a Product."""

    @component.output_types(description=str)
    def run(self, product: Product) -> Dict[str, str]:
        """
        Creates a description for the product.

        :param product: The Product to process.
        :return: A dictionary with the product description.
        """
        return {
            "description": f"The product {product.name} costs ${product.price:.2f}."
        }

@dataclass
class Address:
    """A dataclass representing a physical address.

    Attributes:
        street (str): The street address including house/building number.
        city (str): The name of the city.
    """
    street: str
    city: str

@dataclass
class Person:
    name: str
    address: Address

@component
class PersonProcessor:
    """A component that processes a Person with nested Address."""

    @component.output_types(info=str)
    def run(self, person: Person) -> Dict[str, str]:
        """
        Creates information about the person.

        :param person: The Person to process.
        :return: A dictionary with the person's information.
        """
        return {
            "info": f"{person.name} lives at {person.address.street}, {person.address.city}."
        }

@dataclass
class Profile:
    username: str
    bio: Optional[str] = None

@component
class ProfileProcessor:
    """A component that processes a Profile with an optional bio."""

    @component.output_types(output=str)
    def run(self, profile: Profile) -> Dict[str, str]:
        """
        Creates a profile output.

        :param profile: The Profile to process.
        :return: A dictionary with the profile output.
        """
        bio = profile.bio or "No bio provided"
        return {
            "output": f"User {profile.username}: {bio}"
        }

@component
class OptionalDictComponent:
    """A component that processes an optional dictionary with Any type values."""

    @component.output_types(output=str)
    def run(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Processes an optional dictionary.

        :param data: An optional dictionary with values of any type.
        :return: A dictionary with a message about the input data.
        """
        if data is None:
            return {"output": "No data provided"}
        else:
            keys = ', '.join(data.keys())
            return {"output": f"Received data with keys: {keys}"}


class TestToolPipeline:
    ### Basic Pipeline Tests without LLM involved

    def test_from_pipeline_basic(self):
        pipeline = Pipeline()
        pipeline.add_component("simple", SimpleComponent())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="hello_replying_tool",
            description="A hello replying tool"
        )

        # Test that the tool was created correctly
        assert tool.name == "hello_replying_tool"
        assert tool.description == "A hello replying tool"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "simple.text": {
                    "type": "string",
                    "description": "The text to generate."
                }
            },
            "required": ["simple.text"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {"simple.text": "world"}
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "simple" in result
        assert "reply" in result["simple"]
        assert result["simple"]["reply"] == "Hello, world!"

    def test_from_pipeline_basic_with_two_connected_components(self):
        pipeline = Pipeline()

        pipeline.add_component("simple", SimpleComponent())
        pipeline.add_component("simple2", SimpleComponent())
        pipeline.connect("simple.reply", "simple2.text")

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="hello_replying_tool",
            description="A hello replying tool"
        )

        # Only simple.text is pipeline input because simple2.text is connected to simple.reply
        assert tool.name == "hello_replying_tool"
        assert tool.description == "A hello replying tool"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "simple.text": {
                    "type": "string",
                    "description": "The text to generate."
                }
            },
            "required": ["simple.text"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {"simple.text": "world"}
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "simple2" in result
        assert "reply" in result["simple2"]
        assert result["simple2"]["reply"] == "Hello, Hello, world!!"

    def test_from_pipeline_with_dataclass_input(self):
        pipeline = Pipeline()
        pipeline.add_component("user_greeter", UserGreeter())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="user_info_tool",
            description="A tool that returns user information"
        )

        # Update the assertion to match the actual schema generated by Tool.from_pipeline
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "user_greeter.user": {
                    "type": "object",
                    "description": "The User object to process.",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Field 'name' of 'User'."
                        },
                        "age": {
                            "type": "integer",
                            "description": "Field 'age' of 'User'."
                        }
                    }
                }
            },
            "required": ["user_greeter.user"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {"user_greeter.user": {"name": "Alice", "age": 30}}
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "user_greeter" in result
        assert "message" in result["user_greeter"]
        assert result["user_greeter"]["message"] == "User Alice is 30 years old"

    def test_from_pipeline_with_list_input(self):
        pipeline = Pipeline()
        pipeline.add_component("list_processor", ListProcessor())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="list_processing_tool",
            description="A tool that concatenates a list of strings"
        )

        # Test that the tool was created correctly
        assert tool.name == "list_processing_tool"
        assert tool.description == "A tool that concatenates a list of strings"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "list_processor.texts": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "The list of strings to concatenate."
                    }
                },
            "required": ["list_processor.texts"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {"list_processor.texts": ["hello", "world"]}
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "list_processor" in result
        assert "concatenated" in result["list_processor"]
        assert result["list_processor"]["concatenated"] == "hello world"

    def test_from_pipeline_with_list_of_dataclasses(self):
        pipeline = Pipeline()
        pipeline.add_component("users_processor", UsersProcessor())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="users_processing_tool",
            description="A tool that processes multiple users"
        )

        # Test that the tool was created correctly
        assert tool.name == "users_processing_tool"
        assert tool.description == "A tool that processes multiple users"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "users_processor.users": {
                    "type": "array",
                    "description": "The list of User objects to process.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Field 'name' of 'User'."
                            },
                            "age": {
                                "type": "integer",
                                "description": "Field 'age' of 'User'."
                            }
                        }
                    }
                }
            },
            "required": ["users_processor.users"]
        }

        # Test that the tool can be invoked
        users_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        llm_prepared_input = {"users_processor.users": users_data}
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "users_processor" in result
        assert "summary" in result["users_processor"]
        assert result["users_processor"]["summary"] == "Processing users: Alice, Bob"

    def test_from_pipeline_with_mixed_inputs(self):
        pipeline = Pipeline()
        pipeline.add_component("mixed_input", MixedInputComponent())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="greeting_tool",
            description="A tool that greets users with a greeting message"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "mixed_input.greeting": {
                    "type": "string",
                    "description": "The greeting to use."
                },
                "mixed_input.users": {
                    "type": "array",
                    "description": "The list of User objects to greet.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Field 'name' of 'User'."
                            },
                            "age": {
                                "type": "integer",
                                "description": "Field 'age' of 'User'."
                            }
                        }
                    }
                }
            },
            "required": ["mixed_input.greeting", "mixed_input.users"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {
            "mixed_input.greeting": "Hello",
            "mixed_input.users": [
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ]
        }
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "mixed_input" in result
        assert "result" in result["mixed_input"]
        assert result["mixed_input"]["result"] == "Hello, Alice, Bob!"

    def test_from_pipeline_with_multiple_input_types(self):
        pipeline = Pipeline()
        pipeline.add_component("multi_type", MultiTypeInputComponent())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="summary_tool",
            description="A tool that summarizes inputs"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "multi_type.message": {
                    "type": "string",
                    "description": "A message string."
                },
                "multi_type.user": {
                    "type": "object",
                    "description": "A User object.",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Field 'name' of 'User'."
                        },
                        "age": {
                            "type": "integer",
                            "description": "Field 'age' of 'User'."
                        }
                    }
                },
                "multi_type.tags": {
                    "type": "array",
                    "description": "A list of tags.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["multi_type.message", "multi_type.user", "multi_type.tags"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {
            "multi_type.message": "This is a test",
            "multi_type.user": {"name": "Charlie", "age": 28},
            "multi_type.tags": ["example", "test", "pipeline"]
        }
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "multi_type" in result
        assert "summary" in result["multi_type"]
        expected_summary = "This is a test by Charlie (age 28) with tags: example, test, pipeline"
        assert result["multi_type"]["summary"] == expected_summary

    def test_from_pipeline_with_dict_input(self):
        pipeline = Pipeline()
        pipeline.add_component("dict_input", DictInputComponent())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="sum_tool",
            description="A tool that sums integer values in a dictionary"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "dict_input.data": {
                    "type": "object",
                    "description": "A dictionary of integer values.",
                    "additionalProperties": {
                        "description": "A dictionary of integer values.",
                        "type": "integer"
                    }
                }
            },
            "required": ["dict_input.data"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {
            "dict_input.data": {"a": 1, "b": 2, "c": 3}
        }
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "dict_input" in result
        assert "total" in result["dict_input"]
        assert result["dict_input"]["total"] == 6

    def test_from_pipeline_with_pydantic_model(self):
        pipeline = Pipeline()
        pipeline.add_component("product_processor", ProductProcessor())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="product_description_tool",
            description="A tool that generates product descriptions"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "product_processor.product": {
                    "type": "object",
                    "description": "The Product to process.",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Field 'name' of 'Product'."
                        },
                        "price": {
                            "type": "number",
                            "description": "Field 'price' of 'Product'."
                        }
                    },
                    "required": ["name", "price"]
                }
            },
            "required": ["product_processor.product"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {
            "product_processor.product": {"name": "Widget", "price": 19.99}
        }
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "product_processor" in result
        assert "description" in result["product_processor"]
        assert result["product_processor"]["description"] == "The product Widget costs $19.99."

    def test_from_pipeline_with_nested_dataclass(self):

        pipeline = Pipeline()
        pipeline.add_component("person_processor", PersonProcessor())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="person_info_tool",
            description="A tool that provides information about a person"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "person_processor.person": {
                    "type": "object",
                    "description": "The Person to process.",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Field 'name' of 'Person'."
                        },
                        "address": {
                            "type": "object",
                            "description": "Field 'address' of 'Person'.",
                            "properties": {
                                "street": {
                                    "type": "string",
                                    "description": "Field 'street' of 'Address'."
                                },
                                "city": {
                                    "type": "string",
                                    "description": "Field 'city' of 'Address'."
                                }
                            },
                            "required": ["street", "city"]
                        }
                    },
                    "required": ["name", "address"]
                }
            },
            "required": ["person_processor.person"]
        }

        # Test that the tool can be invoked
        llm_prepared_input = {
            "person_processor.person": {
                "name": "Diana",
                "address": {
                    "street": "123 Elm Street",
                    "city": "Metropolis"
                }
            }
        }
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "person_processor" in result
        assert "info" in result["person_processor"]
        assert result["person_processor"]["info"] == "Diana lives at 123 Elm Street, Metropolis."

    def test_from_pipeline_with_optional_fields(self):
        pipeline = Pipeline()
        pipeline.add_component("profile_processor", ProfileProcessor())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="profile_tool",
            description="A tool that processes user profiles"
        )

        # The 'bio' field is optional, so it should not be required
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "profile_processor.profile": {
                    "type": "object",
                    "description": "The Profile to process.",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "Field 'username' of 'Profile'."
                        },
                        "bio": {
                            "type": "string",
                            "description": "Field 'bio' of 'Profile'.",
                        }
                    },
                    "required": ["username"]
                }
            },
            "required": ["profile_processor.profile"]
        }

        # Test that the tool can be invoked without the optional bio
        llm_prepared_input = {
            "profile_processor.profile": {
                "username": "johndoe"
            }
        }
        result = tool.invoke(**llm_prepared_input)

        assert isinstance(result, dict)
        assert "profile_processor" in result
        assert "output" in result["profile_processor"]
        assert result["profile_processor"]["output"] == "User johndoe: No bio provided"

        # Now invoke with the bio provided
        llm_prepared_input["profile_processor.profile"]["bio"] = "Just another developer"
        result = tool.invoke(**llm_prepared_input)

        assert result["profile_processor"]["output"] == "User johndoe: Just another developer"

    def test_from_pipeline_with_optional_dict_any_input(self):
        """
        Test pipeline with a component that accepts Optional[Dict[str, Any]].
        """
        pipeline = Pipeline()
        pipeline.add_component("optional_dict", OptionalDictComponent())

        tool = Tool.from_pipeline(
            pipeline=pipeline,
            name="optional_dict_tool",
            description="A tool that processes optional dictionary input with Any type values"
        )

        # Assert that the tool's parameters are correctly generated
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "optional_dict.data": {
                    "type": "object",
                    "description": "An optional dictionary with values of any type.",
                    "additionalProperties": {}
                }
            }
            # Note: 'required' is not included since the 'data' parameter is optional
        }

        # Test invocation without providing 'data' (should use default None)
        result = tool.invoke()
        assert isinstance(result, dict)
        assert "optional_dict" in result
        assert result["optional_dict"]["output"] == "No data provided"

        # Test invocation with 'data' provided
        llm_prepared_input = {
            "optional_dict.data": {"key1": 1, "key2": "value2", "key3": [1, 2, 3]}
        }
        result = tool.invoke(**llm_prepared_input)
        assert isinstance(result, dict)
        assert "optional_dict" in result
        assert "Received data with keys: key1, key2, key3" == result["optional_dict"]["output"]




    ### Real integration tests with LLMs (requires OPENAI_API_KEY)
    ### We'll test the end user experience of using pipelines as tools

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_from_pipeline_basic_with_LLM(self):
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("simple", SimpleComponent())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="hello_replying_tool",
            description="A hello replying tool"
        )

        # now the main pipeline that uses the the pipeline above as a tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True))
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Rather than forcing particular tool usage, we slightly hint at it
        messages = [ChatMessage.from_user("Say hello to the world using the tool")]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) == 1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        # Check particularities of the tool result coming from the tool pipeline
        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "simple" in parsed_result
        assert "reply" in parsed_result["simple"]
        assert "hello" in parsed_result["simple"]["reply"].lower()
        assert "world" in parsed_result["simple"]["reply"].lower()


    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_basic_with_two_connected_components_with_LLM(self):
        # Create the tool pipeline with two connected components
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("simple", SimpleComponent())
        tool_pipeline.add_component("simple2", SimpleComponent())
        tool_pipeline.connect("simple.reply", "simple2.text")

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="hello_replying_tool",
            description="A hello replying tool that uses two connected components"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True))
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message to the LLM that hints at using the tool
        messages = [ChatMessage.from_user("Say hello to the world twice using the tool")]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >= 1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "simple2" in parsed_result
        assert "reply" in parsed_result["simple2"]
        reply = parsed_result["simple2"]["reply"].lower()
        assert "hello" in reply
        assert "world" in reply

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_dataclass_input_with_LLM(self):
        # Create the tool pipeline with UserGreeter component
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("user_greeter", UserGreeter())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="user_info_tool",
            description="A tool that returns user information"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to provide information about a user named Alice who is 30 years old"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >= 1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "user_greeter" in parsed_result
        assert "message" in parsed_result["user_greeter"]
        message = parsed_result["user_greeter"]["message"]
        assert "alice" in message.lower()
        assert "30" in message

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_list_input_with_LLM(self):
        # Create the tool pipeline with ListProcessor component
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("list_processor", ListProcessor())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="list_processing_tool",
            description="A tool that concatenates a list of strings"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to concatenate the words 'hello' and 'world'"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >= 1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "list_processor" in parsed_result
        assert "concatenated" in parsed_result["list_processor"]
        concatenated = parsed_result["list_processor"]["concatenated"].lower()
        assert concatenated == "hello world"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_list_of_dataclasses_with_LLM(self):
        # Create the tool pipeline with UsersProcessor component
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("users_processor", UsersProcessor())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="users_processing_tool",
            description="A tool that processes multiple users"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to process users Alice aged 30 and Bob aged 25"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >= 1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "users_processor" in parsed_result
        assert "summary" in parsed_result["users_processor"]
        summary = parsed_result["users_processor"]["summary"]
        assert "alice" in summary.lower()
        assert "bob" in summary.lower()
        assert "processing users" in summary.lower()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_mixed_inputs_with_LLM(self):
        # Create the tool pipeline with MixedInputComponent
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("mixed_input", MixedInputComponent())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="greeting_tool",
            description="A tool that greets users with a greeting message"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to greet users Alice and Bob with 'Hello'"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >= 1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "mixed_input" in parsed_result
        assert "result" in parsed_result["mixed_input"]
        result_text = parsed_result["mixed_input"]["result"]
        assert "hello, alice, bob" in result_text.lower()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_multiple_input_types_with_LLM(self):
        # Create the tool pipeline with MultiTypeInputComponent
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("multi_type", MultiTypeInputComponent())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="summary_tool",
            description="A tool that summarizes inputs"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to summarize 'This is a test' by user Charlie aged 28 with tags: example, test, pipeline"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >=1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "multi_type" in parsed_result
        assert "summary" in parsed_result["multi_type"]
        summary = parsed_result["multi_type"]["summary"]
        assert "this is a test" in summary.lower()
        assert "charlie" in summary.lower()
        assert "28" in summary
        assert "example, test, pipeline" in summary.lower()

    @pytest.mark.integration
    @pytest.mark.skip("There is no support for additionalProperties OpenAI schema option")
    def test_from_pipeline_with_dict_input_with_LLM(self):
        # Create the tool pipeline with DictInputComponent
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("dict_input", DictInputComponent())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="sum_tool",
            description="A tool that sums integer values in a dictionary"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to sum the numbers with dictionary keys 'a':1, 'b':2, 'c':3"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >=1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "dict_input" in parsed_result
        assert "total" in parsed_result["dict_input"]
        assert parsed_result["dict_input"]["total"] == 6

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_pydantic_model_with_LLM(self):
        # Create the tool pipeline with ProductProcessor component
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("product_processor", ProductProcessor())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="product_description_tool",
            description="A tool that generates product descriptions"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to generate a description for a product named 'Widget' priced at 19.99"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >=1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "product_processor" in parsed_result
        assert "description" in parsed_result["product_processor"]
        description = parsed_result["product_processor"]["description"]
        assert "widget" in description.lower()
        assert "$19.99" in description

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_nested_dataclass_with_LLM(self):
        # Create the tool pipeline with PersonProcessor component
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("person_processor", PersonProcessor())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="person_info_tool",
            description="A tool that provides information about a person"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Provide a message that hints at using the tool
        messages = [
            ChatMessage.from_user(
                "Use the tool to provide information about Diana who lives at 123 Elm Street, Metropolis"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})

        # Check that the tool was used and produced the expected result
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]
        assert len(result["tool_invoker"]["tool_messages"]) >=1

        tool_message: ChatMessage = result["tool_invoker"]["tool_messages"][0]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_result is not None

        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "person_processor" in parsed_result
        assert "info" in parsed_result["person_processor"]
        info = parsed_result["person_processor"]["info"]
        assert "diana" in info.lower()
        assert "123 elm street" in info.lower()
        assert "metropolis" in info.lower()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_optional_fields_with_LLM(self):
        # Create the tool pipeline with ProfileProcessor component
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("profile_processor", ProfileProcessor())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="profile_tool",
            description="A tool that processes user profiles"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Test without optional bio
        messages = [
            ChatMessage.from_user(
                "Use the tool to process a profile for user 'johndoe' without a bio"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]

        tool_message = result["tool_invoker"]["tool_messages"][0]
        parsed_result = json.loads(tool_message.tool_call_result.result)
        output = parsed_result["profile_processor"]["output"]
        assert "johndoe" in output
        assert "no bio provided" in output.lower()

        # Test with optional bio
        messages = [
            ChatMessage.from_user(
                "Use the tool to process a profile for user 'johndoe' with bio 'Just another developer'"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})
        tool_message = result["tool_invoker"]["tool_messages"][0]
        parsed_result = json.loads(tool_message.tool_call_result.result)
        output = parsed_result["profile_processor"]["output"]
        assert "johndoe" in output.lower()
        assert "just another developer" in output.lower()

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="Set the OPENAI_API_KEY environment variable to run this test.",
    )
    def test_from_pipeline_with_optional_dict_any_input_with_LLM(self):
        """
        Integration test for pipeline with a component that accepts Optional[Dict[str, Any]],
        using an LLM to generate the tool calls.
        """
        # Create the tool pipeline
        tool_pipeline = Pipeline()
        tool_pipeline.add_component("optional_dict", OptionalDictComponent())

        # Create a tool from the pipeline
        tool = Tool.from_pipeline(
            pipeline=tool_pipeline,
            name="optional_dict_tool",
            description="A tool that processes optional dictionary input with Any type values"
        )

        # Create the main pipeline that uses the tool
        main_pipeline = Pipeline()
        main_pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        main_pipeline.add_component(
            "tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True)
        )
        main_pipeline.connect("llm.replies", "tool_invoker.messages")

        # Test without providing data
        messages = [
            ChatMessage.from_user(
                "Use the tool without providing any data dictionary"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})
        assert "tool_invoker" in result
        assert "tool_messages" in result["tool_invoker"]

        tool_message = result["tool_invoker"]["tool_messages"][0]
        parsed_result = json.loads(tool_message.tool_call_result.result)
        assert "optional_dict" in parsed_result
        assert parsed_result["optional_dict"]["output"] == "No data provided"

        # Test with providing data
        messages = [
            ChatMessage.from_user(
                "Use the tool with a dictionary data field contains the following key/pairs: name: 'Alice', age: 30, and scores: [85, 92, 78]"
            )
        ]
        result = main_pipeline.run(data={"llm": {"messages": messages}})
        tool_message = result["tool_invoker"]["tool_messages"][0]
        parsed_result = json.loads(tool_message.tool_call_result.result)
        output = parsed_result["optional_dict"]["output"]
        # TODO: This doesn't work yet, because the LLM returns an empty dictionary as parameter values
        # Leaving it here as a reminder to explore this further
