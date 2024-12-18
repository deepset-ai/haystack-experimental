# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from haystack import component
from pydantic import BaseModel
from haystack import Pipeline
from haystack_experimental.dataclasses import ChatMessage, ToolCall, ChatRole
from haystack_experimental.components.tools.tool_invoker import ToolInvoker
from haystack_experimental.components.generators.chat import OpenAIChatGenerator

from haystack_experimental.dataclasses.tool import Tool


### Component and Model Definitions

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
    """A dataclass representing a physical address."""
    street: str
    city: str


@dataclass
class Person:
    """A person with an address."""
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


## Unit tests
class TestToolComponent:
    def test_from_component_basic(self):
        component = SimpleComponent()

        tool = Tool.from_component(
            component=component,
            name="hello_tool",
            description="A hello tool"
        )

        assert tool.name == "hello_tool"
        assert tool.description == "A hello tool"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to generate."
                }
            },
            "required": ["text"]
        }

        # Test tool invocation
        result = tool.invoke(text="world")
        assert isinstance(result, dict)
        assert "reply" in result
        assert result["reply"] == "Hello, world!"

    def test_from_component_with_dataclass(self):
        component = UserGreeter()

        tool = Tool.from_component(
            component=component,
            name="user_info_tool",
            description="A tool that returns user information"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "user": {
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
            "required": ["user"]
        }

        # Test tool invocation
        result = tool.invoke(user={"name": "Alice", "age": 30})
        assert isinstance(result, dict)
        assert "message" in result
        assert result["message"] == "User Alice is 30 years old"

    def test_from_component_with_list_input(self):
        component = ListProcessor()

        tool = Tool.from_component(
            component=component,
            name="list_processing_tool",
            description="A tool that concatenates strings"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "array",
                    "description": "The list of strings to concatenate.",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["texts"]
        }

        # Test tool invocation
        result = tool.invoke(texts=["hello", "world"])
        assert isinstance(result, dict)
        assert "concatenated" in result
        assert result["concatenated"] == "hello world"

    def test_from_component_with_pydantic_model(self):
        component = ProductProcessor()

        tool = Tool.from_component(
            component=component,
            name="product_tool",
            description="A tool that processes products"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "product": {
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
            "required": ["product"]
        }

        # Test tool invocation
        result = tool.invoke(product={"name": "Widget", "price": 19.99})
        assert isinstance(result, dict)
        assert "description" in result
        assert result["description"] == "The product Widget costs $19.99."

    def test_from_component_with_nested_dataclass(self):
        component = PersonProcessor()

        tool = Tool.from_component(
            component=component,
            name="person_tool",
            description="A tool that processes people"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "person": {
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
            "required": ["person"]
        }

        # Test tool invocation
        result = tool.invoke(person={
            "name": "Diana",
            "address": {
                "street": "123 Elm Street",
                "city": "Metropolis"
            }
        })
        assert isinstance(result, dict)
        assert "info" in result
        assert result["info"] == "Diana lives at 123 Elm Street, Metropolis."


## Integration tests
class TestToolComponentInPipeline:

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_component_tool_in_pipeline(self):
        # Create component and convert it to tool
        component = SimpleComponent()
        tool = Tool.from_component(
            component=component,
            name="hello_tool",
            description="A tool that generates a greeting message for the user"
        )

        # Create pipeline with OpenAIChatGenerator and ToolInvoker
        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Hello, I'm Vladimir")

        # Run pipeline
        result = pipeline.run({"llm": {"messages": [message]}})

        # Check results
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_result.result == str({"reply": "Hello, Vladimir!"})
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_user_greeter_in_pipeline(self):
        component = UserGreeter()
        tool = Tool.from_component(
            component=component,
            name="user_greeter",
            description="A tool that greets users with their name and age"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="I am Alice and I'm 30 years old")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_result.result == str({"message": "User Alice is 30 years old"})
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_list_processor_in_pipeline(self):
        component = ListProcessor()
        tool = Tool.from_component(
            component=component,
            name="list_processor",
            description="A tool that concatenates a list of strings"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Can you join these words: hello, beautiful, world")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_result.result == str({"concatenated": "hello beautiful world"})
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_product_processor_in_pipeline(self):
        component = ProductProcessor()
        tool = Tool.from_component(
            component=component,
            name="product_processor",
            description="A tool that creates a description for a product with its name and price"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Can you describe a product called Widget that costs $19.99?")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_result.result == str({"description": "The product Widget costs $19.99."})
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_person_processor_in_pipeline(self):
        component = PersonProcessor()
        tool = Tool.from_component(
            component=component,
            name="person_processor",
            description="A tool that processes information about a person and their address"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Diana lives at 123 Elm Street in Metropolis")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert tool_message.tool_call_result.result == str({"info": "Diana lives at 123 Elm Street, Metropolis."})
        assert not tool_message.tool_call_result.error
