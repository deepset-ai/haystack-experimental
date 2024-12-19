# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import pytest
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from haystack import component
from pydantic import BaseModel
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack_experimental.dataclasses import ChatMessage, ChatRole
from haystack_experimental.components.tools.tool_invoker import ToolInvoker
from haystack_experimental.components.generators.chat import OpenAIChatGenerator
from haystack_experimental.components.generators.anthropic.chat import AnthropicChatGenerator
from haystack_experimental.dataclasses.tool import Tool



### Component and Model Definitions

@component
class SimpleComponent:
    """A simple component that generates text."""

    @component.output_types(reply=str)
    def run(self, text: str) -> Dict[str, str]:
        """
        A simple component that generates text.

        :param text: user's introductory message
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


@component
class DocumentProcessor:
    """A component that processes a list of Documents."""

    @component.output_types(concatenated=str)
    def run(self, documents: List[Document]) -> Dict[str, str]:
        """
        Concatenates the content of multiple documents with newlines.

        :param documents: List of Documents whose content will be concatenated
        :returns: Dictionary containing the concatenated document contents
        """
        return {"concatenated": '\n'.join(doc.content for doc in documents)}


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
                    "description": "user's introductory message"
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
                            }
                        }
                    }
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

    def test_from_component_with_document_list(self):
        component = DocumentProcessor()

        tool = Tool.from_component(
            component=component,
            name="document_processor",
            description="A tool that concatenates document contents"
        )

        assert tool.parameters == {
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "List of Documents whose content will be concatenated",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Field 'id' of 'Document'."
                            },
                            "content": {
                                "type": "string",
                                "description": "Field 'content' of 'Document'."
                            },
                            "dataframe": {
                                "type": "string",
                                "description": "Field 'dataframe' of 'Document'."
                            },
                            "blob": {
                                "type": "object",
                                "description": "Field 'blob' of 'Document'.",
                                "properties": {
                                    "data": {
                                        "type": "string",
                                        "description": "Field 'data' of 'ByteStream'."
                                    },
                                    "meta": {
                                        "type": "string",
                                        "description": "Field 'meta' of 'ByteStream'."
                                    },
                                    "mime_type": {
                                        "type": "string",
                                        "description": "Field 'mime_type' of 'ByteStream'."
                                    }
                                }
                            },
                            "meta": {
                                "type": "string",
                                "description": "Field 'meta' of 'Document'."
                            },
                            "score": {
                                "type": "number",
                                "description": "Field 'score' of 'Document'."
                            },
                            "embedding": {
                                "type": "array",
                                "description": "Field 'embedding' of 'Document'.",
                                "items": {
                                    "type": "number"
                                }
                            },
                            "sparse_embedding": {
                                "type": "object",
                                "description": "Field 'sparse_embedding' of 'Document'.",
                                "properties": {
                                    "indices": {
                                        "type": "array",
                                        "description": "Field 'indices' of 'SparseEmbedding'.",
                                        "items": {
                                            "type": "integer"
                                        }
                                    },
                                    "values": {
                                        "type": "array",
                                        "description": "Field 'values' of 'SparseEmbedding'.",
                                        "items": {
                                            "type": "number"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["documents"]
        }

        # Test tool invocation
        result = tool.invoke(documents=[{"content": "First document"}, {"content": "Second document"}])
        assert isinstance(result, dict)
        assert "concatenated" in result
        assert result["concatenated"] == "First document\nSecond document"


## Integration tests
class TestToolComponentInPipelineWithOpenAI:

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

        message = ChatMessage.from_user(text="Vladimir")

        # Run pipeline
        result = pipeline.run({"llm": {"messages": [message]}})

        # Check results
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "Vladimir" in tool_message.tool_call_result.result
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
        assert "Diana" in tool_message.tool_call_result.result and "Metropolis" in tool_message.tool_call_result.result
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_document_processor_in_pipeline(self):
        component = DocumentProcessor()
        tool = Tool.from_component(
            component=component,
            name="document_processor",
            description="A tool that concatenates the content of multiple documents"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool], convert_result_to_json_string=True))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(
            text="I have two documents. First one says 'Hello world' and second one says 'Goodbye world'. Can you concatenate them?"
        )

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        result = json.loads(tool_message.tool_call_result.result)
        assert "concatenated" in result
        assert "Hello world" in result["concatenated"]
        assert "Goodbye world" in result["concatenated"]
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_lost_in_middle_ranker_in_pipeline(self):
        from haystack.components.rankers import LostInTheMiddleRanker

        component = LostInTheMiddleRanker()
        tool = Tool.from_component(
            component=component,
            name="lost_in_middle_ranker",
            description="A tool that ranks documents using the Lost in the Middle algorithm and returns top k results"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(
            text="I have three documents with content: 'First doc', 'Middle doc', and 'Last doc'. Rank them top_k=2. Set only content field of the document only. Do not set id, meta, score, embedding, sparse_embedding, dataframe, blob fields."
        )

        result = pipeline.run({"llm": {"messages": [message]}})

        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1
        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)




## Integration tests
class TestToolComponentInPipelineWithAnthropic:

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
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
        pipeline.add_component("llm", AnthropicChatGenerator(model="claude-3-5-sonnet-20240620", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))

        # Connect components
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Vladimir")

        # Run pipeline
        result = pipeline.run({"llm": {"messages": [message]}})

        # Check results
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "Vladimir" in tool_message.tool_call_result.result
        assert not tool_message.tool_call_result.error

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    def test_user_greeter_in_pipeline(self):
        component = UserGreeter()
        tool = Tool.from_component(
            component=component,
            name="user_greeter",
            description="A tool that greets users with their name and age"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", AnthropicChatGenerator(model="claude-3-5-sonnet-20240620", tools=[tool]))
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

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    def test_list_processor_in_pipeline(self):
        component = ListProcessor()
        tool = Tool.from_component(
            component=component,
            name="list_processor",
            description="A tool that concatenates a list of strings"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", AnthropicChatGenerator(model="claude-3-5-sonnet-20240620", tools=[tool]))
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

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    def test_product_processor_in_pipeline(self):
        component = ProductProcessor()
        tool = Tool.from_component(
            component=component,
            name="product_processor",
            description="A tool that creates a description for a product with its name and price"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", AnthropicChatGenerator(model="claude-3-5-sonnet-20240620", tools=[tool]))
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

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    def test_person_processor_in_pipeline(self):
        component = PersonProcessor()
        tool = Tool.from_component(
            component=component,
            name="person_processor",
            description="A tool that processes information about a person and their address"
        )

        pipeline = Pipeline()
        pipeline.add_component("llm", AnthropicChatGenerator(model="claude-3-5-sonnet-20240620", tools=[tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[tool]))
        pipeline.connect("llm.replies", "tool_invoker.messages")

        message = ChatMessage.from_user(text="Diana lives at 123 Elm Street in Metropolis")

        result = pipeline.run({"llm": {"messages": [message]}})
        tool_messages = result["tool_invoker"]["tool_messages"]
        assert len(tool_messages) == 1

        tool_message = tool_messages[0]
        assert tool_message.is_from(ChatRole.TOOL)
        assert "Diana" in tool_message.tool_call_result.result and "Metropolis" in tool_message.tool_call_result.result
        assert not tool_message.tool_call_result.error
