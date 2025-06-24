# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import component
from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters.html import HTMLToDocument
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ByteStream
from haystack.dataclasses import ChatMessage, Document, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import Toolset
from haystack.tools import tool, Tool
from typing import Optional, List, Dict

from haystack_experimental.core.pipeline import Pipeline
from haystack_experimental.dataclasses.breakpoints import Breakpoint, ToolBreakpoint

document_store = InMemoryDocumentStore() # create a document store or an SQL database

@component
class MockLinkContentFetcher:

    @component.output_types(streams=List[ByteStream])
    def run(self, urls: List[str]) -> Dict[str, List[ByteStream]]:

        mock_html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deepset - About Our Team</title>
        </head>
        <body>
            <h1>About Deepset</h1>
            <p>Deepset is a company focused on natural language processing and AI.</p>
            
            <h2>Our Leadership Team</h2>
            <div class="team-member">
                <h3>Malte Pietsch</h3>
                <p>Malte Pietsch is the CEO and co-founder of Deepset. He has extensive experience in machine learning and natural language processing.</p>
                <p>Job Title: Chief Executive Officer</p>
            </div>
            
            <div class="team-member">
                <h3>Milos Rusic</h3>
                <p>Milos Rusic is the CTO and co-founder of Deepset. He specializes in building scalable AI systems and has worked on various NLP projects.</p>
                <p>Job Title: Chief Technology Officer</p>
            </div>
            
            <h2>Our Mission</h2>
            <p>Deepset aims to make natural language processing accessible to developers and businesses worldwide through open-source tools and enterprise solutions.</p>
        </body>
        </html>
        """

        bytestream = ByteStream(
            data=mock_html_content.encode('utf-8'),
            mime_type="text/html",
            meta={"url": urls[0] if urls else "https://en.wikipedia.org/wiki/Deepset"}
        )
        
        return {"streams": [bytestream]}

@tool
def add_database_tool(name: str, surname: str, job_title: Optional[str], other: Optional[str]):
    """Use this tool to add names to the database with information about them"""
    document_store.write_documents(
        [Document(content=name + " " + surname + " " + (job_title or ""), meta={"other":other})]
    )
    return

def create_agent():
    # Create a real OpenAIChatGenerator and mock its run method
    generator = OpenAIChatGenerator()

    call_count = 0
    def mock_run(messages, tools=None, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "I'll extract the information about the people mentioned in the context.",
                        tool_calls=[
                            ToolCall(
                                tool_name="add_database_tool",
                                arguments={
                                    "name": "Malte",
                                    "surname": "Pietsch", 
                                    "job_title": "Chief Executive Officer",
                                    "other": "CEO and co-founder of Deepset with extensive experience in machine learning and natural language processing"
                                }
                            ),
                            ToolCall(
                                tool_name="add_database_tool",
                                arguments={
                                    "name": "Milos",
                                    "surname": "Rusic",
                                    "job_title": "Chief Technology Officer", 
                                    "other": "CTO and co-founder of Deepset specializing in building scalable AI systems and NLP projects"
                                }
                            )
                        ]
                    )
                ]
            }
        else:
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "I have successfully extracted and stored information about the following people:\n\n"
                        "1. **Malte Pietsch** - Chief Executive Officer\n"
                        "   - CEO and co-founder of Deepset\n"
                        "   - Extensive experience in machine learning and natural language processing\n\n"
                        "2. **Milos Rusic** - Chief Technology Officer\n"
                        "   - CTO and co-founder of Deepset\n"
                        "   - Specializes in building scalable AI systems and NLP projects\n\n"
                        "Both individuals have been added to the knowledge base with their respective information."
                    )
                ]
            }
    
    generator.run = mock_run
    
    database_assistant = Agent(
        chat_generator=generator,
        tools=[add_database_tool],
        system_prompt="""
        You are a database assistant.
        Your task is to extract the names of people mentioned in the given context and add them to a knowledge base, 
        along with additional relevant information about them that can be extracted from the context.
        Do not use you own knowledge, stay grounded to the given context.
        Do not ask the user for confirmation. Instead, automatically update the knowledge base and return a brief 
        summary of the people added, including the information stored for each.
        """,
        exit_conditions=["text"],
        max_agent_steps=100,
        raise_on_tool_invocation_failure=False
    )

    extraction_agent = Pipeline()
    extraction_agent.add_component("fetcher", MockLinkContentFetcher())
    extraction_agent.add_component("converter", HTMLToDocument())
    extraction_agent.add_component("builder", ChatPromptBuilder(
        template=[ChatMessage.from_user("""
        {% for doc in docs %}
        {{ doc.content|default|truncate(25000) }}
        {% endfor %}
        """)],
        required_variables=["docs"]
    ))

    extraction_agent.add_component("database_agent", database_assistant)
    extraction_agent.connect("fetcher.streams", "converter.sources")
    extraction_agent.connect("converter.documents", "builder.docs")
    extraction_agent.connect("builder", "database_agent")

    return extraction_agent

def test_breakpoints_agent_in_pipeline():

    extraction_agent = create_agent()

    # define breakpoints
    converter_breakpoint = Breakpoint("converter", 0)
    agent_generator_breakpoint = Breakpoint("chat_generator", 0)
    agent_tool_breakpoint = ToolBreakpoint("tool_invoker", 0, "add_database_tool")

    agent_output = extraction_agent.run(
        data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}},
        breakpoints={converter_breakpoint, }
    )


    print(agent_output["database_agent"]["messages"][-1].text)
