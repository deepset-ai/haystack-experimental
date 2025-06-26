# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict

from haystack import component

from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ByteStream
from haystack.dataclasses import ChatMessage, Document, ToolCall
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import tool

from haystack_experimental.components.agents import Agent
from haystack_experimental.core.errors import PipelineBreakpointException
from haystack_experimental.core.pipeline import Pipeline
from haystack_experimental.core.pipeline.breakpoint import load_state
from haystack_experimental.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint

document_store = InMemoryDocumentStore()

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

@component
class MockHTMLToDocument:

    @component.output_types(documents=List[Document])
    def run(self, sources: List[ByteStream]) -> Dict[str, List[Document]]:
        """Mock HTML to Document converter that extracts text content from HTML ByteStreams."""
        
        documents = []
        for source in sources:
            # Extract the HTML content from the ByteStream
            html_content = source.data.decode('utf-8')
            
            # Simple text extraction - remove HTML tags and extract meaningful content
            # This is a simplified version that extracts the main content
            import re
            
            # Remove HTML tags
            text_content = re.sub(r'<[^>]+>', ' ', html_content)
            # Remove extra whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # Create a Document with the extracted text
            document = Document(
                content=text_content,
                meta={
                    "url": source.meta.get("url", "unknown"),
                    "mime_type": source.mime_type,
                    "source_type": "html"
                }
            )
            documents.append(document)
        
        return {"documents": documents}

@tool
def add_database_tool(name: str, surname: str, job_title: Optional[str], other: Optional[str]):
    document_store.write_documents(
        [Document(content=name + " " + surname + " " + (job_title or ""), meta={"other":other})]
    )

def create_pipeline():
    
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
                                    "other": "CEO and co-founder of Deepset with extensive experience in machine "
                                             "learning and natural language processing"
                                }
                            ),
                            ToolCall(
                                tool_name="add_database_tool",
                                arguments={
                                    "name": "Milos",
                                    "surname": "Rusic",
                                    "job_title": "Chief Technology Officer", 
                                    "other": "CTO and co-founder of Deepset specializing in building scalable "
                                             "AI systems and NLP projects"
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
    extraction_agent.add_component("converter", MockHTMLToDocument())
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

def test_pipeline_breakpoint_and_resume():
        
    extraction_agent = create_pipeline()
        
    with tempfile.TemporaryDirectory() as debug_path:
                
        converter_breakpoint = Breakpoint("fetcher", 0)
        
        # Run pipeline with breakpoint - should raise PipelineBreakpointException
        try:
            extraction_agent.run(
                data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}},
                breakpoints=[converter_breakpoint],
                debug_path=debug_path
            )
        except PipelineBreakpointException as e:
            
            assert e.component == "fetcher"
            assert "state" in e.__dict__
            
            # Find the saved state file
            state_files = list(Path(debug_path).glob("fetcher_*.json"))
            assert len(state_files) > 0
            
            # Load the latest state file
            latest_state_file = str(max(state_files, key=os.path.getctime))
            resume_state = load_state(latest_state_file)
            
            # Resume the pipeline from the saved state
            result = extraction_agent.run(data={}, resume_state=resume_state)
            
            # Verify the pipeline completed successfully
            assert "database_agent" in result
            assert "messages" in result["database_agent"]
            assert len(result["database_agent"]["messages"]) > 0
            
            # Verify the final message contains the expected summary
            final_message = result["database_agent"]["messages"][-1].text
            assert "Malte Pietsch" in final_message
            assert "Milos Rusic" in final_message
            assert "Chief Executive Officer" in final_message
            assert "Chief Technology Officer" in final_message
            
            print("Pipeline breakpoint and resume test completed successfully!")
            print(f"Final message: {final_message}")
        else:
            # If no exception was raised, the test should fail
            assert False, "Expected PipelineBreakpointException was not raised"

def test_agent_breakpoints_in_pipeline_agent_break_on_first_false():

    pipeline_with_agent = create_pipeline()
    agent_generator_breakpoint = Breakpoint("chat_generator", 0)
    agent_tool_breakpoint = ToolBreakpoint("tool_invoker", 0, "add_database_tool")
    agent_breakpoints = AgentBreakpoint(breakpoints={agent_generator_breakpoint, agent_tool_breakpoint})

    with tempfile.TemporaryDirectory() as debug_path:
        agent_output = pipeline_with_agent.run(
            data={"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}},
            breakpoints=[agent_breakpoints],
            debug_path=debug_path,
            break_on_first=False,
        )

        # Verify that state files were generated for both breakpoints
        chat_generator_state_files = list(Path(debug_path).glob("chat_generator_*.json"))
        tool_invoker_state_files = list(Path(debug_path).glob("tool_invoker_*.json"))
        
        # Assert that at least one state file was created for each breakpoint
        assert len(chat_generator_state_files) > 0, f"No chat_generator state files found in {debug_path}"
        assert len(tool_invoker_state_files) > 0, f"No tool_invoker state files found in {debug_path}"
        
        # Verify the pipeline completed successfully
        assert "database_agent" in agent_output
        assert "messages" in agent_output["database_agent"]
        assert len(agent_output["database_agent"]["messages"]) > 0
        
        # Verify the final message contains the expected summary
        final_message = agent_output["database_agent"]["messages"][-1].text
        assert "Malte Pietsch" in final_message
        assert "Milos Rusic" in final_message
        assert "Chief Executive Officer" in final_message
        assert "Chief Technology Officer" in final_message
        
        print("Agent breakpoints in pipeline test completed successfully!")
        print(f"Generated {len(chat_generator_state_files)} chat_generator state files")
        print(f"Generated {len(tool_invoker_state_files)} tool_invoker state files")
        print(f"Final message: {final_message}")