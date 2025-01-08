from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_experimental.components.generators.chat import OpenAIChatGenerator
from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.dataclasses import ChatMessage
from haystack_experimental.dataclasses.tool import ToolComponent

ds = InMemoryDocumentStore()

ds.write_documents(
    [
        Document(content="Hello, how can I help you?", meta={"id": "1"}),
        Document(content="The capital of France is Paris", meta={"id": "2"}),
        Document(content="The capital of Germany is Berlin", meta={"id": "3"}),
        Document(content="The capital of Italy is Rome", meta={"id": "4"}),
        Document(content="The capital of Spain is Madrid", meta={"id": "5"}),
        Document(content="Italy is a country in Europe", meta={"id": "6"}),
        Document(content="The capital of Greece is Athens", meta={"id": "7"}),
        Document(content="The capital of Turkey is Ankara", meta={"id": "8"}),
        Document(content="The capital of Bulgaria is Sofia", meta={"id": "9"}),
        Document(content="The capital of Romania is Bucharest", meta={"id": "10"}),
    ]
)


bm25_retriever = InMemoryBM25Retriever(document_store=ds)
retriever_tool = ToolComponent(bm25_retriever, "retriever", "Useful for searching documents based on a query")

pipeline = Pipeline()
pipeline.add_component("OpenAIChatGenerator", OpenAIChatGenerator(tools=[retriever_tool]))
pipeline.add_component("ToolInvoker", ToolInvoker(tools=[retriever_tool]))
pipeline.connect("OpenAIChatGenerator.replies", "ToolInvoker.messages")


print(
    pipeline.run(
        {"OpenAIChatGenerator": {"messages": [ChatMessage.from_user("Retrieve the top 3 documents similar to: Italy")]}}
    )
)
# {'ToolInvoker': {'tool_messages': [ChatMessage(_role=<ChatRole.TOOL: 'tool'>, _content=[ToolCallResult
# (result="{'documents': [Document(id=6, content: 'Italy is a country in Europe', score: 1.8765898222989232),
# Document(id=4, content: 'The capital of Italy is Rome', score: 1.7922635575696158),
# Document(id=1, content: 'Hello, how can I help you?', score: 0.9260028380776347)]}",
# origin=ToolCall(tool_name='retriever', arguments={'query': 'Italy', 'top_k': 3}, id='call_zy0HOJwgjnAe2LZ9ep2UKzgO'),
# error=False)], _meta={})]}}


yaml = pipeline.dumps()
print(yaml)
# components:
#  OpenAIChatGenerator:
#    init_parameters:
#      api_base_url: null
#      api_key:
#        env_vars:
#        - OPENAI_API_KEY
#        strict: true
#        type: env_var
#      generation_kwargs: {}
#      max_retries: null
#      model: gpt-4o-mini
#      organization: null
#      streaming_callback: null
#      timeout: null
#      tools:
#      - data:
#          component:
#            init_parameters:
#              document_store:
#                init_parameters:
#                  bm25_algorithm: BM25L
#                  bm25_parameters: &id001 {}
#                  bm25_tokenization_regex: (?u)\b\w\w+\b
#                  embedding_similarity_function: dot_product
#                  index: c4297f1f-75e2-4f7a-9db9-0e578007eb1c
#                type: haystack.document_stores.in_memory.document_store.InMemoryDocumentStore
#              filter_policy: replace
#              filters: null
#              scale_score: false
#              top_k: 10
#            type: haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever
#          description: Useful for searching documents based on a query
#          name: retriever
#          parameters: &id002
#            properties:
#              filters:
#                description: A dictionary with filters to narrow down the search space
#                  when retrieving documents.
#                type: string
#              query:
#                description: The query string for the Retriever.
#                type: string
#              scale_score:
#                description: 'When `True`, scales the score of retrieved documents
#                  to a range of 0 to 1, where 1 means extremely relevant.
#
#                  When `False`, uses raw similarity scores.'
#                type: boolean
#              top_k:
#                description: The maximum number of documents to return.
#                type: integer
#            required:
#            - query
#            type: object
#        type: haystack_experimental.dataclasses.tool.ToolComponent
#      tools_strict: false
#    type: haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator
#  ToolInvoker:
#    init_parameters:
#      convert_result_to_json_string: false
#      raise_on_failure: true
#      tools:
#      - data:
#          component:
#            init_parameters:
#              document_store:
#                init_parameters:
#                  bm25_algorithm: BM25L
#                  bm25_parameters: *id001
#                  bm25_tokenization_regex: (?u)\b\w\w+\b
#                  embedding_similarity_function: dot_product
#                  index: c4297f1f-75e2-4f7a-9db9-0e578007eb1c
#                type: haystack.document_stores.in_memory.document_store.InMemoryDocumentStore
#              filter_policy: replace
#              filters: null
#              scale_score: false
#              top_k: 10
#            type: haystack.components.retrievers.in_memory.bm25_retriever.InMemoryBM25Retriever
#          description: Useful for searching documents based on a query
#          name: retriever
#          parameters: *id002
#        type: haystack_experimental.dataclasses.tool.ToolComponent
#    type: haystack_experimental.components.tools.tool_invoker.ToolInvoker
# connections:
# - receiver: ToolInvoker.messages
#  sender: OpenAIChatGenerator.replies
# max_runs_per_component: 100
# metadata: {}


new_pipeline = Pipeline.loads(yaml)
print(
    new_pipeline.run(
        {"OpenAIChatGenerator": {"messages": [ChatMessage.from_user("Retrieve the top 3 documents similar to: Italy")]}}
    )
)
# {'ToolInvoker': {'tool_messages': [ChatMessage(_role=<ChatRole.TOOL: 'tool'>, _content=[ToolCallResult
# (result="{'documents': [Document(id=6, content: 'Italy is a country in Europe', score: 1.8765898222989232),
# Document(id=4, content: 'The capital of Italy is Rome', score: 1.7922635575696158),
# Document(id=1, content: 'Hello, how can I help you?', score: 0.9260028380776347)]}",
# origin=ToolCall(tool_name='retriever', arguments={'query': 'Italy', 'top_k': 3}, id='call_zy0HOJwgjnAe2LZ9ep2UKzgO'),
# error=False)], _meta={})]}}
