import csv
import json
from collections import defaultdict
from typing import List

from tqdm import tqdm

from haystack import Document, Pipeline
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.utils import Secret, ComponentDevice, Device

from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from prompt import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_SYSTEM_HF
from prompt import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_USER_HF

from neo4j import GraphDatabase

def read_documents(file: str) -> List[Document]:
    with open(file, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader, None)  # skip the headers
        documents = []
        for row in reader:
            category = row[0].strip()
            title = row[2].strip()
            text = row[3].strip()
            documents.append(Document(content=text, meta={"category": category, "title": title}))

    return documents


def doc2graph():
    """
    HF_API_TOKEN needs to be set as an environment variable

    :return:
    :rtype:
    """
    from prompt import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_SYSTEM_HF
    from prompt import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_USER_HF

    pipe_local = Pipeline()
    generator = HuggingFaceLocalChatGenerator(
        token=Secret.from_token("HF_API_TOKEN"),
        task="text-generation",
        model="EmergentMethods/Phi-3-mini-4k-instruct-graph",
        # device=ComponentDevice.from_single(Device.mps()),
        device=ComponentDevice.from_single(Device.cpu()),
        generation_kwargs={
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False
        }
    )
    prompt = ChatPromptBuilder()
    pipe_local.add_component("prompt_builder", prompt)
    pipe_local.add_component("generator", generator)
    pipe_local.connect("prompt_builder", "generator")

    # Phi-3-mini-4k-instruct-graph
    messages = [ChatMessage.from_system(ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_SYSTEM_HF),
                ChatMessage.from_user(ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_USER_HF)]

    return pipe_local, messages

def load_data_to_neo4j(nodes, edges):
    # Connect to the Neo4j database
    driver = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "neo4j"))

    with driver.session() as session:
        # Load nodes using MERGE
        for node in nodes:
            session.run(
                "MERGE (n:Node {id: $id}) SET n.name = $name, n.type = $type, n.detailed_type = $detailed_type",
                id=node['id'],
                name=node['name'],
                type=node['type'],
                detailed_type=node['detailed_type']
            )

        # Load edges using MATCH to find existing nodes
        for edge in edges:
            session.run(
                """
                MATCH (a:Node {id: $source}), (b:Node {id: $target})
                MERGE (a)-[r:RELATIONSHIP {description: $description}]->(b)
                """,
                source=edge['source'],
                target=edge['target'],
                rel_type=edge['description'],
                description=edge['description']
            )

        driver.close()

def extract():
    documents = read_documents("bbc-news-data.csv")
    docs = [doc for doc in documents if doc.meta['category'] == 'business' ] # only business category

    # get an information extraction pipeline
    pipeline, messages = doc2graph()
    extracted_graph = defaultdict(list)

    for d in tqdm(docs[0:10], desc="Processing documents"):
        data = {"prompt_builder": {"template_variables": {"input_text": d.content}, "template": messages}}
        result = pipeline.run(data=data)
        result_json = json.loads(result['generator']['replies'][0].content.strip())
        extracted_graph["nodes"].extend(result_json['nodes'])
        extracted_graph["edges"].extend(result_json['edges'])

