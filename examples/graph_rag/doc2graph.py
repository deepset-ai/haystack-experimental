import csv
import json
from typing import List, Tuple, Dict, Optional

from neo4j import GraphDatabase
from tqdm import tqdm

from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from prompt import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_SYSTEM_HF
from prompt import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_USER_HF


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


@component
class ExtractGraph:

    def __init__(self):
        self.messages = [
            ChatMessage.from_system(ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_SYSTEM_HF),
            ChatMessage.from_user(ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT_USER_HF)
        ]

        self.generator = OpenAIChatGenerator(model="gpt-4o-mini")
        self.prompt = ChatPromptBuilder()

    @component.output_types(extracted_graphs=List[Dict])
    def run(self, documents: List[Document]) -> Dict[str, List[Dict]]:
        extracted_graphs = []
        for d in tqdm(documents, desc="Processing documents"):
            filled_prompt = self.prompt.run(template_variables={"input_text": d.content}, template=self.messages)
            result = self.generator.run(messages=filled_prompt['prompt'])
            result_json = json.loads(result['replies'][0].content.strip())
            extracted_graphs.append(result_json)

        return {'extracted_graphs': extracted_graphs}


@component
class Neo4jLoader:

    def __init__(self, url: str = "bolt://0.0.0.0:7687", user: Optional[str] = None, password: Optional[str] = None):
        if user is None or password is None:
            self.driver = GraphDatabase.driver(url)
        else:
            self.driver = GraphDatabase.driver(url, auth=(user, password))

    @staticmethod
    def set_global_ids(extracted_graph: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Considering the extracted graph is a list of JSON where each node as local id, we need to convert the local
        ids to global ids. Similarly, we also need to update the edges with the global ids.
        """
        nodes = []
        edges = []
        node_mapping = {}
        for graph in extracted_graph:
            for node in graph['nodes']:
                if node['name'] not in node_mapping:
                    node_mapping[node['name']] = len(nodes)
                    nodes.append(
                        {'id': node_mapping[node['name']],
                         'name': node['name'],
                         'type': node['type'],
                         'detailed_type': node['detailed_type'],
                         }
                    )

            for edge in graph['edges']:
                new_edge = {}
                for node in graph['nodes']:
                    if node['id'] == edge['from']:
                        new_edge['source'] = node_mapping[node['name']]
                        break

                for node in graph['nodes']:
                    if node['id'] == edge['to']:
                        new_edge['target'] = node_mapping[node['name']]
                        break

                new_edge['description'] = edge['label']
                edges.append(new_edge)

        return edges, nodes

    @component.output_types(extracted_graphs=List[Dict])
    def run(self, extracted_graphs: List[Dict]) -> Dict[str, List[Dict]]:
        edges, nodes = self.set_global_ids(extracted_graphs)

        with self.driver.session() as session:
            for node in nodes:
                session.run(
                    "MERGE (n:Node {id: $id}) SET n.name = $name, n.type = $type, n.detailed_type = $detailed_type",
                    id=node['id'],
                    name=node['name'],
                    type=node['type'],
                    detailed_type=node['detailed_type']
                )

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

            self.driver.close()

        return {'nodes': nodes, "edges": edges}


def extract_graph():
    documents = read_documents("bbc-news-data.csv")
    docs = [doc for doc in documents if doc.meta['category'] == 'business']  # only business category

    extractor = ExtractGraph()
    loader = Neo4jLoader(url="bolt://localhost:7687", user="neo4j", password="password")

    pipeline = Pipeline()
    pipeline.add_component(instance=extractor, name="extractor")
    pipeline.add_component(instance=loader, name="loader")
    pipeline.connect("extractor.extracted_graphs", "loader.extracted_graphs")

    result = pipeline.run(data={'extractor': {'documents': docs[10:23]}}, include_outputs_from={'loader'})
    # ToDo: save results to file

    # ToDo:
    # 1. enhance the prompt to include a summary of the named entities and relationships
    # 2. Graph Communities detection
    # 3. Generate Communities Summaries
