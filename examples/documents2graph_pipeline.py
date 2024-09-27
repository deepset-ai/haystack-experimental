import time

import csv
from dataclasses import dataclass
from typing import List, Tuple, Set

from haystack import Pipeline, component
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import Document
from haystack.components.extractors import NamedEntityExtractor
from neo4j import GraphDatabase

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

import requests
import spacy
import claucy

nlp = spacy.load("en_core_web_sm")
claucy.add_to_pipe(nlp)

@dataclass
class Entity:
    e_type: str
    surface_string: str

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.e_type == other.e_type and self.surface_string == other.surface_string

    def __hash__(self):
        return hash((self.e_type, self.surface_string))

@dataclass
class Relationship:
    """
    A relationship between two entities.

    The relationship is a sentence, where both entities are mentioned.
    """
    ent1: Entity
    ent2: Entity
    relationship: str

@component
class BuildGraph:

    def __init__(self):
        pass

    @staticmethod
    def extract_entities(entity, doc_text: str):
        e_type = entity.entity
        start = entity.start
        end = entity.end
        surface_string = doc_text[start:end]
        return Entity(e_type=e_type, surface_string=surface_string)

    @component.output_types(documents=Tuple[List[Entity], List[Relationship]])
    def run(self, documents: List[Document]):
        all_entities = []
        all_relationships = []
        for doc in documents:
            entities_sorted_by_occurrence = sorted(doc.meta["named_entities"], key=lambda entity: entity.start)
            entities = [self.extract_entities(entity, doc.content) for entity in entities_sorted_by_occurrence]
            consecutive_pairs = [(entities[i], entities[i + 1]) for i in range(len(entities) - 1)]
            relationships = [Relationship(ent1=ent1, ent2=ent2, relationship=doc.id) for ent1, ent2 in consecutive_pairs]
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        return {"entities": all_entities, "relationships": all_relationships}


class Neo4jHandler:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node(self, label, properties):
        with self.driver.session() as session:
            session.execute_write(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} {{ {', '.join([f'{k}: ${k}' for k in properties.keys()])} }})"
        tx.run(query, **properties)

    def create_relationship(self, label1, properties1, label2, properties2, rel_type, rel_properties):
        with self.driver.session() as session:
            session.execute_write(self._create_relationship, label1, properties1, label2, properties2, rel_type, rel_properties)

    @staticmethod
    def _create_relationship(tx, label1, properties1, label2, properties2, rel_type, rel_properties):
        query = (
            f"""MATCH (a:{label1} {{ {', '.join([f"{k}: ${'a_' + k}" for k in properties1.keys()])} }}), """
            f"""(b:{label2} {{ {', '.join([f"{k}: ${'b_' + k}" for k in properties2.keys()])} }}) """
            f"""CREATE (a)-[r:{rel_type} {{ {', '.join([f"{k}: ${'r_' + k}" for k in rel_properties.keys()])} }}]->(b)"""
        )
        params = {**{f'a_{k}': v for k, v in properties1.items()}, **{f'b_{k}': v for k, v in properties2.items()}, **{f'r_{k}': v for k, v in rel_properties.items()}}
        tx.run(query, **params)

    def clean_database(self):
        with self.driver.session() as session:
            session.execute_write(self._clean_database)

    @staticmethod
    def _clean_database(tx):
        tx.run("MATCH (n) DETACH DELETE n")


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

def entity_normalizer(entities: List[Entity]) -> Set[Entity]:
    """
    Simple approach, same surface string -> same entity
    """
    seen = set()
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
        else:
            print(f"Duplicated entity: {entity}")

    return seen


def extract_relationship_clausie(documents: List[Document]) -> List[Relationship]:
    # spacy NLP process a batch of sentences
    texts = [doc.content for doc in documents]
    spacy_docs = list(nlp.pipe(texts))

    for doc in spacy_docs:
        if not doc._.clauses:
            continue
        print(doc)
        for ent in doc.ents:
            print(ent.text, ent.label_)
        print(doc._.clauses[0].to_propositions(as_text=False, inflect=None))
        print()
        print("-" * 100)



def measure_time():
    """Measure the time it takes to run a given function. Can be used as a decorator."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"Function {func.__name__} took {time.time() - start:.2f} seconds.")
            return result
        return wrapper
    return decorator


@measure_time()
def extract_relationships(data):
    url = 'http://localhost:11434/api/generate'
    response = requests.post(url, json=data)
    return response

def extract_relationships_llm(documents: List[Document]):

    from .prompt import ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
    prompt = ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
    entity_types = ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

    for doc in docs:
        data = {
            # "model": "llama3.2",
            "model": "phi",
            "prompt": prompt.format(language='English', input_text=doc.content, entity_types=entity_types),
            "stream": False
        }
        response = extract_relationships(data)
        resp_json = response.json()
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(resp_json)
        print("\n\n")
        print("source text:")
        print(doc.content)
        print("----------------")
        print(resp_json['response'])
        print("=" * 80)
        print("\n\n")




def main():

    """
    docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/neo4j/data:/data neo4j
    """

    # ToDo:
    # 1. Documents to Named Entities and Relationships

    # https://huggingface.co/EmergentMethods/Phi-3-mini-4k-instruct-graph
    # LLM to do information extraction

    # 2. Named Entities and Relationships are summarized by the LLM into descriptive text blocks for each element.
    # 3. Graph Communities detection
    # 4. Graph Communities to Community Summaries

    ner_extractor = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")
    splitter = DocumentSplitter(split_overlap=0, split_by="sentence", split_length=1)
    graph_builder = BuildGraph()
    doc_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(doc_store, policy=DuplicatePolicy.OVERWRITE)

    pipeline = Pipeline()
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("doc_writer", doc_writer)
    pipeline.add_component("ner_extractor", ner_extractor)
    pipeline.add_component("entity_builder", graph_builder)

    # pipeline.connect("splitter", "ner_extractor")
    pipeline.connect("splitter", "doc_writer")
    # pipeline.connect("ner_extractor", "entity_builder")

    docs = read_documents("bbc-news-data.csv")

    """
    'sport': 511,
    'business': 510,
    'politics': 417,
    'tech': 401,
    'entertainment': 386
    """

    docs = [doc for doc in docs if doc.meta['category'] == 'business' ]
    result = pipeline.run(data={'documents': docs})

    entities_normalised = entity_normalizer(result["entity_builder"]["entities"])

    neo4j = Neo4jHandler(user='neo4j', password='xpto1234', uri='neo4j://localhost/:7687')
    neo4j.clean_database()

    print("Creating nodes")
    for entity in entities_normalised:
        neo4j.create_node(label=entity.e_type, properties={"surface_string": entity.surface_string})

    print("Creating edges")
    for relationship in result["entity_builder"]["relationships"]:
        neo4j.create_relationship(
            label1=relationship.ent1.e_type,
            properties1={"surface_string": relationship.ent1.surface_string},
            label2=relationship.ent2.e_type,
            properties2={"surface_string": relationship.ent2.surface_string},
            rel_type="RELATED",
            rel_properties={"relationship": relationship.relationship}
        )



if __name__ == '__main__':
    main()
