from dataclasses import dataclass
from typing import List, Tuple

from haystack import Pipeline, component
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document
from haystack.components.extractors import NamedEntityExtractor
from neo4j import GraphDatabase

@dataclass
class Entity:
    e_type: str
    surface_string: str

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



def main():

    ner_extractor = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")
    splitter = DocumentSplitter(split_overlap=0, split_by="sentence")
    graph_builder = BuildGraph()

    """
    https://hub.docker.com/_/neo4j
    
    docker run --publish=7474:7474 --publish=7687:7687 --volume=$HOME/neo4j/data:/data neo4j
    
    This binds two ports (7474 and 7687) for HTTP and Bolt access to the Neo4j API. A volume is bound to /data to allow the database to be persisted outside the container.
    By default, this requires you to login with neo4j/neo4j and change the password. You can, for development purposes, disable authentication by passing --env=NEO4J_AUTH=none to docker run.
    """

    # ToDo:
    #   connect chunks where the same entity is mentioned
    #   embed the chunks
    #   index nodes and edges in a graph database, neo4j or dgl

    pipeline = Pipeline()
    pipeline.add_component("splitter", splitter)
    pipeline.add_component("ner_extractor", ner_extractor)
    pipeline.add_component("entity_builder", graph_builder)

    pipeline.connect("splitter", "ner_extractor")
    pipeline.connect("ner_extractor", "entity_builder")

    docs = [Document(content="My name is Clara and I live in Berkeley, California."),
            Document(content="I'm Merlin, the happy pig!"),
            Document(content="New York State is home to the Empire State Building.")
            ]

    neo4j = Neo4jHandler(user='neo4j', password='xpto1234', uri='neo4j://localhost/:7687')

if __name__ == '__main__':
    main()
