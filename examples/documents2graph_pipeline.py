from dataclasses import dataclass
from typing import List, Tuple

from torch.cuda import graph

from haystack import Pipeline, component
from haystack.components.preprocessors import NLTKDocumentSplitter
from haystack.dataclasses import Document
from haystack.components.extractors import NamedEntityExtractor

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


def main():

    ner_extractor = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")
    splitter = NLTKDocumentSplitter(split_overlap=0, split_by="sentence")
    graph_builder = BuildGraph()

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


if __name__ == '__main__':
    main()
