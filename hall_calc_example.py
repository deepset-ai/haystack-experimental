import os

from haystack.dataclasses import ChatMessage
from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator

# os.environ["OPENAI_API_KEY"] = ""

llm = OpenAIChatGenerator(model_name="gpt-4o")

# Closed Book Example
closed_book_result = llm.run(messages=[ChatMessage(role="user", content="Who won the 2019 Nobel Prize in Physics?")])
print(f"Decision: {closed_book_result['replies'][0].meta['hallucination_decision']}")
print(f"Risk bound: {closed_book_result['replies'][0].meta['hallucination_risk']:.3f}")
print(f"Risk bound: {closed_book_result['replies'][0].meta['hallucination_rationale']:.3f}")
print(f"Answer:\n{closed_book_result['replies'][0].text}")
print("---\n")

# Evidence-based Example
rag_result = llm.run(
    messages=[
        ChatMessage(
            role="user",
            content="Who won the 2019 Nobel Prize in Physics?"
        )
    ]
)
print(f"Decision: {rag_result['replies'][0].meta['hallucination_decision']}")
print(f"Risk bound: {rag_result['replies'][0].meta['hallucination_risk']:.3f}")
print(f"Rationale: {rag_result['replies'][0].meta['hallucination_rationale']}")
print(f"Answer:\n{closed_book_result['replies'][0].text}")
print("---")
