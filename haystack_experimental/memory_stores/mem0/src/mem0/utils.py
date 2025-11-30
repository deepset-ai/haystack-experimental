# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

SEMANTIC_MEMORY_EXTRACTION_PROMPT = """
You are extracting semantic memories from a conversation chain. Semantic memories are general
preferences or personality traits or facts about the user that occurred during the conversation.
while extracting facts from a chain of conversations, you should make sure that assistant
messages are turned into factsonly if the user has liked the assistant's response.


WHAT TO EXTRACT:
- General preferences or personality traits
- Facts about the user


WHAT NOT TO EXTRACT:
- Vague statements without concrete facts
- Specific events or actions that occurred during the conversation
- Contextual details from the conversation flow
- Problem-solution pairs that emerged during the conversation
- Questions without answers
- Speculative or hypothetical statements


OUTPUT FORMAT:
You MUST return a JSON object with ONLY a "facts" key containing an array of strings.
Each fact should be a string that contains fact about the user.

{
    "facts": [
        "User likes to listen to Russian pop music"
    ]
}

IMPORTANT:
- Return JSON with ONLY the "facts" key
- Each fact must be a STRING, not an object
- If no semantic memories can be extracted, return {"facts": []}

EXAMPLES:

Example 1:
Conversation:
USER: I like to listen to Russian pop music
ASSISTANT: That's interesting! What other types of music do you like?
USER: I also like to listen to jazz music
ASSISTANT: Jazz is great! Do you have a favorite jazz artist?
USER: Yes, I like to listen to John Coltrane

Extracted semantic memory:
{
    "facts": [
        "User likes to listen to Russian pop music"
        "User likes to listen to jazz music"
        "User likes to listen to John Coltrane"
    ]
}

Example 2:
Conversation:
USER: I live in Florence Italy and I love mountains.
ASSISTANT: That sounds exciting! Which countries are you planning to visit?
USER: I'm thinking of France, Italy, and Spain. I've already booked flights to Paris for next month.
ASSISTANT: Great itinerary! Paris is beautiful. Have you been to Europe before?
USER: Yes, I visited London last year for a conference. It was my first time in Europe.

Extracted semantic memory:
{
    "facts": [
        "User lives in Florence Italy and loves mountains"
    ]
}

Example 3 (Empty Output):
Conversation:
USER: Hello, how are you?
ASSISTANT: I'm doing well, thank you! How can I help you today?
USER: Just checking in, nothing specific.

Extracted semantic memory:
{
    "facts": []
}


"""

semantic_memory_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano-2025-04-14",
            "temperature": 0.2,
            "max_tokens": 2000,
        },
    },
    "custom_fact_extraction_prompt": SEMANTIC_MEMORY_EXTRACTION_PROMPT,
}


SEMANTIC_MEMORY_PROMPT_WITH_METADATA = """
You are extracting semantic memories from a conversation chain. Semantic memories are general
preferences, personality traits, or stable facts about the user that occurred during the conversation.

You MUST decide, for each extracted memory:
- What the fact is (text)
- Who it comes from (source)
- How confident you are (confidence score)

Assistant messages may only produce a fact if the user clearly LIKES, AGREES WITH, or CONFIRMS
what the assistant said (explicitly or implicitly). Otherwise, DO NOT create a fact from assistant text.


WHAT TO EXTRACT:
- General preferences (e.g., likes/dislikes, style, habits)
- Personality traits (e.g., “is very detail-oriented”)
- Stable facts about the user (e.g., where they live, job, long-term goals)


WHAT NOT TO EXTRACT:
- Vague statements without concrete facts
- One-off events or actions during the conversation
- Contextual details from the conversation flow
- Problem-solution pairs from this specific chat
- Questions without answers
- Speculative or hypothetical statements


METADATA REQUIREMENTS:
For each fact, you must output a STRING that contains a valid JSON object with this shape:

{
  "text": "<the fact about the user>",
  "source": "user" | "assistant",
  "confidence": <number between 0.0 and 1.0>,
  "tags": ["preference" | "trait" | "biography" | "meta", ...]
}

Guidelines:
- "source" = "user" if the user said or clearly confirmed the fact.
- "source" = "assistant" only if the assistant proposed it AND the user clearly agreed, liked it,
  or behaved consistently with it.
- Default confidence:
  - 0.9 for strong, explicit user statements (e.g., “I love X”, “I live in Y”).
  - 0.7 for weaker/implicit statements (e.g., “I usually…”, “I kind of like…”).
  - 0.5 for cautiously inferred assistant-based facts the user agreed with.
- Use tags to roughly classify the fact (e.g., ["preference", "music"], ["biography"], ["trait"]).


OUTPUT FORMAT:
You MUST return a JSON object with ONLY a "facts" key containing an array of strings.
Each string must itself be a JSON object (as text) containing the fields described above.

Valid final output example:

{
  "facts": [
    "{\\"text\\": \\"User likes to listen to Russian pop music\\", \\"source\\": \\"user\\",
    \\"confidence\\": 0.9, \\"tags\\": [\\"preference\\", \\"music\\"]}"
  ]
}

IMPORTANT:
- Return JSON with ONLY the "facts" key.
- Each item in "facts" MUST be:
  - A STRING
  - That STRING must contain a valid JSON object with keys: "text", "source", "confidence", "tags".
- If no semantic memories can be extracted, return {"facts": []}.


EXAMPLES:

Example 1:
Conversation:
USER: I like to listen to Russian pop music
ASSISTANT: That's interesting! What other types of music do you like?
USER: I also like to listen to jazz music
ASSISTANT: Jazz is great! Do you have a favorite jazz artist?
USER: Yes, I like to listen to John Coltrane

Extracted semantic memory:
{
  "facts": [
    "{\\"text\\": \\"User likes to listen to Russian pop music\\", \\"source\\": \\"user\\",
    \\"confidence\\": 0.9, \\"tags\\": [\\"preference\\", \\"music\\"]}",
    "{\\"text\\": \\"User likes to listen to jazz music\\", \\"source\\": \\"user\\",
    \\"confidence\\": 0.9, \\"tags\\": [\\"preference\\", \\"music\\"]}",
    "{\\"text\\": \\"User likes to listen to John Coltrane\\", \\"source\\": \\"user\\",
    \\"confidence\\": 0.9, \\"tags\\": [\\"preference\\", \\"artist\\"]}"
  ]
}

Example 2:
Conversation:
USER: I live in Florence Italy and I love mountains.
ASSISTANT: That sounds exciting! Which countries are you planning to visit?
USER: I'm thinking of France, Italy, and Spain. I've already booked flights to Paris for next month.
ASSISTANT: Great itinerary! Paris is beautiful. Have you been to Europe before?
USER: Yes, I visited London last year for a conference. It was my first time in Europe.

Extracted semantic memory:
{
  "facts": [
    "{\\"text\\": \\"User lives in Florence Italy and loves mountains\\",
    \\"source\\": \\"user\\", \\"confidence\\": 0.9, \\"tags\\":
    [\\"biography\\", \\"preference\\"]}"
  ]
}

Example 3 (Empty Output):
Conversation:
USER: Hello, how are you?
ASSISTANT: I'm doing well, thank you! How can I help you today?
USER: Just checking in, nothing specific.

Extracted semantic memory:
{
  "facts": []
}
"""
