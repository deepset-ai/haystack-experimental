# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any, Dict, List, Optional, Union

from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.core.component import Component, component
from haystack.dataclasses.chat_message import ChatMessage

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = """
You are part of an information system that processes user queries for retrieval.
You have to expand a given query into {{ n_expansions }} queries that are
semantically similar to improve retrieval recall.

Structure:
Follow the structure shown below in examples to generate expanded queries.

Examples:
1.  Query: "climate change effects"
    ["impact of climate change", "consequences of global warming", "effects of environmental changes"]

2.  Query: "machine learning algorithms"
    ["neural networks", "clustering techniques", "supervised learning methods", "deep learning models"]

3.  Query: "open source NLP frameworks"
    ["natural language processing tools", "free nlp libraries", "open-source language processing platforms"]

Guidelines:
- Generate queries that use different words and phrasings
- Include synonyms and related terms
- Maintain the same core meaning and intent
- Make queries that are likely to retrieve relevant information the original might miss
- Focus on variations that would work well with keyword-based search
- Respond in the same language as the input query

Your Task:
Query: "{{ query }}"

You *must* respond only with a compact, inline JSON array of strings, no other text.
Example: ["query1","query2","query3"]"""


@component
class QueryExpander:
    """
    A component that returns a list of semantically similar queries to improve retrieval recall in RAG systems.

    Usage example:

    ```python
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
    from haystack_experimental.components.query import QueryExpander

    expander = QueryExpander(
        generator=OpenAIChatGenerator(model="gpt-4o-mini"),
        n_expansions=3
    )

    result = expander.run(query="green energy sources")
    print(result["queries"])
    # Output: ['alternative query 1', 'alternative query 2', 'alternative query 3', 'green energy sources']
    # Note: 3 additional queries + 1 original = 4 total queries

    # To get exactly 3 total queries:
    expander = QueryExpander(n_expansions=2, include_original_query=True)
    # or
    expander = QueryExpander(n_expansions=3, include_original_query=False)
    ```
    """

    def __init__(
        self,
        *,
        generator: Optional[Union[Component, OpenAIChatGenerator]] = None,
        prompt_template: Optional[str] = None,
        n_expansions: int = 4,
        include_original_query: bool = True,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the QueryExpander component.

        :param generator: The chat generator component to use for query expansion.
            If None, defaults to OpenAIChatGenerator with `gpt-4o-mini`.
        :param prompt_template: Custom PromptBuilder template for query expansion.
            If None, uses DEFAULT_PROMPT_TEMPLATE.
        :param n_expansions: Number of alternative queries to generate (default: 4).
        :param include_original_query: Whether to include the original query
            in the output (default: `True`).
        :param generation_kwargs: Additional generation kwargs to pass to the generator.
        """
        self.n_expansions = n_expansions
        self.include_original_query = include_original_query

        if generator is None:
            self.generator: Union[Component, OpenAIChatGenerator] = OpenAIChatGenerator(
                model="gpt-4o-mini",
                generation_kwargs=generation_kwargs or {"temperature": 0.7},
            )
        else:
            self.generator = generator

        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.prompt_builder = PromptBuilder(template=self.prompt_template)

    @component.output_types(queries=List[str])
    def run(
        self,
        query: str,
        n_expansions: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """
        Expand the input query into multiple semantically similar queries.

        This method follows the approach described in the Haystack blog on query expansion,
        generating alternative queries to improve retrieval recall, especially for
        keyword-based search systems.

        :param query: The original query to expand.
        :param n_expansions: Number of additional queries to generate (not including the original).
            If None, uses the value from initialization. Can be 0 to generate no additional queries.
        :return: Dictionary with "queries" key containing the list of expanded queries.
            If include_original_query=True, the original query will be included in addition
            to the n_expansions alternative queries.
        """
        if not query.strip():
            logger.warning("Empty query provided to QueryExpander")
            return {"queries": [query] if self.include_original_query else []}

        expansion_count = n_expansions if n_expansions is not None else self.n_expansions

        # Optimize for zero expansions - no need to call generator
        if expansion_count == 0:
            return {"queries": [query] if self.include_original_query else []}

        try:
            prompt_result = self.prompt_builder.run(query=query.strip(), n_expansions=expansion_count)

            generator_result = self.generator.run(messages=[ChatMessage.from_user(prompt_result["prompt"])])

            if not generator_result.get("replies") or len(generator_result["replies"]) == 0:
                logger.warning(f"Generator returned no replies for query: '{query}'")
                return {"queries": [query] if self.include_original_query else []}

            expanded_text = generator_result["replies"][0].text.strip()
            expanded_queries = self._parse_expanded_queries(expanded_text)

            # Limit the number of expanded queries to the requested amount
            if len(expanded_queries) > expansion_count:
                expanded_queries = expanded_queries[:expansion_count]

            if self.include_original_query and query not in expanded_queries:
                expanded_queries.append(query)

            return {"queries": expanded_queries}

        except Exception as e:
            # Fallback: return original query to maintain pipeline functionality
            logger.error(f"Failed to expand query '{query}': {str(e)}")
            return {"queries": [query] if self.include_original_query else []}

    def _parse_expanded_queries(self, generator_response: str) -> List[str]:
        """
        Parse the generator response to extract individual expanded queries.

        :param generator_response: The raw text response from the generator.
        :return: List of parsed expanded queries.
        """
        if not generator_response.strip():
            return []

        try:
            return json.loads(generator_response)
        except json.JSONDecodeError:
            return []
