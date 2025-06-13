# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, logging
from haystack.components.builders.answer_builder import AnswerBuilder as HaystackAnswerBuilder
from haystack.dataclasses.chat_message import ChatMessage

from haystack_experimental.dataclasses import GeneratedAnswer

logger = logging.getLogger(__name__)


@component
class AnswerBuilder(HaystackAnswerBuilder):
    """
    Converts a query and Generator replies into a `GeneratedAnswer` object.

    AnswerBuilder parses Generator replies using custom regular expressions.
    Check out the usage example below to see how it works.
    Optionally, it can also take documents and metadata from the Generator to add to the `GeneratedAnswer` object.
    AnswerBuilder works with both non-chat and chat Generators.

    ### Usage example

    ```python
    from haystack.components.builders import AnswerBuilder

    builder = AnswerBuilder(pattern="Answer: (.*)")
    builder.run(query="What's the answer?", replies=["This is an argument. Answer: This is the answer."])
    ```
    """

    @component.output_types(answers=List[GeneratedAnswer])
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        query: str,
        replies: Union[List[str], List[ChatMessage]],
        meta: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Document]] = None,
        pattern: Optional[str] = None,
        reference_pattern: Optional[str] = None,
    ) -> Dict[str, List[GeneratedAnswer]]:
        """
        Turns the output of a Generator into `GeneratedAnswer` objects using regular expressions.

        :param query:
            The input query used as the Generator prompt.
        :param replies:
            The output of the Generator. Can be a list of strings or a list of `ChatMessage` objects.
        :param meta:
            The metadata returned by the Generator. If not specified, the generated answer will contain no metadata.
        :param documents:
            The documents used as the Generator inputs. If specified, they are added to
            the`GeneratedAnswer` objects.
            If both `documents` and `reference_pattern` are specified, the documents referenced in the
            Generator output are extracted from the input documents and added to the `GeneratedAnswer` objects.
        :param pattern:
            The regular expression pattern to extract the answer text from the Generator.
            If not specified, the entire response is used as the answer.
            The regular expression can have one capture group at most.
            If present, the capture group text
            is used as the answer. If no capture group is present, the whole match is used as the answer.
                Examples:
                    `[^\\n]+$` finds "this is an answer" in a string "this is an argument.\\nthis is an answer".
                    `Answer: (.*)` finds "this is an answer" in a string
                    "this is an argument. Answer: this is an answer".
        :param reference_pattern:
            The regular expression pattern used for parsing the document references.
            If not specified, no parsing is done, and all documents are referenced.
            References need to be specified as indices of the input documents and start at [1].
            Example: `\\[(\\d+)\\]` finds "1" in a string "this is an answer[1]".

        :returns: A dictionary with the following keys:
            - `answers`: The answers received from the output of the Generator.
        """
        if not meta:
            meta = [{}] * len(replies)
        elif len(replies) != len(meta):
            raise ValueError(f"Number of replies ({len(replies)}), and metadata ({len(meta)}) must match.")

        if pattern:
            AnswerBuilder._check_num_groups_in_regex(pattern)

        pattern = pattern or self.pattern
        reference_pattern = reference_pattern or self.reference_pattern
        all_answers = []

        replies_to_iterate = replies
        meta_to_iterate = meta

        if self.last_message_only and replies:
            replies_to_iterate = replies[-1:]
            meta_to_iterate = meta[-1:]

        for reply, given_metadata in zip(replies_to_iterate, meta_to_iterate):
            # Extract content from ChatMessage objects if reply is a ChatMessages, else use the string as is
            if isinstance(reply, ChatMessage):
                extracted_reply = reply.text or ""
            else:
                extracted_reply = str(reply)
            extracted_metadata = reply.meta if isinstance(reply, ChatMessage) else {}

            extracted_metadata = {**extracted_metadata, **given_metadata}
            extracted_metadata["all_messages"] = replies

            referenced_docs = []
            if documents:
                if reference_pattern:
                    reference_idxs = AnswerBuilder._extract_reference_idxs(extracted_reply, reference_pattern)
                else:
                    reference_idxs = [doc_idx for doc_idx, _ in enumerate(documents)]

                for idx in reference_idxs:
                    try:
                        referenced_docs.append(documents[idx])
                    except IndexError:
                        logger.warning(
                            "Document index '{index}' referenced in Generator output is out of range. ", index=idx + 1
                        )

            answer_string = AnswerBuilder._extract_answer_string(extracted_reply, pattern)
            answer = GeneratedAnswer(
                data=answer_string, query=query, documents=referenced_docs, meta=extracted_metadata
            )
            all_answers.append(answer)

        return {"answers": all_answers}
