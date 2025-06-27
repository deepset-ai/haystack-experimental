# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Dict, List, Optional, Union

from jinja2 import meta
from jinja2.sandbox import SandboxedEnvironment

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import TextContent
from haystack.utils import deserialize_chatgenerator_inplace

from haystack_experimental.dataclasses import ImageContent
from haystack_experimental.dataclasses.chat_message import ChatMessage

logger = logging.getLogger(__name__)


DEFAULT_PROMPT_TEMPLATE = """
You are part of an information extraction pipeline that extracts the content of image-based documents.

Extract the content from the provided image.
You need to extract the content exactly.
Format everything as markdown.
Make sure to retain the reading order of the document.

**Visual Elements**
Do not extract figures, drawings, maps, graphs or any other visual elements.
Instead, add a caption that describes briefly what you see in the visual element.
You must describe each visual element. If you only see a visual element without other content, you must describe this 
visual element.
Enclose each image caption with [img-caption][/img-caption]

**Tables**
Make sure to format the table in markdown.
Add a short caption below the table that describes the table's content.
Enclose each table caption with [table-caption][/table-caption].
The caption must be placed below the extracted table.

**Forms**
Reproduce checkbox selections with markdown.

Go ahead and extract!

Document:"""


@component
class LLMDocumentContentExtractor:
    """
    Extracts the content of image-based documents using an LLM (Large Language Model).

    This component expects as input a list of documents and a prompt. The prompt should have a variable called
    `document` that will point to a single document in the list of documents. So to access the content of the document,
    you can use `{{ document.content }}` in the prompt.

    The component will run the LLM on each document in the list and extract the content from the document using the
    vision-enabled ChatGenerator.

    If the LLM fails to extract content of a document, the document will be added to the `failed_documents` list.
    The failed documents will have the keys `content_extraction_error` and `content_extraction_response` in their
    metadata. These documents can be re-run with another extractor to extract metadata by using the
    `content_extraction_response` and `content_extraction_error` in the prompt.
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        prompt: str = DEFAULT_PROMPT_TEMPLATE,
        raise_on_failure: bool = False,
        max_workers: int = 3,
    ):
        """
        Initialize the LLMDocumentContentExtractor component.

        :param prompt: The prompt to be used for the LLM.
        :param chat_generator: a ChatGenerator instance which represents the LLM. In order for the component to work,
            the LLM should be configured to return a JSON object. For example, when using the OpenAIChatGenerator, you
            should pass `{"response_format": {"type": "json_object"}}` in the `generation_kwargs`.
        :param raise_on_failure: Whether to raise an error on failure during the execution of the Generator or
            validation of the JSON output.
        :param max_workers: The maximum number of workers to use in the thread pool executor.
        """
        self.prompt = prompt
        # Ensure the prompt does not contain any variables.
        ast = SandboxedEnvironment().parse(prompt)
        template_variables = meta.find_undeclared_variables(ast)
        variables = list(template_variables)
        if len(variables) != 0:
            raise ValueError(
                f"The prompt must not have any variables only instructions on how to extract the content of the "
                f"image-based document. Found {','.join(variables)} in the prompt."
            )
        self.raise_on_failure = raise_on_failure
        self.max_workers = max_workers
        self._chat_generator = chat_generator

    def warm_up(self):
        """
        Warm up the LLM provider component.
        """
        if hasattr(self._chat_generator, "warm_up"):
            self._chat_generator.warm_up()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        return default_to_dict(
            self,
            prompt=self.prompt,
            chat_generator=component_to_dict(obj=self._chat_generator, name="chat_generator"),
            raise_on_failure=self.raise_on_failure,
            max_workers=self.max_workers,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMDocumentContentExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")
        return default_from_dict(cls, data)

    def _prepare_prompts(self, documents: List[Document]) -> List[Union[ChatMessage, None]]:
        all_prompts: List[Union[ChatMessage, None]] = []
        for document in documents:
            text_content = TextContent(text=self.prompt)
            # TODO Add the normal checks and handle PDFs differently etc.
            image_content = ImageContent.from_file_path(document.meta["file_path"])
            message = ChatMessage.from_user(content_parts=[text_content, image_content])
            all_prompts.append(message)
        return all_prompts

    def _run_on_thread(self, prompt: Optional[ChatMessage]) -> Dict[str, Any]:
        # If prompt is None, return an error dictionary
        if prompt is None:
            return {"error": "Document has no content, skipping LLM call."}

        try:
            result = self._chat_generator.run(messages=[prompt])
        except Exception as e:
            if self.raise_on_failure:
                raise e
            logger.error(
                "LLM {class_name} execution failed. Skipping metadata extraction. Failed with exception '{error}'.",
                class_name=self._chat_generator.__class__.__name__,
                error=e,
            )
            result = {"error": "LLM failed with exception: " + str(e)}
        return result

    @component.output_types(documents=List[Document], failed_documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Extract text content from image-based documents using a Large Language Model.

        The original documents will be returned updated with the extracted text content.

        :param documents: List of documents to extract content from.
        :returns:
            A dictionary with the keys:
            - "documents": A list of documents that were successfully updated with the extracted metadata.
            - "failed_documents": A list of documents that failed to extract metadata. These documents will have
            "content_extraction_error" and "content_extraction_response" in their metadata. These documents can be
            re-run with the extractor to extract a textual representation of their content.
        """
        if not documents:
            logger.warning("No documents provided. Skipping content extraction.")
            return {"documents": [], "failed_documents": []}

        # Create ChatMessage prompts for each document
        all_prompts = self._prepare_prompts(documents=documents)

        # Run the LLM on each prompt
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._run_on_thread, all_prompts)

        successful_documents = []
        failed_documents = []
        for document, result in zip(documents, results):
            if "error" in result:
                new_meta = {
                    **document.meta,
                    "content_extraction_error": result["error"],
                    "content_extraction_response": None,
                }
                failed_documents.append(replace(document, meta=new_meta))
                continue

            # Remove content_extraction_error and content_extraction_response if present from previous runs
            new_meta = {**document.meta}
            new_meta.pop("content_extraction_error", None)
            new_meta.pop("content_extraction_response", None)

            extracted_content = result["replies"][0].text
            successful_documents.append(replace(document, content=extracted_content, meta=new_meta))

        return {"documents": successful_documents, "failed_documents": failed_documents}
