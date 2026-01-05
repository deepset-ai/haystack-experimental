# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Optional, Union

from haystack import default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret

from haystack_experimental.chat_message_stores.errors import ChatMessageStoreError

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"opensearch-haystack\"'") as opensearch_import:
    from opensearchpy import AsyncHttpConnection, AsyncOpenSearch, OpenSearch
    from opensearchpy.helpers import async_bulk, bulk
    from haystack_integrations.document_stores.opensearch.auth import AsyncAWSAuth, AWSAuth
    from haystack_integrations.document_stores.opensearch.filters import normalize_filters

Hosts = Union[str, list[Union[str, Mapping[str, Union[str, int]]]]]

DEFAULT_SETTINGS = {}
DEFAULT_MAX_CHUNK_BYTES = 100 * 1024 * 1024


class OpenSearchChatMessageStore:
    """
    An instance of an OpenSearch database you can use to store all types of data.

    This document store is a thin wrapper around the OpenSearch client.
    It allows you to store and retrieve documents from an OpenSearch index.

    Usage example:
    ```python
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore
    from haystack import Document

    document_store = OpenSearchDocumentStore(hosts="localhost:9200")

    document_store.write_documents(
        [
            Document(content="My first document", id="1"),
            Document(content="My second document", id="2"),
        ]
    )

    print(document_store.count_documents())
    # 2

    print(document_store.filter_documents())
    # [Document(id='1', content='My first document', ...), Document(id='2', content='My second document', ...)]
    ```
    """

    def __init__(
        self,
        *,
        hosts: Optional[Hosts] = None,
        index: str = "default",
        max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
        mappings: Optional[dict[str, Any]] = None,
        settings: Optional[dict[str, Any]] = DEFAULT_SETTINGS,
        create_index: bool = True,
        http_auth: Any = (
            Secret.from_env_var("OPENSEARCH_USERNAME", strict=False),  # noqa: B008
            Secret.from_env_var("OPENSEARCH_PASSWORD", strict=False),  # noqa: B008
        ),
        use_ssl: Optional[bool] = None,
        verify_certs: Optional[bool] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Creates a new OpenSearchDocumentStore instance.

        The ``mappings``, and ``settings`` arguments are only used if the index does not
        exist and needs to be created. If the index already exists, its current configurations will be used.

        For more information on connection parameters, see the [official OpenSearch documentation](https://opensearch.org/docs/latest/clients/python-low-level/#connecting-to-opensearch)

        :param hosts: List of hosts running the OpenSearch client. Defaults to None
        :param index: Name of index in OpenSearch, if it doesn't exist it will be created. Defaults to "default"
        :param max_chunk_bytes: Maximum size of the requests in bytes. Defaults to 100MB
        :param mappings: The mapping of how the documents are stored and indexed.
            Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/field-types/)
            for more information. Defaults to None.
        :param settings: The settings of the index to be created. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#index-settings)
            for more information. Defaults to `{"index.knn": True}`.
        :param create_index: Whether to create the index if it doesn't exist. Defaults to True
        :param http_auth: http_auth param passed to the underlying connection class.
            For basic authentication with default connection class `Urllib3HttpConnection` this can be
            - a tuple of (username, password)
            - a list of [username, password]
            - a string of "username:password"
            If not provided, will read values from OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD environment variables.
            For AWS authentication with `Urllib3HttpConnection` pass an instance of `AWSAuth`.
            Defaults to None.
        :param use_ssl: Whether to use SSL. Defaults to None.
        :param verify_certs: Whether to verify certificates. Defaults to None.
        :param timeout: Timeout in seconds. Defaults to None.
        :param **kwargs: Optional arguments that ``OpenSearch`` takes. For the full list of supported kwargs,
            see the [official OpenSearch reference](https://opensearch-project.github.io/opensearch-py/api-ref/clients/opensearch_client.html)
        """
        self._hosts = hosts
        self._index = index
        self._max_chunk_bytes = max_chunk_bytes
        self._mappings = mappings or self._get_default_mappings()
        self._settings = settings
        self._create_index = create_index
        self._http_auth_are_secrets = False

        # Handle authentication
        if isinstance(http_auth, (tuple, list)) and len(http_auth) == 2:  # noqa: PLR2004
            username, password = http_auth
            if isinstance(username, Secret) and isinstance(password, Secret):
                self._http_auth_are_secrets = True
                username_val = username.resolve_value()
                password_val = password.resolve_value()
                http_auth = [username_val, password_val] if username_val and password_val else None

        self._http_auth = http_auth
        self._use_ssl = use_ssl
        self._verify_certs = verify_certs
        self._timeout = timeout
        self._kwargs = kwargs

        # Client is initialized lazily to prevent side effects when
        # the document store is instantiated.
        self._client: Optional[OpenSearch] = None
        self._async_client: Optional[AsyncOpenSearch] = None
        self._initialized = False

    def _get_default_mappings(self) -> dict[str, Any]:
        default_mappings: dict[str, Any] = {
            "properties": {
                # "embedding": {"type": "knn_vector", "index": True, "dimension": self._embedding_dim},
                "content": {"type": "text"},
            },
            "dynamic_templates": [
                {
                    "strings": {
                        "match_mapping_type": "string",
                        "mapping": {"type": "keyword"},
                    }
                }
            ],
        }
        return default_mappings

    def create_index(
        self,
        index: Optional[str] = None,
        mappings: Optional[dict[str, Any]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Creates an index in OpenSearch.

        Note that this method ignores the `create_index` argument from the constructor.

        :param index: Name of the index to create. If None, the index name from the constructor is used.
        :param mappings: The mapping of how the documents are stored and indexed. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/field-types/)
            for more information. If None, the mappings from the constructor are used.
        :param settings: The settings of the index to be created. Please see the [official OpenSearch docs](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/#index-settings)
            for more information. If None, the settings from the constructor are used.
        """
        self._ensure_initialized()
        assert self._client is not None

        if not index:
            index = self._index
        if not mappings:
            mappings = self._mappings
        if not settings:
            settings = self._settings

        if not self._client.indices.exists(index=index):
            self._client.indices.create(index=index, body={"mappings": mappings, "settings": settings})

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        # Handle http_auth serialization
        http_auth: Union[list[dict[str, Any]], dict[str, Any], tuple[str, str], list[str], str] = ""
        if isinstance(self._http_auth, list) and self._http_auth_are_secrets:
            # Recreate the Secret objects for serialization
            http_auth = [
                Secret.from_env_var("OPENSEARCH_USERNAME", strict=False).to_dict(),
                Secret.from_env_var("OPENSEARCH_PASSWORD", strict=False).to_dict(),
            ]
        elif isinstance(self._http_auth, AWSAuth):
            http_auth = self._http_auth.to_dict()
        else:
            http_auth = self._http_auth

        return default_to_dict(
            self,
            hosts=self._hosts,
            index=self._index,
            max_chunk_bytes=self._max_chunk_bytes,
            mappings=self._mappings,
            settings=self._settings,
            create_index=self._create_index,
            http_auth=http_auth,
            use_ssl=self._use_ssl,
            verify_certs=self._verify_certs,
            timeout=self._timeout,
            **self._kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenSearchChatMessageStore":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        if http_auth := init_params.get("http_auth"):
            if isinstance(http_auth, dict):
                init_params["http_auth"] = AWSAuth.from_dict(http_auth)
            elif isinstance(http_auth, (tuple, list)):
                are_secrets = all(isinstance(item, dict) and "type" in item for item in http_auth)
                init_params["http_auth"] = [Secret.from_dict(item) for item in http_auth] if are_secrets else http_auth
        return default_from_dict(cls, data)

    def _ensure_initialized(self):
        # Ideally, we have a warm-up stage for document stores as well as components.
        if not self._initialized:
            self._client = OpenSearch(
                hosts=self._hosts,
                http_auth=self._http_auth,
                use_ssl=self._use_ssl,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
                **self._kwargs,
            )
            async_http_auth = AsyncAWSAuth(self._http_auth) if isinstance(self._http_auth, AWSAuth) else self._http_auth
            self._async_client = AsyncOpenSearch(
                hosts=self._hosts,
                http_auth=async_http_auth,
                use_ssl=self._use_ssl,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
                # IAM Authentication requires AsyncHttpConnection:
                # https://github.com/opensearch-project/opensearch-py/blob/main/guides/auth.md#iam-authentication-with-an-async-client
                connection_class=AsyncHttpConnection,
                **self._kwargs,
            )

            self._initialized = True

            self._ensure_index_exists()

    def _ensure_index_exists(self):
        assert self._client is not None

        if self._client.indices.exists(index=self._index):
            logger.debug(
                "The index '{index}' already exists. The `mappings`, and `settings` values will be ignored.",
                index=self._index,
            )
        elif self._create_index:
            # Create the index if it doesn't exist
            body = {"mappings": self._mappings, "settings": self._settings}
            self._client.indices.create(index=self._index, body=body)

    def count_messages(self, chat_history_id: str) -> int:
        """
        Returns the number of chat messages stored in the store.

        :param chat_history_id:
            The chat history id for which to count messages.

        :returns: The number of messages.
        """
        self._ensure_initialized()
        assert self._client is not None

        return self._client.count(index=self._index)["count"]

    async def count_messages_async(self, chat_history_id: str) -> int:
        """
        Asynchronously returns the total number of chat messages in the store.

        :param chat_history_id:
            The chat history id for which to count messages.

        :returns: The number of messages.
        """
        self._ensure_initialized()

        assert self._async_client is not None
        return (await self._async_client.count(index=self._index))["count"]

    @staticmethod
    def _deserialize_search_hits(hits: list[dict[str, Any]]) -> list[ChatMessage]:
        out = []
        for hit in hits:
            out.append(ChatMessage.from_dict(hit["_source"]))
        return out

    # TODO This should be updated to only use chat_history_id filter
    def _prepare_filter_search_request(self, filters: Optional[dict[str, Any]]) -> dict[str, Any]:
        search_kwargs: dict[str, Any] = {"size": 10_000}
        if filters:
            search_kwargs["query"] = {"bool": {"filter": normalize_filters(filters)}}
        return search_kwargs

    def _search_messages(self, request_body: dict[str, Any]) -> list[ChatMessage]:
        assert self._client is not None
        search_results = self._client.search(index=self._index, body=request_body)
        return self._deserialize_search_hits(search_results["hits"]["hits"])

    async def _search_messages_async(self, request_body: dict[str, Any]) -> list[ChatMessage]:
        assert self._async_client is not None
        search_results = await self._async_client.search(index=self._index, body=request_body)
        return self._deserialize_search_hits(search_results["hits"]["hits"])

    # TODO This needs to be updated to implement last_k
    def retrieve_messages(self, chat_history_id: str, last_k: Optional[int] = None) -> list[ChatMessage]:
        """
        Retrieves chat messages stored under the given chat history id.

        :param chat_history_id:
            The chat history id from which to retrieve messages.
        :param last_k:
            The number of last messages to retrieve. If unspecified, the last_k parameter passed
            to the constructor will be used.

        :returns: A list of chat messages.
        """
        self._ensure_initialized()
        filters = {"field": "chat_history_id", "operator": "==", "value": chat_history_id}
        return self._search_messages(self._prepare_filter_search_request(filters))

    async def retrieve_messages_async(self, chat_history_id: str, last_k: Optional[int] = None) -> list[ChatMessage]:
        """
        Asynchronously retrieves chat messages stored under the given chat history id.

        :param chat_history_id:
            The chat history id from which to retrieve messages.
        :param last_k:
            The number of last messages to retrieve. If unspecified, the last_k parameter passed
            to the constructor will be used.

        :returns: A list of chat messages.
        """
        self._ensure_initialized()
        filters = {"field": "chat_history_id", "operator": "==", "value": chat_history_id}
        return await self._search_messages_async(self._prepare_filter_search_request(filters))

    def _prepare_bulk_write_request(
        self, *, messages: list[ChatMessage], policy: DuplicatePolicy, is_async: bool
    ) -> dict[str, Any]:
        if len(messages) > 0 and not isinstance(messages[0], ChatMessage):
            msg = "param 'messages' must contain a list of objects of type ChatMessage"
            raise ValueError(msg)

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        action = "index" if policy == DuplicatePolicy.OVERWRITE else "create"
        opensearch_actions = []
        for msg in messages:
            msg_dict = msg.to_dict()
            opensearch_actions.append({"_op_type": action, "_id": msg.id, "_source": msg_dict})

        return {
            "client": self._client if not is_async else self._async_client,
            "actions": opensearch_actions,
            "refresh": "wait_for",
            "index": self._index,
            "raise_on_error": False,
            "max_chunk_bytes": self._max_chunk_bytes,
            "stats_only": False,
        }

    @staticmethod
    def _process_bulk_write_errors(errors: list[dict[str, Any]], policy: DuplicatePolicy) -> None:
        if len(errors) == 0:
            return

        duplicate_errors_ids = []
        other_errors = []
        for e in errors:
            # OpenSearch might not return a correctly formatted error, in that case we
            # treat it as a generic error
            if "create" not in e:
                other_errors.append(e)
                continue
            error_type = e["create"]["error"]["type"]
            if policy == DuplicatePolicy.FAIL and error_type == "version_conflict_engine_exception":
                duplicate_errors_ids.append(e["create"]["_id"])
            elif policy == DuplicatePolicy.SKIP and error_type == "version_conflict_engine_exception":
                # when the policy is skip, duplication errors are OK and we should not raise an exception
                continue
            else:
                other_errors.append(e)

        if len(duplicate_errors_ids) > 0:
            msg = f"IDs '{', '.join(duplicate_errors_ids)}' already exist in the document store."
            raise DuplicateDocumentError(msg)

        if len(other_errors) > 0:
            msg = f"Failed to write documents to OpenSearch. Errors:\n{other_errors}"
            raise ChatMessageStoreError(msg)

    def write_messages(self, chat_history_id: str, messages: list[ChatMessage]) -> int:
        """
        Writes chat messages to the chat message store.

        :param chat_history_id:
            The chat history id under which to store the messages.
        :param messages:
            A list of ChatMessages to write.

        :returns: The number of messages written.
        """
        self._ensure_initialized()

        bulk_params = self._prepare_bulk_write_request(messages=documents, policy=policy, is_async=False)
        documents_written, errors = bulk(**bulk_params)
        self._process_bulk_write_errors(errors, policy)
        return documents_written

    async def write_messages_async(self, chat_history_id: str, messages: list[ChatMessage]) -> int:
        """
        Asynchronously write chat messages to the chat message store.

        :param chat_history_id:
            The chat history id under which to store the messages.
        :param messages:
            A list of ChatMessages to write.

        :returns: The number of messages written.
        """
        self._ensure_initialized()
        bulk_params = self._prepare_bulk_write_request(messages=documents, policy=policy, is_async=True)
        documents_written, errors = await async_bulk(**bulk_params)
        # since we call async_bulk with stats_only=False, errors is guaranteed to be a list (not int)
        self._process_bulk_write_errors(errors=errors, policy=policy)  # type: ignore[arg-type]
        return documents_written

    def _prepare_bulk_delete_request(self, *, document_ids: list[str], is_async: bool) -> dict[str, Any]:
        return {
            "client": self._client if not is_async else self._async_client,
            "actions": ({"_op_type": "delete", "_id": id_} for id_ in document_ids),
            "refresh": "wait_for",
            "index": self._index,
            "raise_on_error": False,
            "max_chunk_bytes": self._max_chunk_bytes,
        }

    def delete_documents(self, document_ids: list[str]) -> None:
        """
        Deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """

        self._ensure_initialized()

        bulk(**self._prepare_bulk_delete_request(document_ids=document_ids, is_async=False))

    async def delete_documents_async(self, document_ids: list[str]) -> None:
        """
        Asynchronously deletes documents that match the provided `document_ids` from the document store.

        :param document_ids: the document ids to delete
        """
        self._ensure_initialized()

        await async_bulk(**self._prepare_bulk_delete_request(document_ids=document_ids, is_async=True))

    def _prepare_delete_all_request(self, *, is_async: bool) -> dict[str, Any]:
        return {
            "index": self._index,
            "body": {"query": {"match_all": {}}},  # Delete all documents
            "wait_for_completion": False if is_async else True,  # block until done (set False for async)
        }

    def delete_all_documents(self, recreate_index: bool = False) -> None:  # noqa: FBT002, FBT001
        """
        Deletes all documents in the document store.

        :param recreate_index: If True, the index will be deleted and recreated with the original mappings and
            settings. If False, all documents will be deleted using the `delete_by_query` API.
        """
        self._ensure_initialized()
        assert self._client is not None

        try:
            if recreate_index:
                # get the current index mappings and settings
                index_name = self._index
                body = {
                    "mappings": self._client.indices.get(self._index)[index_name]["mappings"],
                    "settings": self._client.indices.get(self._index)[index_name]["settings"],
                }
                body["settings"]["index"].pop("uuid", None)
                body["settings"]["index"].pop("creation_date", None)
                body["settings"]["index"].pop("provided_name", None)
                body["settings"]["index"].pop("version", None)
                self._client.indices.delete(index=self._index)
                self._client.indices.create(index=self._index, body=body)
                logger.info(
                    "The index '{index}' recreated with the original mappings and settings.",
                    index=self._index,
                )

            else:
                result = self._client.delete_by_query(**self._prepare_delete_all_request(is_async=False))
                logger.info(
                    "Deleted all the {n_docs} documents from the index '{index}'.",
                    index=self._index,
                    n_docs=result["deleted"],
                )
        except Exception as e:
            msg = f"Failed to delete all documents from OpenSearch: {e!s}"
            raise ChatMessageStoreError(msg) from e

    async def delete_all_documents_async(self, recreate_index: bool = False) -> None:  # noqa: FBT002, FBT001
        """
        Asynchronously deletes all documents in the document store.

        :param recreate_index: If True, the index will be deleted and recreated with the original mappings and
            settings. If False, all documents will be deleted using the `delete_by_query` API.
        """
        self._ensure_initialized()
        assert self._async_client is not None

        try:
            if recreate_index:
                # get the current index mappings and settings
                index_name = self._index
                index_info = await self._async_client.indices.get(self._index)
                body = {
                    "mappings": index_info[index_name]["mappings"],
                    "settings": index_info[index_name]["settings"],
                }
                body["settings"]["index"].pop("uuid", None)
                body["settings"]["index"].pop("creation_date", None)
                body["settings"]["index"].pop("provided_name", None)
                body["settings"]["index"].pop("version", None)

                await self._async_client.indices.delete(index=self._index)
                await self._async_client.indices.create(index=self._index, body=body)
            else:
                await self._async_client.delete_by_query(**self._prepare_delete_all_request(is_async=True))

        except Exception as e:
            msg = f"Failed to delete all documents from OpenSearch: {e!s}"
            raise ChatMessageStoreError(msg) from e
