import contextlib
import dataclasses
from typing import Any, Dict, Iterator, Optional

from haystack import logging
from haystack.tracing import Span, Tracer

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class LoggingSpan(Span):
    operation_name: str
    tags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a single tag on the span.

        :param key: the name of the tag.
        :param value: the value of the tag.
        """
        self.tags[key] = value


class LoggingTracer(Tracer):
    """
    A simple tracer that logs the operation name and tags of a span.
    """

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None, parent_span: Optional[Span] = None
    ) -> Iterator[Span]:
        """
        Trace the execution of a block of code.

        :param operation_name: the name of the operation being traced.
        :param tags: tags to apply to the newly created span.
        :param parent_span: the parent span to use for the newly created span. Not used in this simple tracer.
        :returns: the newly created span.
        """

        custom_span = LoggingSpan(operation_name, tags=tags or {})

        def convert_to_dict(value):
            if hasattr(value, 'to_dict') and callable(getattr(value, 'to_dict')):
                return value.to_dict()

            elif isinstance(value, dict):
                return {k: convert_to_dict(v) for k, v in value.items()}

            elif isinstance(value, list):
                return [convert_to_dict(item) for item in value]

            return value

        try:
            yield custom_span
        except Exception as e:
            raise e
        # we make sure to log the operation name and tags of the span when the context manager exits
        # both in case of success and error
        finally:
            operation_name = custom_span.operation_name
            tags = custom_span.tags or {}
            logger.debug("Operation: {operation_name}", operation_name=operation_name)
            for tag_name, tag_value in tags.items():
                tag_value = convert_to_dict(tag_value)
                logger.debug("{tag_name}={tag_value}", tag_name=tag_name, tag_value=tag_value)

    def current_span(self) -> Optional[Span]:
        """Return the current active span, if any."""
        # we don't store spans in this simple tracer
        return None
