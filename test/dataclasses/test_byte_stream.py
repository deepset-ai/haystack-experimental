import pytest
from base64 import b64encode
from pathlib import Path
from unittest.mock import mock_open, patch

from haystack_experimental.dataclasses.byte_stream import ByteStream

@pytest.fixture
def byte_stream():
    test_data = b"test data"
    test_meta = {"key": "value"}
    test_mime = "text/plain"
    return ByteStream(data=test_data, meta=test_meta, mime_type=test_mime)

def test_init(byte_stream):
    assert byte_stream.data == b"test data"
    assert byte_stream.meta == {"key": "value"}
    assert byte_stream.mime_type == "text/plain"

def test_type_property(byte_stream):
    assert byte_stream.type == "text"
    stream_without_mime = ByteStream(data=b"test data")
    assert stream_without_mime.type is None

def test_subtype_property(byte_stream):
    assert byte_stream.subtype == "plain"
    stream_without_mime = ByteStream(data=b"test data")
    assert stream_without_mime.subtype is None

@patch("builtins.open", new_callable=mock_open)
def test_to_file(mock_file, byte_stream):
    path = Path("test.txt")
    byte_stream.to_file(path)
    mock_file.assert_called_once_with(path, "wb")
    mock_file().write.assert_called_once_with(b"test data")

@patch("builtins.open", new_callable=mock_open, read_data=b"test data")
def test_from_file_path(mock_file):
    path = Path("test.txt")
    with patch("mimetypes.guess_type", return_value=("text/plain", None)):
        byte_stream = ByteStream.from_file_path(path)
        assert byte_stream.data == b"test data"
        assert byte_stream.mime_type == "text/plain"

@patch("mimetypes.guess_type", return_value=(None, None))
@patch("haystack_experimental.dataclasses.byte_stream.logger.warning")
def test_from_file_path_unknown_mime(mock_warning, _, byte_stream):
    path = Path("test.txt")
    with patch("builtins.open", new_callable=mock_open, read_data=b"test data"):
        byte_stream = ByteStream.from_file_path(path)
        assert byte_stream.mime_type is None
        mock_warning.assert_called_once()

def test_from_string():
    text = "Hello, World!"
    byte_stream = ByteStream.from_string(text, mime_type="text/plain")
    assert byte_stream.data == text.encode("utf-8")
    assert byte_stream.mime_type == "text/plain"

def test_to_string():
    byte_stream = ByteStream(data=b"Hello, World!")
    assert byte_stream.to_string() == "Hello, World!"

def test_from_base64():
    base64_string = b64encode(b"test data").decode("utf-8")
    byte_stream = ByteStream.from_base64(base64_string, mime_type="text/plain")
    assert byte_stream.data == b"test data"
    assert byte_stream.mime_type == "text/plain"

def test_to_base64(byte_stream):
    expected = b64encode(b"test data").decode("utf-8")
    assert byte_stream.to_base64() == expected

def test_from_dict():
    data = {
        "data": b64encode(b"test data").decode("utf-8"),
        "meta": {"key": "value"},
        "mime_type": "text/plain",
    }
    byte_stream = ByteStream.from_dict(data)
    assert byte_stream.data == b"test data"
    assert byte_stream.meta == {"key": "value"}
    assert byte_stream.mime_type == "text/plain"

def test_to_dict(byte_stream):
    expected = {
        "data": b64encode(b"test data").decode("utf-8"),
        "meta": {"key": "value"},
        "mime_type": "text/plain",
    }
    assert byte_stream.to_dict() == expected
