"""Unit tests for app.py — no network or disk access required."""
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


# Import app with all heavy dependencies mocked so tests run instantly.
def _load_app():
    sys.modules.pop("app", None)

    mock_llm = MagicMock()
    mock_llm._get_inference_client_kwargs.return_value = {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "provider": "auto",
        "token": "fake-token",
        "timeout": None,
        "headers": None,
        "cookies": None,
    }

    mock_qe = AsyncMock()
    mock_index = MagicMock()
    mock_index.as_query_engine.return_value = mock_qe

    with (
        patch("llama_index.llms.huggingface_api.HuggingFaceInferenceAPI", return_value=mock_llm),
        patch("llama_index.embeddings.huggingface.HuggingFaceEmbedding", return_value=MagicMock()),
        patch("llama_index.core.StorageContext.from_defaults", return_value=MagicMock()),
        patch("llama_index.core.load_index_from_storage", return_value=mock_index),
        patch("os.path.exists", return_value=True),
    ):
        import app

    return app, mock_qe


_app, _mock_qe = _load_app()


# Test 1: search_pdf returns a string when the query engine succeeds.
def test_search_pdf_returns_string_on_success():
    _mock_qe.aquery.side_effect = None
    _mock_qe.aquery.return_value = "The document covers interview tips."

    result = asyncio.run(_app.search_pdf("what is this document about?"))

    assert isinstance(result, str)
    assert "interview" in result.lower()


# Test 2: search_pdf returns a friendly error string when the query engine raises.
def test_search_pdf_handles_exception():
    _mock_qe.aquery.side_effect = RuntimeError("connection refused")

    result = asyncio.run(_app.search_pdf("summarize the document"))

    assert result.startswith("Error searching documents:")
    assert "connection refused" in result


# Test 3: _PersistentAsyncInferenceClient.close() is a no-op and never calls the parent.
def test_persistent_client_close_is_noop():
    from huggingface_hub import AsyncInferenceClient

    client = _app._PersistentAsyncInferenceClient(token="fake-token")

    with patch.object(AsyncInferenceClient, "close", new_callable=AsyncMock) as parent_close:
        asyncio.run(client.close())

    parent_close.assert_not_called()
