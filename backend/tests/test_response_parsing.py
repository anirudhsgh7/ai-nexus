"""Response parsing against recorded Ollama fixtures (no network)."""

import pytest

from app.llm.base import ResponseParseError
from app.llm.ollama import OllamaProvider
from tests.conftest import load_fixture_json


def test_chat_response_parses(provider, chat_response):
    result = provider._parse_chat_response(chat_response, "qwen2.5:14b-instruct")
    assert result.content == "41"
    assert result.finish_reason == "stop"
    assert result.model == "qwen2.5:7b-instruct"
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 25
    assert result.usage.load_duration_ms == 1.0
    assert result.usage.eval_duration_ms == 3.0
    assert result.tool_calls == []


def test_tool_call_with_object_arguments(provider, chat_response):
    chat_response["message"]["tool_calls"] = [{
        "function": {"name": "web_search", "arguments": {"query": "france"}},
        "id": "call_1",
    }]
    result = provider._parse_chat_response(chat_response, "m")
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "france"}
    assert result.tool_calls[0].id == "call_1"


def test_tool_call_with_string_arguments(provider, chat_response):
    chat_response["message"]["tool_calls"] = [{
        "function": {"name": "web_search", "arguments": '{"query": "france"}'},
    }]
    result = provider._parse_chat_response(chat_response, "m")
    assert result.tool_calls[0].arguments == {"query": "france"}


def test_tool_call_with_broken_string_arguments_raises(provider, chat_response):
    chat_response["message"]["tool_calls"] = [{
        "function": {"name": "web_search", "arguments": "{not json"},
    }]
    with pytest.raises(ResponseParseError):
        provider._parse_chat_response(chat_response, "m")


def test_tool_call_without_name_raises(provider, chat_response):
    chat_response["message"]["tool_calls"] = [{"function": {"arguments": {}}}]
    with pytest.raises(ResponseParseError):
        provider._parse_chat_response(chat_response, "m")


def test_missing_message_raises(provider):
    with pytest.raises(ResponseParseError):
        provider._parse_chat_response({"done": True}, "m")


def test_missing_usage_fields_stay_none(provider, chat_response):
    chat_response.pop("prompt_eval_count")
    chat_response.pop("eval_count")
    chat_response.pop("load_duration")
    result = provider._parse_chat_response(chat_response, "m")
    assert result.usage.prompt_tokens is None
    assert result.usage.completion_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.load_duration_ms is None


def test_model_info_from_details_context_length(provider):
    tags = load_fixture_json("tags_response.json")
    models = [provider._parse_model_info(m) for m in tags["models"]]
    assert models[0].name == "qwen2.5:7b-instruct"
    assert models[0].context_length == 32768
    assert models[0].parameter_size == "7.6B"
    assert models[0].quantization == "Q4_K_M"
    assert "tools" in models[0].capabilities
    assert models[1].context_length == 131072
