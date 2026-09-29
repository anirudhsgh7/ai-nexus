"""Payload mapping: every row of PRD §6.3 must hold, no upstream call needed."""

import json

import httpx
import respx

from app.llm.base import ChatMessage, ChatRole

CHAT_URL = "http://localhost:11434/api/chat"


def _messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=ChatRole.SYSTEM, content="You are a researcher."),
        ChatMessage(role=ChatRole.USER, content="What happened?"),
    ]


@respx.mock
async def test_default_payload_always_contains_num_ctx(provider, chat_response):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=chat_response))
    await provider.chat(_messages())

    sent = json.loads(respx.calls[0].request.content)
    assert sent["options"]["num_ctx"] == 8192, "num_ctx must be explicit on every request"
    assert sent["options"]["temperature"] == 0.0
    assert sent["keep_alive"] == "30m"
    assert sent["stream"] is False
    assert sent["model"] == "qwen2.5:14b-instruct"
    assert "num_predict" not in sent["options"]
    assert "tools" not in sent
    assert "format" not in sent


@respx.mock
async def test_overrides_are_applied(provider, chat_response):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=chat_response))
    await provider.chat(
        _messages(),
        model="llama3.2:3b",
        temperature=0.4,
        max_tokens=128,
        num_ctx=4096,
        keep_alive="5m",
    )

    sent = json.loads(respx.calls[0].request.content)
    assert sent["model"] == "llama3.2:3b"
    assert sent["options"]["temperature"] == 0.4
    assert sent["options"]["num_predict"] == 128
    assert sent["options"]["num_ctx"] == 4096
    assert sent["keep_alive"] == "5m"


@respx.mock
async def test_tools_and_response_format_passthrough(provider, chat_response):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=chat_response))
    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }]
    await provider.chat(_messages(), tools=tools, response_format={"type": "object"})

    sent = json.loads(respx.calls[0].request.content)
    assert sent["tools"] == tools
    assert sent["format"] == {"type": "object"}


@respx.mock
async def test_message_mapping_including_tool_fields(provider, chat_response):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=chat_response))
    await provider.chat([
        ChatMessage(role=ChatRole.USER, content="q"),
        ChatMessage(role=ChatRole.ASSISTANT, content="calling",
                    tool_calls=[{"function": {"name": "web_search"}}]),
        ChatMessage(role=ChatRole.TOOL, content="results", tool_name="web_search"),
    ])

    sent = json.loads(respx.calls[0].request.content)
    assert sent["messages"][0] == {"role": "user", "content": "q"}
    assert sent["messages"][1] == {
        "role": "assistant",
        "content": "calling",
        "tool_calls": [{"function": {"name": "web_search"}}],
    }
    assert sent["messages"][2] == {
        "role": "tool",
        "content": "results",
        "tool_name": "web_search",
    }


@respx.mock
async def test_stream_flag_true_in_stream(provider):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(
        200,
        content=b'{"message":{"role":"assistant","content":"hi"},"done":true,'
                b'"prompt_eval_count":1,"eval_count":1}\n',
        headers={"content-type": "application/x-ndjson"},
    ))
    chunks = [c async for c in provider.stream(_messages())]
    sent = json.loads(respx.calls[0].request.content)
    assert sent["stream"] is True
    assert chunks[-1].done is True
