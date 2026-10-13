"""Provider-free Responses streams retain tool attributes and bounded completion."""

import asyncio
from types import SimpleNamespace

import pytest

from agent import relay_llm


def test_replayed_tool_item_supports_sdk_attribute_access(monkeypatch):
    from tests.agent.test_run_agent_codex_responses import _build_agent, _codex_request_kwargs

    agent = _build_agent(monkeypatch)
    frames = [
        {"type": "response.output_item.done", "item": {
            "type": "function_call", "name": "terminal", "call_id": "call-1", "arguments": "{}",
        }},
        {"type": "response.completed", "response": {"status": "completed"}},
    ]

    def replay(_request, _factory, **kwargs):
        return iter(kwargs["chunk_adapter"](frame) for frame in frames)

    monkeypatch.setattr(relay_llm, "stream", replay)
    response = agent._run_codex_stream(_codex_request_kwargs())
    assert response.output[0].name == "terminal"
    assert response.output[0].call_id == "call-1"


def test_local_stream_drain_consumes_end_before_closing():
    seen = []

    async def stream():
        yield {"type": "response.completed"}
        seen.append("end")

    loop = asyncio.new_event_loop()
    owner = SimpleNamespace(_loop=loop, _stream=stream(),
        _finish_logical=lambda outcome: seen.append(outcome),
        _close=lambda **kwargs: seen.append("closed"))
    try:
        relay_llm.ManagedLlmStream.drain_local(owner, 2)
        assert seen == ["end", "success", "closed"]
    finally:
        loop.close()


def test_local_stream_drain_has_a_deadline():
    async def stream():
        await asyncio.Event().wait()
        yield {}

    loop = asyncio.new_event_loop()
    owner = SimpleNamespace(_loop=loop, _stream=stream())
    try:
        with pytest.raises(TimeoutError):
            relay_llm.ManagedLlmStream.drain_local(owner, 0.01)
    finally:
        loop.close()
