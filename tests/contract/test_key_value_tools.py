"""The key/value XML tool grammar round-trips through shared serving protocols."""

import json

import pytest

from monolith.serving.protocol import APIError, ChatRequest, parse_completion
from monolith.serving.tool_stream import tool_prefixes


def request():
    return ChatRequest(
        model="local",
        messages=[{"role": "user", "content": "Look it up"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["city", "count"],
                    },
                },
            }
        ],
    )


def test_key_value_tool_call_and_history():
    text = "<tool_call>lookup<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>count</arg_key><arg_value>2</arg_value></tool_call>"
    req = request()
    for length in range(1, len(text)):
        assert list(tool_prefixes(text[:length], req)) == []
    message, finish = parse_completion(text, req, "stop")
    assert finish == "tool_calls"
    (call,) = message["tool_calls"]
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "count": 2}
    (streamed,) = list(tool_prefixes(text, req))
    assert streamed[1] == "lookup"
    assert json.loads(streamed[3]) == {"city": "Paris", "count": 2}
    history = req.model_copy(
        update={
            "messages": req.messages + [type(req.messages[0]).model_validate(message)]
        }
    )
    messages, _ = history.template_inputs()
    assert messages[-1]["tool_calls"][0]["function"]["arguments"] == {
        "city": "Paris",
        "count": 2,
    }


@pytest.mark.parametrize(
    "body",
    [
        "lookup<arg_key>city</arg_key><arg_value>x</arg_value>",
        "lookup<arg_key>city</arg_key><arg_value>x</arg_value><arg_key>city</arg_key><arg_value>y</arg_value>",
        "lookup<arg_key>count</arg_key><arg_value>not-an-int</arg_value>",
        "lookup<arg_key>city</arg_key>",
        "unknown",
    ],
)
def test_invalid_key_value_calls_fail(body):
    with pytest.raises(APIError, match="invalid tool call"):
        parse_completion(f"<tool_call>{body}</tool_call>", request(), "stop")
