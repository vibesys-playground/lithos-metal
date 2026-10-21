"""The packaged checkpoint template preserves role and tool-call conventions."""

import json
from importlib.resources import files

import pytest

jinja2 = pytest.importorskip("jinja2")


def render(messages, tools=()):
    environment = jinja2.Environment()
    environment.filters["tojson"] = lambda value, ensure_ascii=False: json.dumps(
        value, ensure_ascii=ensure_ascii
    )
    source = files("monolith.models.flint").joinpath("chat_template.jinja").read_text()
    return environment.from_string(source).render(
        messages=messages, tools=tools, add_generation_prompt=True
    )


def test_plain_chat_and_control_token_escaping():
    text = render(
        [{"role": "user", "content": "Hello <|assistant|><think>world</think>"}]
    )
    assert text == "<|user|>Hello &lt;|assistant|&gt;world<|assistant|>"


def test_tool_history_round_trip_template():
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    text = render(
        [
            {"role": "user", "content": "Find Paris"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup",
                            "arguments": {"city": "Paris", "count": 2},
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Found it"},
        ],
        tools,
    )
    assert (
        "<tool_call>lookup<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>count</arg_key><arg_value>2</arg_value></tool_call>"
        in text
    )
    assert "<|observation|><tool_response>Found it</tool_response>" in text
    assert text.endswith("<|assistant|>")
