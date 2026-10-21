"""Raw-ID diagnostic through the actual backend and completion serialization."""

from types import SimpleNamespace

import pytest

from monolith.serve import Backend
from monolith.serving.protocol import ChatRequest


class TokenSession:
    eos = 99

    def generate(self, ids, limit, **options):
        tokens = [10, 98, 11, 99][:limit]
        if options.get("on_tokens"):
            options["on_tokens"](tokens)
        return SimpleNamespace(tokens=tokens)


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1, 2, 3]

    def decode(self, ids, **kwargs):
        return "".join({10: "Hello", 11: " world"}.get(i, "") for i in ids)


@pytest.mark.parametrize(
    "limit,expected",
    [(1, [10]), (2, [10, 98]), (3, [10, 98, 11]), (4, [10, 98, 11, 99])],
)
def test_raw_ids_include_special_and_eos_and_are_request_local(limit, expected):
    backend = Backend.__new__(Backend)
    backend.max_context, backend.select_session = 64, lambda request, n: None
    backend.session, backend.tokenizer = TokenSession(), Tokenizer()
    request = ChatRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=limit,
        return_token_ids=True,
    )
    seen = []
    content, finish, _, count = backend.complete(request, on_token_ids=seen.extend)
    assert seen == expected and len(seen) == count
    assert content == backend.tokenizer.decode(expected)
    assert finish == ("stop" if 99 in expected else "length")
    other = []
    backend.complete(request, on_token_ids=other.extend)
    assert other == expected


def test_diagnostic_rejects_streaming():
    with pytest.raises(ValueError, match="non-streaming"):
        ChatRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            return_token_ids=True,
            stream=True,
        )


def test_stop_string_truncates_raw_ids_and_usage_at_the_same_committed_token():
    backend = Backend.__new__(Backend)
    backend.max_context, backend.select_session = 64, lambda request, n: None
    backend.session, backend.tokenizer = TokenSession(), Tokenizer()
    request = ChatRequest(
        model="test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=4,
        stop=" world",
        return_token_ids=True,
    )
    seen = []
    content, finish, _, count = backend.complete(request, on_token_ids=seen.extend)
    assert content == "Hello" and finish == "stop"
    assert seen == [10, 98, 11] and count == 3
