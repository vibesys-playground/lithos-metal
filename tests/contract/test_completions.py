"""Token-prompt HTTP contract through the real backend, with a deterministic session."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from monolith.serve import Backend, create_app


class Tokenizer:
    def __len__(self):
        return 100

    def decode(self, ids, **options):
        return "".join(f"<{i}>" for i in ids)

    def apply_chat_template(self, *args, **options):
        raise AssertionError("Raw completions must not apply a chat template")


class Session:
    eos = 99
    prefix_cache = None

    def __init__(self):
        self.prompts = []

    def generate(self, ids, limit, **options):
        self.prompts.append(ids)
        tokens = [7, 99, 8][:limit]
        for n in range(1, len(tokens) + 1):
            if options.get("on_tokens"):
                options["on_tokens"](tokens[:n])
        return SimpleNamespace(tokens=tokens, cached_prompt_tokens=2)


def client():
    backend = Backend.__new__(Backend)
    backend.max_context = 32
    backend.session, backend.tokenizer = Session(), Tokenizer()
    backend.select_session = lambda request, n: None
    return TestClient(create_app(backend, "test")), backend


def test_stream_preserves_ids_and_reports_actual_cache_usage():
    api, backend = client()
    body = dict(
        model="test",
        prompt=[1, 2, 3],
        max_tokens=3,
        ignore_eos=True,
        return_token_ids=True,
        stream=True,
        stream_options={"include_usage": True},
        rid="request-1",
    )
    import json

    response = api.post("/v1/completions", json=body)
    frames = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    ids = [
        i
        for frame in frames
        for choice in frame["choices"]
        for i in choice.get("token_ids", [])
    ]
    assert ids == [7, 99, 8]
    assert frames[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    assert frames[-2]["choices"][0]["finish_reason"] == "length"
    assert backend.session.prompts == [[1, 2, 3]]
    assert response.text.endswith("data: [DONE]\n\n")


def test_nonstream_and_request_locality():
    api, backend = client()
    for prompt in ([1, 2], [3, 4, 5]):
        result = api.post(
            "/v1/completions",
            json=dict(model="test", prompt=prompt, max_tokens=2, return_token_ids=True),
        ).json()
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["completion_token_ids"] == [7, 99]
        assert result["usage"]["prompt_tokens"] == len(prompt)
    assert backend.session.prompts == [[1, 2], [3, 4, 5]]


@pytest.mark.parametrize("prompt", [[], [-1], [100], [True], "text", [[1]]])
def test_invalid_raw_prompts_never_reach_generation(prompt):
    api, backend = client()
    assert (
        api.post("/v1/completions", json=dict(model="test", prompt=prompt)).status_code
        == 400
    )
    assert backend.session.prompts == []
