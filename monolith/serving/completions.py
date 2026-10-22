"""OpenAI token-prompt Completions wire, sharing the server's admission and engine."""

from .events import WireResponse, sse


class CompletionWireResponse(WireResponse):
    def __init__(self, protocol, model, custom=(), usage=None, **options):
        super().__init__("chat", model, custom, usage, **options)
        self.protocol = "completion"
        self.ids = []

    def start(self):
        return iter(())

    def tokens(self, ids, tokenizer):
        if ids[: len(self.ids)] != self.ids:
            raise RuntimeError("Committed completion token IDs changed")
        delta = ids[len(self.ids) :]
        if not delta:
            return
        self.ids = list(ids)
        # Token IDs are authoritative even when a special token has no text or
        # a UTF-8 character straddles token boundaries.
        text = tokenizer.decode(
            delta, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        self.text += text
        self.output_tokens = len(ids)
        yield self.chunk(
            [{"index": 0, "text": text, "token_ids": delta, "finish_reason": None}]
        )

    def chunk(self, choices, **fields):
        return sse(
            {
                "id": self.id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": choices,
                **fields,
            }
        )

    def body(self, message, finish, prompt_tokens, completion_tokens):
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {
                "cached_tokens": self.metrics.get("cached_prompt_tokens", 0)
            },
        }
        return {
            "id": self.id,
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "text": message.get("content") or "",
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        }

    def finish(self, body, include_usage):
        yield self.chunk(
            [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": body["choices"][0]["finish_reason"],
                }
            ]
        )
        if include_usage:
            yield self.chunk([], usage=body["usage"])
        yield sse("[DONE]")
