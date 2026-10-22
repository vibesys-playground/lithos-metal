"""Real CLI chat/tool round trip against the shared Flint pack; run under lockf."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (Path(args.pack) / "manifest.json").is_file():
        raise ValueError("serve smoke requires the existing shared weight pack")
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [
        str(Path(sys.prefix) / "bin/lithos-metal"),
        "serve",
        "--model",
        args.model,
        "--pack",
        args.pack,
        "--no-draft",
        "--max-context",
        "9216",
        "--prefill-chunk-size",
        "32",
        "--no-commute-norm",
        "--no-warmup",
        "--keep-warm",
        "0",
        "--local-files-only",
        "--served-model-name",
        "flint",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    headers = {"Content-Type": "application/json"}
    key = next(
        (
            os.environ[name]
            for name in ("LITHOS_METAL_API_KEY", "LMK_API_KEY", "MONOLITH_API_KEY")
            if os.environ.get(name)
        ),
        None,
    )
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def complete(messages, tools=None):
        body = {
            "model": "flint",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 256,
        }
        if tools:
            body["tools"] = tools
        request = Request(
            url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urlopen(request, timeout=300) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".server.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited {process.returncode}; see {log.name}"
                    )
                try:
                    with urlopen(url + "/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "server did not become healthy within 120 seconds"
                    )
                time.sleep(0.2)
            chat = complete(
                [
                    {
                        "role": "user",
                        "content": "What is 1 + 1? Reply with only the answer.",
                    }
                ]
            )
            content = chat["choices"][0]["message"].get("content")
            if not content or "2" not in content:
                raise AssertionError(f"unexpected arithmetic response: {chat}")
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather in a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
            messages = [
                {
                    "role": "system",
                    "content": "Use the available tool for weather questions. Call it immediately; do not guess the weather.",
                },
                {
                    "role": "user",
                    "content": "What is the weather in Paris? Call get_weather with city Paris.",
                },
            ]
            tool = complete(messages, tools)
            assistant = tool["choices"][0]["message"]
            calls = assistant.get("tool_calls", [])
            if len(calls) != 1 or calls[0]["function"]["name"] != "get_weather":
                raise AssertionError(f"missing get_weather call: {tool}")
            arguments = json.loads(calls[0]["function"]["arguments"])
            if arguments != {"city": "Paris"}:
                raise AssertionError(f"unexpected tool arguments: {arguments}")
            messages += [
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": calls[0]["id"],
                    "content": "Sunny, 20 degrees Celsius.",
                },
            ]
            followup = complete(messages, tools)
            if not followup["choices"][0]["message"].get("content"):
                raise AssertionError(f"empty tool-result reply: {followup}")
            args.out.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "command": command,
                        "chat": chat,
                        "tool": tool,
                        "followup": followup,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(args.out.read_text(), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
