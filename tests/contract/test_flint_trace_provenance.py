"""The diagnostic CLI rejects changed reference code before importing a model."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("changed_file", ["model.py", "quantized.py"])
def test_reference_source_must_match_pinned_commit(tmp_path, changed_file):
    reference = tmp_path / "reference"
    source = reference / "engine/src/rec_mlx"
    source.mkdir(parents=True)
    for name in ("model.py", "quantized.py"):
        (source / name).write_text("raise RuntimeError('reference was imported')\n")

    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                "-C",
                str(reference),
                *args,
            ],
            text=True,
            timeout=10,
        ).strip()

    git("init", "--quiet")
    git("add", ".")
    git(
        "-c",
        "user.name=Trace Test",
        "-c",
        "user.email=trace@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "reference",
    )
    revision = git("rev-parse", "HEAD")
    with (source / changed_file).open("a") as stream:
        stream.write("# changed after the pinned commit\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "reference": {"commit": revision},
                "examples": [{"prompt_ids": [1], "greedy_ids": [2]}],
            }
        )
    )
    script = Path(__file__).resolve().parents[2] / "tools/goldens/flint_trace.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "reference",
            "--rec",
            str(reference),
            "--model",
            str(model),
            "--fixture",
            str(fixture),
            "--example",
            "0",
            "--arrays",
            str(tmp_path / "capture"),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode != 0
    assert "reference source differs from its pinned commit" in result.stderr
    assert "reference was imported" not in result.stderr
