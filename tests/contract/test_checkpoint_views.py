"""Package-declared byte views stay bounded and preserve storage exactly."""

from dataclasses import replace

import numpy as np
import pytest

from monolith.formats.safetensors_reader import SafetensorsDir, write_safetensors


def test_stacked_views_are_zero_copy_typed_slices(tmp_path):
    values = np.arange(4 * 8 * 32, dtype=np.uint8).reshape(4, 8, 32)
    write_safetensors(tmp_path / "model.safetensors", {"stack.weight": ("U8", values)})

    def views(info):
        stride = (info.end - info.start) // 4
        for i in range(4):
            yield replace(
                info,
                name=f"e{i}.weight",
                dtype="U32",
                shape=(8, 8),
                start=info.start + i * stride,
                end=info.start + (i + 1) * stride,
            )

    checkpoint = SafetensorsDir(tmp_path, views=views)
    try:
        assert len(checkpoint.names()) == 4
        for i in range(4):
            assert np.array_equal(
                checkpoint.get(f"e{i}.weight"), values[i].view(np.uint32)
            )
    finally:
        checkpoint.close()


@pytest.mark.parametrize("change", [{"start": 0}, {"end": 10**9}, {"shape": (999,)}])
def test_invalid_view_fails_before_access(tmp_path, change):
    write_safetensors(
        tmp_path / "model.safetensors", {"x.weight": ("U8", np.zeros((16,), np.uint8))}
    )
    with pytest.raises(ValueError, match="checkpoint view.*x.weight"):
        SafetensorsDir(tmp_path, views=lambda info: [replace(info, **change)])


def test_duplicate_view_names_fail(tmp_path):
    write_safetensors(
        tmp_path / "model.safetensors", {"x.weight": ("U8", np.zeros((16,), np.uint8))}
    )
    with pytest.raises(ValueError, match="duplicate checkpoint view"):
        SafetensorsDir(tmp_path, views=lambda info: [info, info])


def test_views_follow_renaming(tmp_path):
    values = np.arange(16, dtype=np.uint8)
    write_safetensors(tmp_path / "model.safetensors", {"stored": ("U8", values)})
    checkpoint = SafetensorsDir(
        tmp_path, rename=lambda name: "renamed", views=lambda info: [info]
    )
    try:
        assert checkpoint.names() == ["renamed"]
        assert np.array_equal(checkpoint.get("renamed"), values)
    finally:
        checkpoint.close()
