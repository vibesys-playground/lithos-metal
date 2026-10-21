"""Shared-cache admission, bounds and failure cleanup without model execution."""

import pytest

from monolith.serving.pack_policy import SharedPackCache, require_pack_space


def build_pack(path):
    (path / "manifest.json").write_text("{}")
    (path / "weights.pack").write_bytes(b"weights")


@pytest.mark.parametrize(
    "pack,margin", [(1, 0), (100, 20), (22_736_175_104, 2 * 1024**3)]
)
def test_admission_boundary(pack, margin):
    require_pack_space(pack + margin, pack, margin)
    with pytest.raises(ValueError, match="Insufficient free disk"):
        require_pack_space(pack + margin - 1, pack, margin)


def test_only_one_extra_and_pinned_pack_never_evicted(tmp_path):
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    build_pack(pinned)
    original = {p.name: p.read_bytes() for p in pinned.iterdir()}
    cache = SharedPackCache(
        tmp_path / "extra", pinned=pinned, pack_bytes=100, margin_bytes=0
    )
    for key in ("a", "b", "a", "c"):
        path = cache.resolve(key * 64, build_pack)
        assert path.name == key * 64
        assert [p.name for p in cache.root.iterdir() if p.is_dir()] == [key * 64]
        assert {p.name: p.read_bytes() for p in pinned.iterdir()} == original

    def unexpected(_):
        pytest.fail("Cache hit rebuilt weights")

    assert cache.resolve("c" * 64, unexpected) == path


def test_failed_builder_and_crash_partial_are_reclaimed(tmp_path):
    cache = SharedPackCache(
        tmp_path / "extra", pinned=tmp_path / "pinned", pack_bytes=100, margin_bytes=0
    )

    def interrupted(path):
        (path / "weights.pack").write_bytes(b"partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        cache.resolve("a" * 64, interrupted)
    assert not list(cache.root.glob("*/weights.pack"))
    partial = cache.root / ".building-interrupted"
    partial.mkdir()
    build_pack(partial)
    cache.resolve("b" * 64, build_pack)
    assert not partial.exists()


@pytest.mark.parametrize("key", ["../escape", "", "x" * 64, "a" * 63])
def test_rejects_non_keys_without_writes(tmp_path, key):
    cache = SharedPackCache(
        tmp_path / "extra", pinned=tmp_path / "pinned", pack_bytes=100, margin_bytes=0
    )
    with pytest.raises(ValueError, match="layout_key"):
        cache.resolve(key, build_pack)
    assert not cache.root.exists()


def test_size_and_disk_failure_leave_no_pack(tmp_path):
    cache = SharedPackCache(
        tmp_path / "extra", pinned=tmp_path / "pinned", pack_bytes=1, margin_bytes=0
    )
    with pytest.raises(ValueError, match="exceeds admitted"):
        cache.resolve("a" * 64, build_pack)
    assert not list(cache.root.glob("*/weights.pack"))
    cache.pack_bytes = 2**100
    with pytest.raises(ValueError, match="Insufficient free disk"):
        cache.resolve("b" * 64, build_pack)
    assert not list(cache.root.glob("*/weights.pack"))


def test_read_only_missing_pack_never_creates_directory(tmp_path):
    from monolith.formats import PackLayout
    from monolith.serving.cache import ensure_pack

    root = tmp_path / "missing"
    with pytest.raises(ValueError, match="Read-only pack"):
        ensure_pack(
            "unused",
            None,
            root,
            capacity=1,
            layout=PackLayout(),
            backend="unused",
            read_only=True,
        )
    assert not root.exists()


@pytest.mark.parametrize("metadata_bytes", [2, 16, 100])
def test_metadata_uses_the_admitted_margin_without_rejecting_a_full_weight_pack(
    tmp_path, metadata_bytes
):
    cache = SharedPackCache(
        tmp_path / "extra",
        pinned=tmp_path / "pinned",
        pack_bytes=7,
        margin_bytes=metadata_bytes,
    )

    def build(path):
        (path / "weights.pack").write_bytes(b"weights")
        (path / "manifest.json").write_bytes(b" " * metadata_bytes)

    assert cache.resolve("a" * 64, build).is_dir()
