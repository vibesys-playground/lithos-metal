"""The model's public forward oracle reads the same exact affine weight grid."""

import numpy as np
import pytest

from monolith.models.flint import FlintModel, load_oracle
from tests.contract.test_flint_package import write_checkpoint

torch = pytest.importorskip("torch")


def test_affine_weights_stay_fp32_until_the_projection_rounds(tmp_path):
    write_checkpoint(tmp_path)
    model = FlintModel.from_checkpoint(tmp_path, max_context=16)
    load_oracle(model, tmp_path)
    block = model.layers()[1]
    router = block.mlp.router
    assert router.param("gate").dtype == torch.bfloat16
    projection = block.mixer.qkv
    weights = [projection.param(part.local) for part in projection.parts]
    assert all(weight.dtype == torch.float32 for weight in weights)
    assert any(
        torch.any(weight != weight.bfloat16().float()).item() for weight in weights
    )
    rng = np.random.default_rng(45)
    x = torch.from_numpy(rng.normal(size=(2, 256)).astype(np.float32)).bfloat16()
    expected = x.float() @ torch.cat(weights).t()
    assert torch.equal(projection.forward(x), expected.bfloat16())
    state = {
        entry.name: torch.zeros(entry.shape, dtype=torch.bfloat16)
        for entry in model.state_spec().entries
    }
    logits, hidden, updated = model.forward(torch.tensor([1, 2]), state, 0)
    assert logits.shape == (2, 256) and logits.dtype == torch.bfloat16
    assert len(hidden) == 4 and updated is state
    assert torch.isfinite(logits).all()
    # The next call advances the same caches rather than recomputing a prefix.
    next_logits, _, _ = model.forward(torch.tensor([3]), state, 2)
    assert next_logits.shape == (1, 256) and torch.isfinite(next_logits).all()
