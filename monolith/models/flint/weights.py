"""Checkpoint byte views: U8-packed MLX words and stacked routed experts.

No dequantization or re-quantization occurs here. Each expert becomes an
ordinary matrix view for the existing shared Experts/slab pack mechanism.
"""

from dataclasses import replace

from ...formats.safetensors_reader import SafetensorsDir
from ...nn.pack_plan import bind_formats, dequantized_tensors


def checkpoint_views(info):
    name = info.name
    if ".mlp.switch_mlp." in name:
        if len(info.shape) != 3:
            raise ValueError(f"{name}: expected stacked [experts, rows, columns]")
        count = info.shape[0]
        stride, remainder = divmod(info.end - info.start, count)
        if remainder:
            raise ValueError(f"{name}: expert byte size does not divide storage")
        views = [
            replace(
                info,
                name=name.replace(".mlp.switch_mlp.", f".mlp.experts.{e}."),
                shape=info.shape[1:],
                start=info.start + e * stride,
                end=info.start + (e + 1) * stride,
            )
            for e in range(count)
        ]
    else:
        views = [info]
    for view in views:
        if view.name.endswith(".weight") and view.dtype == "U8":
            if view.shape[-1] % 4:
                raise ValueError(
                    f"{view.name}: packed byte width must divide into uint32 words"
                )
            view = replace(
                view, dtype="U32", shape=(*view.shape[:-1], view.shape[-1] // 4)
            )
        if view.name == "model.embed_tokens.weight":
            view = replace(view, format="int8_affine")
        yield view


def bind_checkpoint_formats(model, path):
    checkpoint = SafetensorsDir(path, views=checkpoint_views)
    try:
        formats = bind_formats(model, checkpoint)
        # U32 affine storage is ambiguous without the model's declared bit width.
        model.embed_tokens.set_format("weight", "int8_affine")
        formats["model.embed_tokens.weight"] = "int8_affine"
        weight_map = model.full_weight_map()
        expected = set(weight_map)
        actual = set(checkpoint.names())
        for name in expected:
            spec = weight_map[name][2]
            info = checkpoint.info(name)
            expected_dtype = "BF16" if spec.aux or ".mlp.gate.weight" in name else "U32"
            if info.dtype != expected_dtype:
                raise ValueError(
                    f"{name}: expected storage dtype {expected_dtype}, got {info.dtype}"
                )
            if spec.aux:
                if info.shape != spec.shape:
                    raise ValueError(f"{name}: expected {spec.shape}, got {info.shape}")
                continue
            bits = 8 if name == "model.embed_tokens.weight" else 4
            if info.dtype == "BF16":
                logical = info.shape
            elif info.dtype == "U32":
                logical = (*info.shape[:-1], info.shape[-1] * (32 // bits))
                for side in ("scales", "biases"):
                    side_name = name[:-6] + side
                    if checkpoint.info(side_name).shape != (
                        spec.shape[0],
                        spec.shape[1] // 64,
                    ):
                        raise ValueError(
                            f"{side_name}: incompatible group_size or shape"
                        )
                    actual.discard(side_name)
            else:
                raise ValueError(f"{name}: unsupported storage dtype {info.dtype}")
            if logical != spec.shape:
                raise ValueError(f"{name}: expected {spec.shape}, got {logical}")
        unknown = actual - expected
        if unknown:
            raise ValueError(f"unknown checkpoint tensor {min(unknown)}")
        return formats
    finally:
        checkpoint.close()


def load_oracle(model, path, *, device=None):
    import numpy as np
    import torch

    checkpoint = SafetensorsDir(path, views=checkpoint_views)
    try:

        def stream():
            for name, array, storage in dequantized_tensors(model, checkpoint):
                # Quantized matmul accumulates the affine grid in FP32. Only the
                # embedding gather rounds its decoded rows before consumption.
                dtype = torch.float32 if storage == "int4_affine" else torch.bfloat16
                tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
                yield name, tensor.to(dtype=dtype, device=device)

        model.load_weights(stream(), strict=True)
    finally:
        checkpoint.close()
