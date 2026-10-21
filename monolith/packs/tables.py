"""Extend checkpoint-independent F32 tables in memory, preserving packed prefixes."""

import numpy as np

__all__ = ["extended_tables"]


def extended_tables(pack, tables):
    """Return bytes for leading-axis growth only; never alter the mapped pack.

    A model owns the deterministic generator. Every old value must match bit for
    bit before its replacement is accepted. Learned auxiliaries are never inputs
    to this API. Shape/dtype changes and changed prefixes fail closed.
    """
    result = {}
    for name, (dtype, values) in tables.items():
        if name not in pack.aux:
            raise ValueError(f"Computed table missing from pack: {name}")
        record = pack.aux[name]
        array = np.asarray(values)
        shape = tuple(record["shape"])
        if array.nbytes <= record["nbytes"]:
            continue
        if (
            record.get("transform") != "table"
            or dtype != "F32"
            or record["dtype"] != "F32"
            or array.dtype != np.float32
            or len(shape) != array.ndim
            or not shape
            or shape[1:] != array.shape[1:]
            or array.shape[0] <= shape[0]
        ):
            raise ValueError(
                f"Computed table {name} can only extend F32 positions with unchanged trailing dimensions"
            )
        if array[: shape[0]].tobytes() != pack.aux_array(name).tobytes():
            raise ValueError(
                f"Computed table {name} disagrees with the packed prefix; use a compatible pack"
            )
        result[name] = array.tobytes()
    return result
