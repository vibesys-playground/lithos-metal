from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.ir import Op, OpClass, Value
from ..registry import Registry

WILDCARD = "*"


class MetalEmitterContext(Protocol):
    """Stable services available to a package-owned native Metal emitter.

    Buffer slices, parameter ownership, kernel construction and declared writes
    remain compiler-owned. Emitted generation operations bind StepState even
    when their compiled row count is static.
    """

    def shape(self, value: Value) -> tuple[int, ...]: ...
    def rows_of(self, op: Op) -> tuple[int, int]: ...
    def buf(self, value: Value) -> tuple[str, int]: ...
    def step_state_buffer(self) -> tuple[str, int]: ...
    def params(self, name: str, data: bytes) -> str: ...
    def kernel(
        self,
        key: str,
        source: str,
        function: str,
        macros: dict[str, str],
        language_version: int = 0,
    ) -> str: ...
    def add(
        self,
        kernel: str,
        bindings: list[tuple[int, str, int]],
        grid: tuple[int, int, int],
        tg: tuple[int, int, int],
        name: str,
        *,
        writes: Sequence[int] | None = None,
        **metadata: Any,
    ) -> None: ...


@dataclass(frozen=True)
class KernelBinding:
    """Which MSL kernel implements an op on a profile, with its function constants and threadgroups per core."""

    kernel: str
    function_constants: Mapping[str, Any] = field(default_factory=dict)
    threadgroups_per_core: int = 1


@dataclass(frozen=True)
class CostModel:
    """Per-block cost the compiler and the autotuner reason with (bytes streamed, ALU work, ALU-bound flag)."""

    bytes_per_block: int = 0
    flops_per_block: int = 0
    alu_bound: bool = False


@dataclass
class OpDef:
    """An op kind: its class, the kind of block domain it partitions over, its cost, and its kernel bindings."""

    name: str
    klass: OpClass
    domain_kind: str
    cost: CostModel = field(default_factory=CostModel)
    kernels: dict[str, KernelBinding] = field(
        default_factory=dict
    )  # profile key -> binding, "*" = any

    # Optional package-owned Metal emitter; receives the compiler context and typed op.
    metal_emitter: Callable[[MetalEmitterContext, Op], None] | None = None
    # Target KV capacity is the leading dimension of this input, when present.
    cache_input: int | None = None

    def binding_for(self, profile_key: str) -> KernelBinding | None:
        return self.kernels.get(profile_key) or self.kernels.get(WILDCARD)

    def bind(self, profile_key: str, binding: KernelBinding) -> OpDef:
        if profile_key in self.kernels and self.kernels[profile_key] != binding:
            raise ValueError(
                f"op {self.name}: profile {profile_key!r} already bound to {self.kernels[profile_key]}"
            )
        self.kernels[profile_key] = binding
        return self


OPS: Registry[OpDef] = Registry("op")


def register_op(opdef: OpDef) -> OpDef:
    return OPS.register(opdef.name, opdef)
