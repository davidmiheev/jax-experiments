"""Load PyTorch weights (as a dict of NumPy arrays) into the NNX transformer.

Takes NumPy rather than torch tensors so that nothing here imports torch: the
caller does ``{k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}``.

Two conventions differ between the frameworks:

- ``torch.nn.Linear`` stores its weight as [out, in] and computes x W^T;
  ``nnx.Linear`` stores ``kernel`` as [in, out] and computes x K. So K = W^T.
  Loaded the wrong way round, a non-square layer is a shape error, but a square
  one (the attention output projection) is silently a different function.
- Names: LayerNorm ``weight`` is Flax's ``scale``, Embedding ``weight`` is
  ``embedding``, and C3's ``mlp`` Sequential indices 0 and 2 are ``up``/``down``.

Loading is strict both ways: every PyTorch tensor must land somewhere, and
every NNX parameter must be assigned.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

_TORCH_TO_NNX = (("mlp.0", "mlp.up"), ("mlp.2", "mlp.down"))


def _nnx_path(torch_path: str) -> str:
    for old, new in _TORCH_TO_NNX:
        torch_path = torch_path.replace(old, new)
    return torch_path


def _torch_path(nnx_path: str) -> str:
    for old, new in _TORCH_TO_NNX:
        nnx_path = nnx_path.replace(new, old)
    return nnx_path


def _attr(kind: str, leaf: str) -> str:
    if leaf == "bias":
        return "bias"
    return {"LayerNorm": "scale", "Embed": "embedding"}.get(kind, "kernel")


def _module(root: nnx.Module, path: str) -> nnx.Module:
    module = root
    for part in path.split(".") if path else ():
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def load_torch_state(model: nnx.Module, state: dict[str, np.ndarray], *, transpose_linear: bool = True) -> None:
    """Copy ``state`` (a torch ``state_dict`` as NumPy) into ``model`` in place.

    ``transpose_linear=False`` reproduces the orientation bug on purpose (R1
    measures it); it only applies to square weights, the case it is silent.
    """
    assigned: set[tuple[str, str]] = set()
    for name, value in state.items():
        path, leaf = name.rsplit(".", 1)
        module = _module(model, _nnx_path(path))
        kind = type(module).__name__
        if kind not in ("Linear", "LayerNorm", "Embed"):
            raise KeyError(f"{name}: maps to a {kind}, not a parameterised layer")
        attr = _attr(kind, leaf)
        value = np.asarray(value, dtype=np.float32)
        if kind == "Linear" and attr == "kernel":
            square = value.shape[0] == value.shape[1]
            value = value.T if (transpose_linear or not square) else value
        param = getattr(module, attr)
        if param.get_value().shape != value.shape:
            raise ValueError(f"{name}: torch {value.shape} vs nnx {param.get_value().shape}")
        param.set_value(jnp.asarray(value))
        assigned.add((path, attr))

    expected = {(".".join(map(str, p[:-1])), str(p[-1])) for p, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    expected = {(_torch_path(p), a) for p, a in expected}
    missing = expected - assigned
    if missing:
        raise KeyError(f"nnx parameters left unassigned: {sorted(missing)}")


def numpy_state(module) -> dict[str, np.ndarray]:
    """``module.state_dict()`` as float32 NumPy (torch passed in, not imported)."""
    return {k: v.detach().cpu().float().numpy() for k, v in module.state_dict().items()}
