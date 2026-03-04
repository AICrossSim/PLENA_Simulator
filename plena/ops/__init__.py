"""
plena.ops — Convenience dispatch API for PLENA operators.

After OpRegistry.load() is called, every registered operator is available
as a module-level function:

    import plena.ops as ops
    result = ops.softmax(prog, X_batch, scale=1.0)

The active backend is controlled by OpRegistry.set_backend(Backend.CPU)
or OpRegistry.set_backend(Backend.PLENA).
"""
from plena.ops.registry import OpRegistry, Backend  # noqa: F401


def _make_dispatch_fn(op_name: str):
    """Create a module-level function dispatching through the registry."""
    def fn(*args, **kwargs):
        return OpRegistry.get().dispatch(op_name, *args, **kwargs)
    fn.__name__ = op_name
    fn.__qualname__ = op_name
    fn.__doc__ = f"Dispatch plena op '{op_name}' to the active backend."
    return fn


# Static exports (for IDE completion; also dynamically populated after load)
softmax = _make_dispatch_fn("softmax")
linear = _make_dispatch_fn("linear")
rms_norm = _make_dispatch_fn("rms_norm")
layer_norm = _make_dispatch_fn("layer_norm")
ffn = _make_dispatch_fn("ffn")
flash_attention = _make_dispatch_fn("flash_attention")
conv2d = _make_dispatch_fn("conv2d")
embedding_add = _make_dispatch_fn("embedding_add")
rope = _make_dispatch_fn("rope")
