from .build_sys_tools import init_mem

__all__ = ["create_mem_for_sim", "init_mem"]


def __getattr__(name):
    if name == "create_mem_for_sim":
        from .build_env import create_mem_for_sim

        return create_mem_for_sim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
