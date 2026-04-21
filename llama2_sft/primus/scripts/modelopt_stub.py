"""
Import-compatibility shim for Megatron-Bridge checkpoint conversion on ROCm.

Handles two classes of import failures:
  1. modelopt.* -- broken torch.onnx._type_utils in newer PyTorch/ROCm.
     Fixed via a meta-path finder that stubs all modelopt.* imports.
  2. Megatron-Bridge eagerly imports ALL model bridges (GLM-VL, Qwen-VL, …)
     even when only the Llama/GPT bridge is needed.  Some of these bridges
     require transformers classes/submodules that don't exist in the installed
     version.  Fixed via a builtins.__import__ hook that catches ImportError
     from megatron.bridge.models.__init__ when loading bridge subpackages,
     replacing the failed subpackage with a stub module.

Usage -- import this module BEFORE any Megatron-Bridge / NeMo imports:
    import modelopt_stub  # noqa: F401
"""
import builtins
import importlib.machinery
import sys
import types


# ---------------------------------------------------------------------------
# Universal stub objects
# ---------------------------------------------------------------------------

class _StubMeta(type):
    """Metaclass so _Stub works both as a class and as an attribute namespace."""
    def __getattr__(cls, name):
        return _Stub
    def __call__(cls, *a, **kw):
        return object.__new__(cls)
    def __bool__(cls):
        return False
    def __instancecheck__(cls, instance):
        return True


class _Stub(metaclass=_StubMeta):
    """Universal stub: usable as base class, callable, attribute-absorbent."""
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _Stub()
    def __getattr__(self, name): return _Stub
    def __bool__(self): return False
    def __iter__(self): return iter([])
    def __len__(self): return 0
    def __str__(self): return ""
    def __repr__(self): return "<modelopt stub>"

    @staticmethod
    def is_converted(model):
        return False


_DUNDER_PASSTHROUGH = frozenset({
    '__name__', '__loader__', '__package__', '__spec__', '__path__',
    '__file__', '__cached__', '__builtins__', '__doc__',
})


class _StubModule(types.ModuleType):
    """Module that returns _Stub for non-dunder attribute access."""
    def __getattr__(self, name):
        if name in _DUNDER_PASSTHROUGH:
            raise AttributeError(name)
        return _Stub


# ---------------------------------------------------------------------------
# Fix 1: meta-path finder that intercepts all modelopt.* imports
# ---------------------------------------------------------------------------

class _StubLoader:
    def create_module(self, spec):
        mod = _StubModule(spec.name)
        mod.__loader__ = self
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__file__ = f"<modelopt_stub:{spec.name}>"
        return mod

    def exec_module(self, module):
        pass


class _ModeloptStubFinder:
    """Always intercept modelopt.* with stubs."""
    def find_spec(self, fullname, path, target=None):
        if fullname == 'modelopt' or fullname.startswith('modelopt.'):
            return importlib.machinery.ModuleSpec(
                fullname, _StubLoader(), is_package=True,
            )
        return None


sys.meta_path.insert(0, _ModeloptStubFinder())


# ---------------------------------------------------------------------------
# Fix 2: builtins.__import__ hook for broken bridge model subpackages
# ---------------------------------------------------------------------------
# megatron.bridge.models.__init__ eagerly does:
#   from megatron.bridge.models.glm_vl import (…)
#   from megatron.bridge.models.qwen_vl import (…)
#   …etc…
# Some of these fail because they need transformers classes/submodules that
# don't exist in the installed version.  We catch the ImportError at the
# top level (caller == megatron.bridge.models) and replace the failed
# subpackage with a _StubModule so the remaining imports can proceed.
# This never touches the transformers package itself.

_orig_import = builtins.__import__


def _safe_bridge_import(name, globals=None, locals=None, fromlist=(), level=0):
    try:
        return _orig_import(name, globals, locals, fromlist, level)
    except (ImportError, ModuleNotFoundError):
        caller = (globals or {}).get('__name__', '')
        if (caller == 'megatron.bridge.models'
                and name.startswith('megatron.bridge.models.')):
            mod = _StubModule(name)
            mod.__path__ = []
            mod.__package__ = name
            mod.__file__ = f"<bridge_stub:{name}>"
            sys.modules[name] = mod
            return mod
        raise


builtins.__import__ = _safe_bridge_import
