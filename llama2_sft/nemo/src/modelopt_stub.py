"""
Stub for nvidia-modelopt package which is not available on ROCm.

NeMo v2.6.0 has deep imports of modelopt in core model code paths.
This stub intercepts all modelopt.* imports and provides dummy modules
so NeMo can proceed without the actual package.

Import this module BEFORE any NeMo imports:
    import modelopt_stub  # noqa: F401
"""
import sys
import types


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


class _ModeloptStubFinder:
    """Meta path finder that intercepts all modelopt.* imports with stub modules."""
    def find_module(self, fullname, path=None):
        if fullname == 'modelopt' or fullname.startswith('modelopt.'):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _StubModule(fullname)
        mod.__loader__ = self
        mod.__path__ = []
        mod.__package__ = fullname
        mod.__file__ = f"<modelopt_stub:{fullname}>"
        sys.modules[fullname] = mod
        return mod


sys.meta_path.insert(0, _ModeloptStubFinder())
