# Lazy imports to allow minimal installations
def __getattr__(name):
    if name in ["predictors", "anomaly"]:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["predictors", "anomaly"]
