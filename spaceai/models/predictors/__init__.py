def __getattr__(name):
    if name == "SequenceModel":
        from .seq_model import SequenceModel
        return SequenceModel
    if name == "LSTM":
        from .lstm import LSTM
        return LSTM
    if name == "ESN":
        from .esn import ESN
        return ESN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["SequenceModel", "LSTM", "ESN"]
