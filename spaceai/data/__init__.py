def __getattr__(name):
    if name == "AnomalyDataset":
        from .anomaly_dataset import AnomalyDataset
        return AnomalyDataset
    if name == "NASA":
        from .nasa import NASA
        return NASA
    if name == "ESA":
        from .esa import ESA
        return ESA
    if name == "ESAMissions":
        from .esa import ESAMissions
        return ESAMissions
    if name == "ESAMission":
        from .esa import ESAMission
        return ESAMission
    if name == "OPSSAT":
        from .ops_sat import OPSSAT
        return OPSSAT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["AnomalyDataset", "NASA", "ESA", "ESAMissions", "ESAMission", "OPSSAT"]
