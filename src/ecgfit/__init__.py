"""Joint beat-morphology and baseline estimation for multi-lead ECG."""
from .fit import FitResult, fit

__all__ = ["fit", "FitResult"]
