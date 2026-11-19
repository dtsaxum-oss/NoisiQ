"""Simulation results and propagation traces."""

from .result import Estimate, Result
from .trace import FaultEvent, PropagationTrace

__all__ = ["Result", "Estimate", "PropagationTrace", "FaultEvent"]
