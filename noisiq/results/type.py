from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Frame:
    """
    One 'snapshot' in time for visualization / debugging.

    labels: a per-qubit label for what you want to visualize (e.g. "I", "X", "Z", "Y")
    tag: optional tag like "before", "after", "inject"
    """
    t: int
    tag: str
    labels: List[str]
    note: str = ""


@dataclass(frozen=True)
class RunResult:
    """
    Standard output of running a circuit on a backend (with optional noise).
    """
    backend: str
    seed: int
    shots: int
    frames: List[Frame]

    # optional extras (safe to ignore early on)
    stats: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None
