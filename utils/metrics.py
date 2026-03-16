from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

State = Tuple[int, int]


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    path: List[State]
    cost: float
    nodes_expanded: int
    frontier_peak: int
    elapsed_ms: float
    visit_order: List[State]

    @property
    def path_length(self) -> int:
        return max(0, len(self.path) - 1)
