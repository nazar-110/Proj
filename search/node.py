from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

State = Tuple[int, int]


@dataclass(order=True)
class PrioritizedNode:
    priority: float
    node: "Node" = field(compare=False)


@dataclass
class Node:
    state: State
    parent: Optional["Node"] = None
    action: Optional[str] = None
    cost: float = 0.0
    depth: int = 0

    def path(self):
        current = self
        result = []
        while current is not None:
            result.append(current.state)
            current = current.parent
        return list(reversed(result))
