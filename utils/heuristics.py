from __future__ import annotations

from typing import Tuple

State = Tuple[int, int]


def manhattan(state: State, goal: State) -> int:
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def euclidean(state: State, goal: State) -> float:
    return ((state[0] - goal[0]) ** 2 + (state[1] - goal[1]) ** 2) ** 0.5
