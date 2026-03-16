from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

State = Tuple[int, int]


class GridProblem:
    """2D grid pathfinding problem.

    Legend:
      S = start
      G = goal
      . = free cell with cost 1
      # = blocked cell
      digits 1-9 = traversable cell with that movement cost
      letters a-i = traversable cell with cost 1-9 respectively
    """

    def __init__(self, grid: List[List[str]]):
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.start = None
        self.goal = None
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] == "S":
                    self.start = (r, c)
                elif grid[r][c] == "G":
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("Grid must contain both 'S' and 'G'")

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "GridProblem":
        cleaned = [list(line.strip()) for line in lines if line.strip()]
        return cls(cleaned)

    def in_bounds(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_goal(self, state: State) -> bool:
        return state == self.goal

    def is_blocked(self, state: State) -> bool:
        r, c = state
        return self.grid[r][c] == "#"

    def step_cost(self, state: State) -> int:
        r, c = state
        val = self.grid[r][c]
        if val in {"S", "G", "."}:
            return 1
        if val.isdigit():
            return int(val)
        if val.isalpha() and val.islower():
            return ord(val) - ord("a") + 1
        return 1

    def neighbors(self, state: State):
        r, c = state
        directions = [
            ((r - 1, c), "UP"),
            ((r + 1, c), "DOWN"),
            ((r, c - 1), "LEFT"),
            ((r, c + 1), "RIGHT"),
        ]
        for nxt, action in directions:
            if self.in_bounds(nxt) and not self.is_blocked(nxt):
                yield nxt, action, self.step_cost(nxt)

    def overlay_path(self, path: Iterable[State]) -> List[List[str]]:
        path_set = set(path)
        result = [row[:] for row in self.grid]
        for r, c in path_set:
            if result[r][c] not in {"S", "G"}:
                result[r][c] = "*"
        return result

    def to_text(self, path: Iterable[State] | None = None) -> str:
        grid = self.overlay_path(path) if path else self.grid
        return "\n".join(" ".join(row) for row in grid)
