from __future__ import annotations

import time
from collections import deque

from search.node import Node
from utils.metrics import SearchResult


def solve(problem) -> SearchResult:
    start_time = time.perf_counter()
    start = Node(problem.start)
    frontier = deque([start])
    visited = {problem.start}
    visit_order = []
    nodes_expanded = 0
    frontier_peak = 1

    while frontier:
        frontier_peak = max(frontier_peak, len(frontier))
        current = frontier.popleft()
        visit_order.append(current.state)
        nodes_expanded += 1

        if problem.is_goal(current.state):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                algorithm="BFS",
                found=True,
                path=current.path(),
                cost=current.cost,
                nodes_expanded=nodes_expanded,
                frontier_peak=frontier_peak,
                elapsed_ms=elapsed_ms,
                visit_order=visit_order,
            )

        for neighbor, action, step_cost in problem.neighbors(current.state):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(
                    Node(
                        state=neighbor,
                        parent=current,
                        action=action,
                        cost=current.cost + step_cost,
                        depth=current.depth + 1,
                    )
                )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult("BFS", False, [], float("inf"), nodes_expanded, frontier_peak, elapsed_ms, visit_order)
