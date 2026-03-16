from __future__ import annotations

import time

from search.node import Node
from utils.metrics import SearchResult


def solve(problem) -> SearchResult:
    start_time = time.perf_counter()
    start = Node(problem.start)
    frontier = [start]
    visited = set()
    frontier_states = {problem.start}
    visit_order = []
    nodes_expanded = 0
    frontier_peak = 1

    while frontier:
        frontier_peak = max(frontier_peak, len(frontier))
        current = frontier.pop()
        frontier_states.discard(current.state)

        if current.state in visited:
            continue

        visited.add(current.state)
        visit_order.append(current.state)
        nodes_expanded += 1

        if problem.is_goal(current.state):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                algorithm="DFS",
                found=True,
                path=current.path(),
                cost=current.cost,
                nodes_expanded=nodes_expanded,
                frontier_peak=frontier_peak,
                elapsed_ms=elapsed_ms,
                visit_order=visit_order,
            )

        neighbors = list(problem.neighbors(current.state))
        for neighbor, action, step_cost in reversed(neighbors):
            if neighbor not in visited and neighbor not in frontier_states:
                frontier.append(
                    Node(
                        state=neighbor,
                        parent=current,
                        action=action,
                        cost=current.cost + step_cost,
                        depth=current.depth + 1,
                    )
                )
                frontier_states.add(neighbor)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult("DFS", False, [], float("inf"), nodes_expanded, frontier_peak, elapsed_ms, visit_order)
