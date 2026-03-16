from __future__ import annotations

import heapq
import time

from search.node import Node, PrioritizedNode
from utils.heuristics import manhattan
from utils.metrics import SearchResult


def solve(problem) -> SearchResult:
    start_time = time.perf_counter()
    start = Node(problem.start)
    frontier = [PrioritizedNode(manhattan(problem.start, problem.goal), start)]
    visited = set()
    visit_order = []
    nodes_expanded = 0
    frontier_peak = 1

    while frontier:
        frontier_peak = max(frontier_peak, len(frontier))
        current = heapq.heappop(frontier).node

        if current.state in visited:
            continue

        visited.add(current.state)
        visit_order.append(current.state)
        nodes_expanded += 1

        if problem.is_goal(current.state):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                algorithm="Greedy Best-First",
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
                heapq.heappush(
                    frontier,
                    PrioritizedNode(
                        manhattan(neighbor, problem.goal),
                        Node(
                            state=neighbor,
                            parent=current,
                            action=action,
                            cost=current.cost + step_cost,
                            depth=current.depth + 1,
                        ),
                    ),
                )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult("Greedy Best-First", False, [], float("inf"), nodes_expanded, frontier_peak, elapsed_ms, visit_order)
