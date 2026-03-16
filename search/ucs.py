from __future__ import annotations

import heapq
import time

from search.node import Node, PrioritizedNode
from utils.metrics import SearchResult


def solve(problem) -> SearchResult:
    start_time = time.perf_counter()
    start = Node(problem.start)
    frontier = [PrioritizedNode(0, start)]
    best_cost = {problem.start: 0}
    visit_order = []
    nodes_expanded = 0
    frontier_peak = 1

    while frontier:
        frontier_peak = max(frontier_peak, len(frontier))
        current = heapq.heappop(frontier).node

        if current.cost > best_cost.get(current.state, float("inf")):
            continue

        visit_order.append(current.state)
        nodes_expanded += 1

        if problem.is_goal(current.state):
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                algorithm="UCS",
                found=True,
                path=current.path(),
                cost=current.cost,
                nodes_expanded=nodes_expanded,
                frontier_peak=frontier_peak,
                elapsed_ms=elapsed_ms,
                visit_order=visit_order,
            )

        for neighbor, action, step_cost in problem.neighbors(current.state):
            new_cost = current.cost + step_cost
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                heapq.heappush(
                    frontier,
                    PrioritizedNode(
                        new_cost,
                        Node(
                            state=neighbor,
                            parent=current,
                            action=action,
                            cost=new_cost,
                            depth=current.depth + 1,
                        ),
                    ),
                )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult("UCS", False, [], float("inf"), nodes_expanded, frontier_peak, elapsed_ms, visit_order)
