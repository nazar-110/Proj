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
                algorithm="A*",
                found=True,
                path=current.path(),
                cost=current.cost,
                nodes_expanded=nodes_expanded,
                frontier_peak=frontier_peak,
                elapsed_ms=elapsed_ms,
                visit_order=visit_order,
            )

        for neighbor, action, step_cost in problem.neighbors(current.state):
            g = current.cost + step_cost
            if g < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = g
                f = g + manhattan(neighbor, problem.goal)
                heapq.heappush(
                    frontier,
                    PrioritizedNode(
                        f,
                        Node(
                            state=neighbor,
                            parent=current,
                            action=action,
                            cost=g,
                            depth=current.depth + 1,
                        ),
                    ),
                )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return SearchResult("A*", False, [], float("inf"), nodes_expanded, frontier_peak, elapsed_ms, visit_order)
