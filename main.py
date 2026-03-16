from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import animation

from problems.grid_problem import GridProblem
from search import astar, bfs, dfs, greedy, ucs

ALGORITHMS = {
    "bfs": bfs.solve,
    "dfs": dfs.solve,
    "ucs": ucs.solve,
    "greedy": greedy.solve,
    "astar": astar.solve,
}

DEFAULT_GRID = [
    "S...#....",
    ".##.#.2..",
    ".#..#....",
    ".#..###..",
    ".#....#..",
    ".####.#..",
    ".2....#G.",
    "........#",
]


def load_problem(grid_file: str | None) -> GridProblem:
    if grid_file is None:
        return GridProblem.from_lines(DEFAULT_GRID)
    path = Path(grid_file)
    return GridProblem.from_lines(path.read_text().splitlines())


def print_summary(result):
    print(f"\nAlgorithm: {result.algorithm}")
    print(f"Found path: {result.found}")
    print(f"Path length: {result.path_length}")
    print(f"Path cost: {result.cost}")
    print(f"Nodes expanded: {result.nodes_expanded}")
    print(f"Peak frontier size: {result.frontier_peak}")
    print(f"Elapsed time: {result.elapsed_ms:.3f} ms")


def compare_algorithms(problem: GridProblem, selected: List[str]):
    results = []
    for key in selected:
        result = ALGORITHMS[key](problem)
        results.append(result)
    print("\nComparison")
    print("-" * 86)
    print(f"{'Alg':<10}{'Found':<8}{'Len':<8}{'Cost':<10}{'Expanded':<12}{'Frontier':<12}{'Time (ms)':<10}")
    print("-" * 86)
    for r in results:
        cost_str = f"{r.cost:.1f}" if r.found else "inf"
        print(f"{r.algorithm:<10}{str(r.found):<8}{r.path_length:<8}{cost_str:<10}{r.nodes_expanded:<12}{r.frontier_peak:<12}{r.elapsed_ms:<10.3f}")
    return results


def cell_value(ch: str) -> float:
    if ch == '#':
        return -1
    if ch in {'S', 'G', '.'}:
        return 1
    if ch.isdigit():
        return int(ch)
    if ch.isalpha() and ch.islower():
        return ord(ch) - ord('a') + 1
    return 1


def draw_grid(ax, problem: GridProblem, visited=None, path=None, title="Grid"):
    visited = set(visited or [])
    path = set(path or [])
    data = [[cell_value(problem.grid[r][c]) for c in range(problem.cols)] for r in range(problem.rows)]
    ax.clear()
    ax.imshow(data, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(problem.cols))
    ax.set_yticks(range(problem.rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.set_title(title)

    for r in range(problem.rows):
        for c in range(problem.cols):
            state = (r, c)
            base = problem.grid[r][c]
            label = base
            if state in visited and base not in {'S', 'G', '#'}:
                label = '·'
            if state in path and base not in {'S', 'G'}:
                label = '*'
            ax.text(c, r, label, ha="center", va="center")


def animate_result(problem: GridProblem, result, interval: int = 250, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    frames = result.visit_order + ["PATH"]

    def update(frame):
        if frame == "PATH":
            draw_grid(ax, problem, visited=result.visit_order, path=result.path, title=f"{result.algorithm} Final Path")
        else:
            idx = result.visit_order.index(frame) + 1
            draw_grid(ax, problem, visited=result.visit_order[:idx], title=f"{result.algorithm} Search Progress")
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, repeat=False)
    if save_path:
        anim.save(save_path, writer="pillow")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Search Algorithm Visualizer")
    parser.add_argument("--algorithm", choices=list(ALGORITHMS.keys()) + ["all"], default="astar")
    parser.add_argument("--grid-file", help="Path to a text file containing the grid")
    parser.add_argument("--animate", action="store_true", help="Animate the selected algorithm")
    parser.add_argument("--save-gif", help="Optional output GIF path")
    args = parser.parse_args()

    problem = load_problem(args.grid_file)
    print("Loaded grid:\n")
    print(problem.to_text())

    if args.algorithm == "all":
        results = compare_algorithms(problem, list(ALGORITHMS.keys()))
        best = min((r for r in results if r.found), key=lambda r: r.cost, default=None)
        if best:
            print(f"\nBest cost found by: {best.algorithm}")
            print(problem.to_text(best.path))
        return

    result = ALGORITHMS[args.algorithm](problem)
    print_summary(result)
    if result.found:
        print("\nPath overlay:\n")
        print(problem.to_text(result.path))
    else:
        print("\nNo path found.")

    if args.animate:
        animate_result(problem, result, save_path=args.save_gif)


if __name__ == "__main__":
    main()
