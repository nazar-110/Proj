# Search Algorithm Visualizer

A complete Python project for comparing classical search algorithms on a 2D grid pathfinding problem.

## Algorithms included
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform Cost Search (UCS)
- Greedy Best-First Search
- A* Search

## Features
- Weighted and unweighted grids
- Text-based comparison mode
- Matplotlib animation mode
- Clean project structure for GitHub and internship portfolios
- Metrics: path cost, path length, nodes expanded, frontier peak, runtime

## Grid legend
- `S` = start
- `G` = goal
- `.` = open cell with cost 1
- `#` = blocked cell
- `1-9` = traversable weighted cells

## Install
```bash
pip install -r requirements.txt
```

## Run a single algorithm
```bash
python main.py --algorithm astar
python main.py --algorithm bfs
python main.py --algorithm ucs
```

## Compare all algorithms
```bash
python main.py --algorithm all
```

## Animate the search
```bash
python main.py --algorithm astar --animate
```

## Save a GIF
```bash
python main.py --algorithm astar --animate --save-gif astar.gif
```

## Use your own grid file
Create a text file like this:

```text
S...#....
.##.#.2..
.#..#....
.#..###..
.#....#..
.####.#..
.2....#G.
........#
```

Then run:
```bash
python main.py --algorithm astar --grid-file examples/grid1.txt
```

## Why this is good for internships
This project shows:
- algorithm implementation skills
- data structures knowledge
- clean modular software design
- experimentation and benchmarking
- visualization and presentation skills

## Good resume bullet
Built a Python search algorithm visualizer implementing BFS, DFS, UCS, Greedy Best-First Search, and A* for weighted grid pathfinding, with runtime, frontier, and node-expansion metrics plus animated matplotlib visualizations.
