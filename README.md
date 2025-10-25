# Labyrinth Pathfinder

## Introduction

This is a Python command line tool that solves a certain type of labyrinths.
It can be used to solve labyrinths from files, directories, or standard input.
It can also be used to generate random labyrinths with various parameters of depth, rooms count and hallway length.

### Task:
Given initial state, find the minimum cost of reaching the goal state, where goal state is:
- hallway: (0**hallway_length)
- rooms: room(index) = decoded(index + 1)
   - i.e. each room should be full and only contain objects of its type. 

#### Weights per one node move
```
A:    1
B:   10
C:  100
D: 1000
N: encoded(index)**10
```
  
#### Encoding
Each object is encoded as an integer starting from `1` (`0` is for empty cells).
```
0: empty cell
1: 'A'
2: 'B'
...
N: 'index + 1'
```

#### Other constraints:
- Execution time < 10 s
- Memory usage < 200 MB

## Description

<details closed>

<summary>So how does it work?</summary>

### A* Search

It uses [A* Search algorythm](https://www.redblobgames.com/pathfinding/a-star/introduction.html) to solve the labyrinths. Estimates the lowest cost to reach the goal with heuristic function. It assumes that:
   1. There are no other objects in the labyrinth;
   2. We exit/enter the closest cell in both rooms.

Results of the heuristic function are then added to the cost of the current state
to put an estimate on minimal completion cost. This estimate is then used to
order a heap containing possible moves.

f_score (best_estimate): Is a total estimate of cost so far + best approximation.\
At the start:
f_score = 0 + initial_heuristic = initial_heuristic

Afterwards:
f_score = g_score + h_score\
Which translates to:\
cost_best_possible = cost_current + cost_heuristic

### Heapq

[Heapq](https://docs.python.org/3/library/heapq.html) (Python's priority queue) is a data structure that orders heap in a binary tree. Each parent leaf has children so that: child_left < parent < child_right. This is used to order all possible moves in the A* Search algorythm. The first element of the heap (root of the tree) is always the smallest element.

Heapq is ordered by the first element. If the first element is the same, it orders by the second element, and so on. Current implementation uses move counter as the second element, so it serves as an ID.

Addition: `heapq.heappush(heap, new_entry)`  | O(n logn)\
Retrieval: `heapq.heapop(heap)`              | O(n logn)\
Peek: `heap[0]`                              | O(1)

### State representation

State is stored as its own hashable object, containg two tuples representing hallway and rooms:

```python
state = State(Tuple(int), Tuple(Tuple(int, ...), ...))
```
Hash is then pre-computed and stored for easier and faster access.

Each object is encoded as an integer starting from `1` (`0` is for empty cells).
0: empty cell
1: `A`
2: `B`
...
N: `index + 1`

Next step would be to use binary encoding, where a single integer could represent the whole state (each cell only taking three bits). It would drastically reduce memory usage and provide hashing alternative: the resulting number would be its own hash.

</details>

## Getting started

### Usage

Clone this repository and run `python run.py` with or without arguments.
```shell
// To see help, run:
> python run.py -h

// To process an input file, run:
> python run.py "input_file.txt"

// To process an entire directory, run:
> python run.py "/path/to/directory/"

// Manual input
> python run.py

// To generate random labyrinths, use any of the following options.
// These are all essentially the same:
> python run.py --generate 5
> python run.py -G 7
> python run.py --count 37
> python run.py -C 4
```

There are essentially two ways of running the program:
1. Generate and solve labyrinths with default or specified parameters.
2. Solve pre-defined labyrinth either from text, or from file, or from console input.
See more on generator options and other parameters:
- [CLI usage examples](docs/examples/cli.md)
- [Python interpeter/scripting usage examples](docs/examples/scripting.md)

### Command line arguments

### Common
#### Positional arguments:
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `INPUT_FILE` | - | path | stdin | Path to the input file or folder (optional). If neither is provided, parses the user input. Enter an empty line and use `Ctrl+D` on Linux or `Ctrl+Z` on Windows to finilize the input |

#### Optional arguments:
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--help` | `-h` | - | - | Show help |
| `--verbose` | `-v` | flag | off | Verbose output |
| `--debug` | `-d` | flag | off | Debug output |

### Generator
#### Major Generator options:
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--generate AMOUNT` | `-G` | int | - | Number of labyrinths to generate |
| `--generate_nonstandard` | `-N` | flag | off | If provided, generates labyrinths with various depths, numbers of rooms and hallway lengths |

#### Generator parameters:
*These have no effect if `--generate_nonstandard` is used, aside from the `--count` parameter.*

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--count AMOUNT` | `-C` | int | 1 | Number of labyrinths to generate. Synonymous to `--generate` |
| `--depth DEPTH` | `-D` | int | - | Room depth to generate. Implicitly defaults to 2 everywhere inside the code. This parameter does not have a default value |
| `--rooms AMOUNT` | `-R` | int | - | Number of rooms to generate. Implicitly defaults to 4 everywhere inside the code. This parameter does not have a default value |
| `--hallway_length LENGTH` | `-L` | int | 11 | Length of the hallway to generate |

##### Tests and profiler
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--profile` | `-P` | flag | off | If passed, performance profiling will be enabled |
| `--tests` | `-T` | flag | off | If passed, the program will run a pre-defined set of tests |

## Links

#### A* Search
- [RedBlobGames | A* Search: Intoduction](https://www.redblobgames.com/pathfinding/a-star/introduction.html)
- [RedBlobGames | A* Search: Implementation](https://www.redblobgames.com/pathfinding/a-star/implementation.html)

#### Python
- [Python | Priority Queue (heapq)](https://docs.python.org/3/library/heapq.html)
