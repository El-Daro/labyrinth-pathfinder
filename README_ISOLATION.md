# Virus:Isolation

## Introduction

This is a Python command line tool that emulates an Alien:Isolation inspired game.
It can be used with files, directories, or standard input.

### Task:
Given a graph defined as edges between nodes, find the correct sequence of edges to sever in order to prevent the virus from reaching the gateways.

- The Virus finds the closest gateway and moves towards it.
- The system acts by severing the edge to a chosen gateway.
  - The gateway is chosen based on the thorough analysis of the graph with multiple simulated steps;
  - Each such step is a BFS Early Exit search from a possible position of the virus to the nearest gateways.
- The cycle repeats until there are no possible moves left.
*Note that at the very first step the system acts first, so, technically, the virus does the search without moving at step 1.*

#### Input example
```
a-b
a-c
b-d
b-A
c-f
d-e
d-B
e-f
f-C
```

#### Visual representation
```
     A   B
     |   |
 a---b---d
 |       |
 c---f---e
     |
     C
```

#### Correct output
```
A-b
B-d
C-f
```

## Description

<details closed>

<summary>So how does it work?</summary>

### BFS Search

This program uses [BFS Search algorythm with early exit](https://www.redblobgames.com/pathfinding/early-exit/) to find the closest gateway.

Results of the search are then processed and analyzed in order to find the priority target (closest gateway, chosen alphabetically in case of a tie), the priority path to it (again, alphabetically) and the distance.

The moves are simulated in advance — each with its own BFS search, — and the edges are severed in the order that prevents the virus from reaching any gateways.

</details>

## Getting started

### Usage

Clone this repository and run `python run2.py` with or without arguments.
```shell
// To see help, run:
> python run2.py -h

// To process an input file, run:
> python run2.py "input_file.txt"

// To process an entire directory, run:
> python run2.py "/path/to/directory/"

// Manual input
> python run2.py

// To see the steps history use the `--verbose` flag:
> python run2.py --verbose
> python run2.py -v

// To see the steps history in all their glory use the `--colored` flag:
> python run2.py -v --colored
> python run2.py -v -C

// To run pre-defined tests use the `--tests` flag:
> python run2.py -v --profiler -C --tests

// These are the examples presented in the original task description:
> python run2.py -v --profiler -C --tests --option EXAMPLE

// These are the examples in the \graphs\tests\ directory
> python run2.py -v --profiler -C --tests --option FROM_DIR

// And from a specific file with example (pre-defined) 
> python run2.py -v --profiler -C --tests --option FROM_FILE
```

## Command line arguments

### Common
### Positional arguments:
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `INPUT_FILE` | - | path | stdin | Path to the input file or folder (optional). If neither is provided, parses the user input. Enter an empty line and use `Ctrl+D` on Linux or `Ctrl+Z` on Windows to finilize the input |

### Optional arguments:
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--help` | `-h` | - | - | Show help |
| `--verbose` | `-v` | flag | off | Verbose output |
| `--debug` | `-d` | flag | off | Debug output |
| `--colored` | `-C` | flag | off | Colored output |

### Tests and profiler
| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--profile` | `-P` | flag | off | If passed, performance profiling will be enabled |
| `--tests` | `-T` | flag | off | If passed, the program will run a pre-defined set of tests |
| `--option OPTION` | `-O` | enum/str | DEFAULT | Defines what specific tests to run. Possible values: `EXAMPLE`, `FROM_DIR`, `FROM_FILE` |
| | | | | 'DEFAULT': A single pre-defined test / 'EXAMPLE': Pre-defined examples fromt he original task description / 'FROM_DIR': Tests from the \graphs\tests\ directory / 'FROM_FILE': A specific file with example (pre-defined) |

## Links

#### BFS Search
- [RedBlobGames | BFS Search algorythm](https://www.redblobgames.com/pathfinding/tower-defense/#diagram-parents)
- [RedBlobGames | BFS Search algorythm with early exit](https://www.redblobgames.com/pathfinding/early-exit/)