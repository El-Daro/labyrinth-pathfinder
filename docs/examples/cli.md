### Usage examples

Run a set of pre-defined tests and enable profiler and verbose output.

```shell
> python .\run.py -v --tests --profiler
```

Generate and solve two different labyrinths with default parameters.
Internally, defaults are: depth: 2, rooms: 4, hallway_length: 11.

```shell
> python .\run.py -d --generate 2
```

Generate and solve five different labyrinths with 3 rooms, default
depth (2) and length of the hallway equal to 15 with debug output.\
NOTE: depth = 2 is an internal default value of various classes, not
a default state of the corresponding command line argument.\
CAUTION: Big labyrinths may take a very long time to process.

```shell
> python .\run.py -d --generate 5 --hallway_length 15 --rooms 3
```

Certain rules and limits are still applied to labyrinth generation.\
All three major parameters must always be in the allowed range.\
Hallway is always generated empty with length in the allwoed range of 5 to 19.\
Rooms are filled at random. The number of rooms is calculated\
based on the hallway length, if not provided as an argument.\
Depth range: 2..4.

This will result in failure, because the `hallway_length` is out of range.

```shell
> python .\run.py -d --hallway_length 37 --rooms 11
```

This, however, will pass and generate+solve one labyrinth (count=1 by default):

```shell
> python .\run.py -d --hallway_length 15
```

Generate and solve 10 different, non-standard labyrinths, each with 
an arbitrary number of rooms, depth and hallway length.\
Initial parameters  for each labyrint are read from a pre-defined dictionary.

```shell
> python .\run.py --generate_nonstandard --count 10
```

CAUTION: Big labyrinths may take a very long time to process.

```shell
> python .\run.py -d --count 10 --rooms = 8 --depth 4 --hallway_length 19 
```