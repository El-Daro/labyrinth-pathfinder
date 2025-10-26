### Usage examples

Simple usage:

```python
>>> input_path = "labyrinth.txt"
>>> labyrinth = Labyrinth(input_path)
>>> labyrinth.solve()
296
```

The Labyrinth class constructor accepts various types of input, including:
- Text file
- String
- State object: `State: Tuple(hallway = Tuple(), rooms = Tuple(Tuple(int, int)))`

```python
>>> from src.pathfinder import *
>>> input_path = "labyrinth.txt"
>>> labyrinth_from_text = Labyrinth(input_path)
>>> file = open(input_path).read()
>>> str(labyrinth_from_text) == file
True
>>> labyrinth_from_file = Labyrinth(file)
>>> str(labyrinth_from_file) == file
True
```

You can set up a state manually. An object is represented as an integer, where
resident = room_index + 1:

```python
>>> from src.pathfinder import *
>>> hallway = tuple([0] * 11)
>>> rooms = tuple((2, 1), (3, 4), (2, 3), (4, 1))
>>> state = State(hallway, rooms)
>>> lab = Labyrinth(state)
>>> print(lab)

   #############
   #...........#
   ###B#C#B#D###
     #A#D#C#A#
     #########
```

#### Generator:
```python
>>> generator = Generator(rooms = 4, depth = 4, hallway_length = 11)
>>> labyrinth = generator.new_labyrinth()
>>> print(labyrinth)

   #############
   #...........#
   ###A#C#C#D###
     #C#D#A#D#
     #B#B#B#C#
     #A#B#A#D#
     #########

>>> gen.depth = 2
>>> labyrinth = generator.new_labyrinth()
>>> print(labyrinth)

   #############
   #...........#
   ###A#B#C#C###
     #B#D#D#A#
     #########

>>> labyrinth

Hallway: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
Rooms: ((1, 2), (2, 4), (3, 4), (3, 1))

>>> labyrinth = generator.new_labyrinth(rooms = 2, depth = 4, hallway_length = 7)
>>> print(labyrinth)

   #########
   #.......#
   ###B#A###
     #B#B#
     #A#B#
     #A#A#
     #####

>>> labyrinth

Hallway: (0, 0, 0, 0, 0, 0, 0)
Rooms: ((3, 2, 1, 1), (4, 2, 3, 4))

# Retrieve the last generated state:
>>> last_state = generator.state_history[-1]
>>> labyrinth_last = Labyrinth(last_state)
>>> labyrinth_last
Hallway: (0, 0, 0, 0, 0, 0, 0)
Rooms: ((3, 2, 1, 1), (4, 2, 3, 4))
>>> labyrinth_last.state_current == labyrinth.state_current
True
>>> labyrinth_last == labyrinth
True
```