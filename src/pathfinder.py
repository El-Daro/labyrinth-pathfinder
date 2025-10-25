#pylint:disable=W0312

#---------------------------------------------------------------#
# Version: 0.9.9												#
# Labyrinth Pathfinder											#
# Assuming hash can hash cache, how many cache hashes hash has? #
#---------------------------------------------------------------#

#---------------------------------------------------------------
# 25.10.22
# Added decoding from compacted state to readable and graphical formats

# 25.10.23
# - Finalized input processing and parsing
# - Decided on the state representation (State class)
# - Changed the processing rules to accomodate any valid initial state
# - Added simple test functions
# - Added main method of testing: put your labyrinths as files in the ..\labyrinths folder
# - Implemented user interface
# - Refactored UI to a separate file
# - Implemented labyrinth generator
# - Added support for different room sizes
# - Added support for different hallway lengths
# - Added support for different start positions
# - Added support for different room depth

# 24.10.25
# - Fixed Generator
# - Improved generator
# - Added history of generated states
# - Fixed generator once again
# - Added generator tests
# - Implemented goal state
# Slept a bit
# - Added room positions
# - Improved calculation of accessible hallway nodes
# - SOLVER:
#   - Logic for pathfinding and cost calculation
#   - Implemented solve() method
#   - Implemented get_possible_moves() method
#   - Implemented _moves_room_to_hallway() method
#   - Implemented _moves_room_to_room() method
#   - Implemented _try_move_to_destination() method
#   - Implemented _is_path_clear() method
#   - Modified _moves_room_to_hallway() method
#   - Implemented _moves_hallway_to_room() method
#
#   - Fixed energy cost computation and heuristics
#   - Fixed move generation and search logic
#   - It works
#
# 25.10.25
#   - Added profiler
#   - Added the simplest way to measure execution time (aside from profiler)
#   - Fixed some visual representation
#   - Cleaned up the code, deleted obsolete stuff
#	- Fixed various issues
#   - Implemented command line arguments:
#
#		positional arguments:
#   		input_path				| Path to the input file or folder
# 
# 		options:
#   		-h, --help				| show this help message and exit
#			-d, --debug				| Debug output
#			-v, --verbose			| Verbose output
#			-G, --generate GENERATE
#					Number of labyrinths to generate
#			-N, --generate_nonstandard
#					Generate labyrinths with various depths, numbers of rooms and hallway lengths
#			-C, --count COUNT		| Number of labyrinths to generate
#			-D, --depth DEPTH		| Room depth
#			-R, --rooms ROOMS		| Number of rooms to generate (has no effect if --generate_nonstandard is used)
#			-H, -L, --hallway_length HALLWAY_LENGTH
#					Hallway length
#			-P, --profiler			| Enable profiler
#			-T, --tests				| Invoke standard tests from a pre-defined input folder (comes with the repo)

# TODO:
#	- REPO
#---------------------------------------------------------------

#---------------------------------------------------------------
# Find the minimal cost of reaching the goal state, where goal state is:
# hallway: (0**hallway_length)
# rooms: roomN = decoded(N + 1)
#   0: always represents an empty cell
# 
# Weights per one node move
# A:	1
# B:   10
# C:  100
# D: 1000
# ...
# N: encoded(N)**10
# 
# Other constraints:
# Execution time < 10 s
# Memory usage < 200 MB
#
# Visual example of a goal state:
# #############
# #...........#
# ###A#B#C#D###
#   #A#B#C#D#
#   #A#B#C#D#
#   #A#B#C#D#
#   #########
#  
# Visual example of an initial state:
# #############
# #...........#
# ###B#C#B#D###
#   #D#C#B#A#
#   #D#B#A#C#
#   #A#D#C#A#
#   #########
# 
# Min cost: 44169
# 
#---------------------------------------------------------------
# 
# Binary encoding (not implemented yet):
#  Each node is represented by a 3-bit binary number
# b000 - . (empty)
# b001 - A
# b010 - B
# b011 - C
# b100 - D
# 
#---------------------------------------------------------------

from dataclasses import dataclass, field
from functools import wraps
from os import name as os_name
from pathlib import Path
from time import perf_counter, process_time, time
from typing import Dict, List, Tuple, Optional, Set, Union
from sys import getsizeof, stdin

import argparse, heapq, random, re, tracemalloc

#---------------------------------------------------------------
# DEFAULTS
VERSION = "0.9.9"
PATHFINDER_TITLE = "Labyrinth Pathfinder by El Daro"
DEFAULT_LABYRINTHS_DIR = "../labyrinths"

# Global


#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------

#----------------------------------------
# Class for handling files and their names
class File():
	'''
	Class for file name validation
	'''

	def __init__(self):
		''' '''
		print("\nWhy would you do that?..")

	@staticmethod
	def read(name, as_list: bool = False):
		'''
		This function reads from specified file

		Args:
			file_name (string): String representing a file name

		Return:
			list of strings
			or None if there were errors
		'''
		try:
			if not as_list:
				with open(name, "r", encoding="utf-8") as f:
					text = f.read()
					if len(text) == 0:										# If file is empty
						print("\nFile is empty")							# Let user know it
						return None											# And return None
					elif text.startswith("\ufeff"):							# If signed
						text = text.replace("\ufeff", '', 1)				# Unsign it
					else:
						return(text)										# Return text
			
			else:
				with open(name, "r", encoding="utf-8") as f:
					text_list = []
					for line in f:
						text_list.append(line.strip("\n"))
					if len(text_list) == 0:										# If file is empty
						print("\nFile is empty")								# Let user know it
						return None												# And return None
					elif text_list[0].startswith("\ufeff"):						# If signed
						text_list[0] = text_list[0].replace("\ufeff", '', 1)	# Unsign it
					else:
						return(text_list)										# Return list

		except FileNotFoundError:
			print ("\nFile not found")
			return None
		except Exception as e:
			print("\nSomething went wrong")
			print(e)
			return None

	@staticmethod
	def write(file_name, content, param = "w"):
		'''
		This function writes text in input file
		
		Args:
			file_name (string): String representing a file name
			content (list): A list of lines
		'''
		with open(file_name, param, encoding="utf-8") as f:
			f.write(content[0])
			for i in range(1, len(content)):
				f.write("\n" + content[i])

# NOTE: Used to track memory usage as well, but it slows the process down dramatically
class Profiler:
	"""Simple but effective profiler for A* testing."""
	
	def __init__(self):
		self.enabled = True
		self.results = []
	
	def __call__(self, func):
		"""Decorator usage."""
		@wraps(func)
		def wrapper(*args, **kwargs):
			if not self.enabled:
				return func(*args, **kwargs)
			
			# tracemalloc.start()
			start = perf_counter()
			
			result = func(*args, **kwargs)
			duration = perf_counter() - start
			# current, peak = tracemalloc.get_traced_memory()
			# tracemalloc.stop()
			
			self.results.append({
				'function': func.__name__,
				'time': duration,
				# 'memory_peak': peak,
				'result': result
			})
			print(f"{func.__name__:>19s} | {duration:>7.3f} s") # | {peak / 1024 / 1024:6.2f} MB")
			
			return result
		
		return wrapper
	
	def summary(self):
		"""Print summary statistics."""
		if not self.results:
			print("\nNo profiling data collected")
			return
		
		times = [result['time'] for result in self.results]
		# memories = [result['memory_peak'] for result in self.results]
		
		print(f"\n{'='*67}")
		print(f"PROFILING SUMMARY ({len(self.results)} runs)")
		print(f"{'='*67}")
		print(f"{'Total time: ':>20} {sum(times):>8,.3f} s")
		print(f"{'Average time: ':>20} {sum(times) / len(times):>7,.3f} s")
		print(f"{'Min time: ':>20} {min(times):>7.3f} s")
		print(f"{'Max time: ':>20} {max(times):>7.3f} s")
		# print(f"Avg memory:   {sum(memories) / len(memories) / 1024 / 1024:.2f} MB")
		# print(f"Peak memory:  {max(memories) / 1024 / 1024:.2f} MB")
		print(f"{'='*67}")

@dataclass(frozen=True)
class State:
	hallway: tuple
	rooms: tuple
	_hash: Optional[int] = field(default=None, init=False, compare=False)
	
	def __post_init__(self):
		object.__setattr__(self, '_hash', hash((self.hallway, self.rooms)))
	
	def __hash__(self):
		return self._hash

class Labyrinth:
	'''
	Labyrinth class that represents its state and various parameters, such as
	hallway length, accessible hallway positions, the number of rooms and depth of each room.
	Provides various methods to import the labyrinth from:
	  - Text
	  - File (given a path to the file)
	  - State (given a state)

	After importing a labyrinth you can use various ways of representing it, such as:
	  - Internal representation (__repr__)
		 accessed through get_state_readable() method or just through a state member.
	  - Graphical representation (__str__)
		 accessed through get_state_graphical() method or a print() function

	Examples:
	>>> input_path = "labyrinth.txt"
	>>> labyrinth_from_text = Labyrinth(input_path)
	>>> file = open(input_path).read()
	>>> str(labyrinth_from_text) == file
	True
	>>> labyrinth_from_file = Labyrinth(file)
	>>> str(labyrinth_from_file) == file
	True
	
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

	Retrieve the last generated state:
	>>> last_state = generator.state_history[-1]
	>>> labyrinth_last = Labyrinth(last_state)
	>>> labyrinth_last
	Hallway: (0, 0, 0, 0, 0, 0, 0)
	Rooms: ((3, 2, 1, 1), (4, 2, 3, 4))
	>>> labyrinth_last.state_current == labyrinth.state_current
	True
	>>> labyrinth_last == labyrinth
	True

	>>> input_path = "labyrinth.txt"
	>>> labyrinth = Labyrinth(input_path)
	>>> labyrinth.solve()
	296
	'''
	depth: int = 2
	hallway_length: int = 11
	# Should be able to always access by encoded room representation - 1
	ENERGY_COST_DEFAULT = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
	ROOM_POSITIONS_DEFAULT = (2, 4, 6, 8)
	
	states: list[State] = []
	state_initial: State
	state_current: State
	state_goal: State

	profiler = Profiler()

	def __init__(self, source: Optional[Union[str, State]] = None, *,
		labyrinth_as_text: str = "", path_to_labyrinth: str = "", state: Optional[State] = None):
		self.hallway = ()
		self.rooms = ()
		self.states = []
		self.initial_state_strict = True

		if (labyrinth_as_text is not None and labyrinth_as_text != ""):
			self.labyrinth_as_text = labyrinth_as_text
			self.import_from_text(labyrinth_as_text)
		elif (path_to_labyrinth is not None and path_to_labyrinth != ""):
			self.import_from_file(path_to_labyrinth)
		elif state is not None:
			self.import_from_state(state)
		
		elif isinstance(source, str) and source != "":
			# TODO: Check if this is actually a path
			self.labyrinth_as_text = labyrinth_as_text
			self.import_from_text(source)
		elif isinstance(source, State):
			self.import_from_state(source)
		
		# else:
			# raise ValueError("No valid input provided to Labyrinth constructor")

	def __eq__(self, other):
		if not isinstance(other, Labyrinth):
			return NotImplemented
		return self.state_current == other.state_current

	def __repr__(self):
		return self.get_state_readable()

	def __str__(self):
		return self.get_state_graphical()

	def import_from_text(self, labyrinth_as_text: str = ""):
		if labyrinth_as_text is not None and labyrinth_as_text != "":
			if not self.is_valid_input(labyrinth_as_text):
				print("Invalid labyrinth format")
				return
			self.labyrinth_as_text = labyrinth_as_text
		if self.labyrinth_as_text is None or self.labyrinth_as_text == "":
			raise ValueError("No labyrinth text provided")
		self.states = []
		self.rooms_positions = self._get_rooms_positions()
		self.state_initial = self.state_current = State(*self._parse(self.labyrinth_as_text))
		self.states.append(self.state_initial)
		self.state_goal = self.get_state_goal()
		self.energy_cost = self._get_energy_dict()

	def import_from_file(self, file_path: str):
		with open(file_path, 'r') as file:
			self.labyrinth_as_text = file.read()
			try:
				if not self.is_valid_input(self.labyrinth_as_text):
					print("Invalid labyrinth format")
					return
			except Exception as ex:
				raise ex
			self.import_from_text(self.labyrinth_as_text)

	def is_valid_input(self, labyrinth_as_text: str = "") -> bool:
		'''
		Checks if the input text string is valid.

		Args:
			labyrinth_as_text (str): Input text string representing the labyrinth

		Returns:
			bool: True if the input is valid, False otherwise
		'''
		if labyrinth_as_text is not None and labyrinth_as_text.strip() != "":
			self.labyrinth_as_text = labyrinth_as_text

		if self.labyrinth_as_text is None or self.labyrinth_as_text == "":
			raise ValueError("No labyrinth text provided")

		labyrinth_rows = self.labyrinth_as_text.strip().split()
		if len(labyrinth_rows) <= 3:				# At the simplest:
			return False							#  no walls, depth 2
		
		max_len = len(labyrinth_rows[0])
		for line in labyrinth_rows:
			if len(line) > max_len:					# Assume first line is the wall
				return False						# That defines the length
			if len(line) < 5:						# At the simplest: #A#B#
				return False

		if not re.fullmatch(r"^#+$", labyrinth_rows[0]):
			return False
		
		self.hallway_length = len(labyrinth_rows[1]) - 2
		if not re.fullmatch(r"^#\.+#$", labyrinth_rows[1]):
			self.initial_state_strict = False
			if self.hallway_length < 5 or self.hallway_length % 2 != 1:
				return False
		
		self.rooms_count = (self.hallway_length - 3) // 2

		# regex_room_pattern = r"^\s*#{1,3}((?:\w#)+)#{0,2}$"
		regex_single_pattern = r"#(?:([\w.]))"
		for line in labyrinth_rows[2:-1]:

			matches = re.findall(regex_single_pattern, line)
			if not matches:				# Should match any of the two kinds
				return False			# of room representation
			if len(matches) == 0:
				return False
			# Actual rooms count should match with the expected count
			# if (len(match.group(1)) // 2 != self.rooms_count):
			# 	return False

		return True

	def import_from_state(self, state: State):
		self.hallway = state.hallway
		self.rooms = state.rooms
		self.depth = len(state.rooms[0])
		self.rooms_count = len(state.rooms)
		self.hallway_length = len(state.hallway)
		self.rooms_positions = self._get_rooms_positions()
		self.hallway_accessible = self._get_hallway_accessible()
		self.state_initial = self.state_current = state
		self.states = []
		self.states.append(state)
		self.state_goal = self.get_state_goal()
		self.energy_cost = self._get_energy_dict()

	# Used by __repr__
	def get_state_readable(self, state: Optional[State] = None) -> str:
		if state is None:
			state = self.state_current
		return f"Hallway; {state.hallway}\nRooms: {state.rooms}"

	# Used by __str__
	def get_state_graphical(self, state: Optional[State] = None) -> str:
		'''
		Returns the current state of the labyrinth as a text string
		represeting the labyrinth.
		
		Args:
			state [State]: The state to unpack the rooms from. Optional.
			It is assumed that any passed state is compatible with initial state, i.e.
			has the same depth, number of rooms and hallway length. If not provided,
			inner initial state is used.

		Returns:
			str: String with graphical representation of the current state
		'''
		if state is None:
			state = self.state_current

		if (not state.hallway or not state.rooms or
				len(state.hallway) == 0 or len(state.rooms) == 0):
			return "No hallway or rooms to represent"
		
		labyrint_graphical = "#"*(self.hallway_length + 2) + "\n"
		labyrint_graphical += self.unpack_hallway(state)
		labyrint_graphical += self.unpack_rooms(state)
		
		return labyrint_graphical

	@staticmethod
	def _letter_to_index(char) -> int:
		if ('A' <= char <= 'z'):
			return ord(char) - ord('A') + 1
		elif ('a' <= char <= 'z'):
			return 26 + ord(char) - ord('a') + 1
		else:
			raise ValueError("Not a letter in range A-Z or a-z")
		
	@staticmethod
	def _index_to_letter(index) -> str:
		if (index == 0):
			return "."
		elif (1 <= index <= 25):
			return chr(index + ord('A') - 1)
		elif (26 <= index <= 51):
			return chr(index + ord('a') - 27)
		else:
			return "*"						# Unrecognised object
			# raise ValueError("Object unrecognised: {0}".format(char(index + ord('A') - 1)))

	# TODO: Move the binary encoding into a different function
	#		This one should only return int. And then it should either be used directly or compacted
	def _encode(self, char: str, binary: bool = False, bits_per_char = 3) -> int:
		'''
		Encodes a character into its binary or integer representation.

		Args:
			char (str): Character to encode
			binary (bool): If True, encodes as binary and packs into an integer.

		Returns:
			int: Encoded integer value
		'''
		try:
			if (not binary):
				if not char:
					char_encoded = -1
				elif char == '.':
					char_encoded = 0
				elif len(char) == 1 and char.isalpha():
					char_encoded = Labyrinth._letter_to_index(char)
				else:
					raise ValueError("Character to encode is not a single letter")
				return char_encoded
				# return int(string, 10)
			else:
				print("Not implemented yet (default bits per char: {bits_per_char})")
				raise NotImplementedError("Binary encoding not implemented yet")
				# return int(string, 2)
		except ValueError as ex:
			raise ValueError(ex)
		except Exception as ex:
			print("Unknown exception")
			raise Exception(ex)

	def _decode(self, obj: int, binary: bool = False, bits_per_char = 3) -> str:
		'''
		Decodes a compacted representation of a cell back to a character.

		Args:
			obj (int): Object to decode
			binary (bool): If True, decodes from a binary.

		Returns:
			str: Decoded character
		'''
		try:
			if (not binary):
				if obj is None:
					char_decoded = "#"
				char_decoded = Labyrinth._index_to_letter(obj)
				return char_decoded
				# return int(string, 10)
			else:
				print("Not implemented yet (default bits per char: {bits_per_char})")
				raise NotImplementedError("Binary encoding not implemented yet")
				# return int(string, 2)
		except ValueError as ex:
			raise ValueError(ex)
		except Exception as ex:
			print("Unknown exception")
			raise Exception(ex)

	def _get_parsed_rooms(self, labyrinth_rows: list, rooms_count: Optional[int] = None) -> tuple[tuple[int, ...], ...]:
		'''
		Parses the rooms from the labyrinth rows and returns them as a tuple of tuples,
		each containing the encoded values of the cells in a room.

		Args:
			labyrinth_rows (list): List of strings, each representing a row in a labyrinth
			rooms_count (int): Number of rooms in the labyrinth

		Returns:
			Tuple[tuple[int]]: Tuple of tuples, each containing the encoded values of the cells in a room
		'''
		# regex_residents_pattern = r"^\s*#{1,3}(?:(\w)#)+#{0,2}$"
		regex_single_pattern = r"#(?:([\w.]))"
		if rooms_count is not None:
			self.rooms_count = rooms_count
		rooms = [[] for _ in range(self.rooms_count)]
		depth = 0
		for row in labyrinth_rows[2:]:
			# matches = re.findall(regex_residents_pattern, row)
			matches = re.findall(regex_single_pattern, row)
			if len(matches) > 0:
				depth += 1
				for count, group in enumerate(matches, start = 0):
					rooms[count].append(self._encode(group, binary = False))
		self.depth = depth
		self.room_depth = depth

		return tuple(tuple(room) for room in rooms)

	def _get_hallway_accessible(self) -> tuple[int, ...]:
		'''
		Generates a tuple of accessible hallway positions based on the number of rooms. 
		Dependant on the rooms being parsed first
		
		Returns:
			tuple[int]: Tuple of accessible hallway positions
		'''
		if self.hallway_appendix_len is None:
			self.hallway_appendix_len = (self.hallway_length - self.rooms_count * 2 + 1) // 2
		hallway_accessible = []
		self.valid_hallway_positions = []

		for node in range(self.hallway_length):
			if node not in self.rooms_positions:
				self.valid_hallway_positions.append(node)
				if self.hallway[node] == 0:
					hallway_accessible.append(node)

			# Simpler, but is affected by initial state
			# if self.hallway[node] == 0 and node not in self.rooms_positions:
			# 	# Adding all the available hallway positions, except for the ones that are
			# 	# directly in front of the rooms
			# 	hallway_accessible.append(node)

		return tuple(hallway_accessible)

	def _get_energy_dict(self) -> Dict:
		try:
			energy_cost = {}
			energy_base = 1
			energy_multiplier = 10
			for room in self.state_goal.rooms:
				energy_cost[room[0]] = energy_base * energy_multiplier ** (room[0] - 1)

			return energy_cost
		except:
			return self.ENERGY_COST_DEFAULT

	def _get_rooms_positions(self):
		try:
			rooms_positions = []
			self.hallway_appendix_len = (self.hallway_length - self.rooms_count * 2 + 1) // 2
			for room in range(0, self.rooms_count):
				rooms_positions.append(room * 2 + self.hallway_appendix_len)
			# Short, but hardly readable:
			# rooms_positions = tuple([room * 2 + hallway_appendix_len for room in range(0, self.rooms_count)])
			return tuple(rooms_positions)
		except:
			return self.ROOM_POSITIONS_DEFAULT

	def _parse(self, labyrinth_as_text: str = "") -> Tuple[tuple, tuple]:
		'''
		Parses the input text string and returns the hallway and rooms as tuples.
		Does not create a classic graph representation and relies on a specific
		set of rules. Assumes the input is valid.

		Args:
			labyrinth_as_text (str): Input text string representing the labyrinth

		Returns:
			Tuple[tuple, tuple]: Tuple of the hallway and rooms as tuples
		'''

		if labyrinth_as_text is not None and labyrinth_as_text.strip() != "":
			self.labyrinth_as_text = labyrinth_as_text
		
		if self.labyrinth_as_text is None or self.labyrinth_as_text == "":
			raise ValueError("No labyrinth text provided")

		labyrinth_rows = self.labyrinth_as_text.strip().split()
		self.hallway_length = len(labyrinth_rows[1]) - 2
		# rooms_count = (self.hallway_length - 3) // 2
		# Incorrect when any rooms are empty on the specified row
		# rooms_count = len(re.findall(r"#(?:(\w))", labyrinth_rows[2]))
		self.rooms_count = len(re.findall(r"(?:(#))", labyrinth_rows[3])) - 1
		# Hallway setup
		self.hallway = tuple( self._encode(cell) for cell in labyrinth_rows[1][1:-1] )
		# Rooms setup
		self.rooms = self._get_parsed_rooms(labyrinth_rows)
		self.rooms_positions = self._get_rooms_positions()
		# Hallway accessible positions (dependant on rooms being parsed first)
		self.hallway_accessible = self._get_hallway_accessible()

		return self.hallway, self.rooms

	# (0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0) -> #...B.C.....#
	def unpack_hallway(self, state: Optional[State] = None) -> str:
		'''
		Unpacks the hallway tuple into a string representation.
		
		Args:
			state [State]: The state to unpack the rooms from. Optional.
			It is assumed that any passed state is compatible with initial state, i.e.
			has the same depth, number of rooms and hallway length.

		Returns:
			str: hallway string representation

		Example:
			(0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0) -> #...B.C.....#
		'''
		if state is not None:
			hallway = state.hallway
		else:
			hallway = self.state_current.hallway
		hallway_string = "#"
		for cell in hallway:
			hallway_string += self._decode(cell)
		return hallway_string + "#\n"

	# ([2, 1], [3, 4], [2, 3], [4, 1]) -> ###B#C#B#D###
											#A#D#C#A#
											#########
	def unpack_rooms(self, state: Optional[State] = None) -> str:
		'''
		Unpacks the rooms into a string representation.
		
		Args:
			state [State]: The state to unpack the rooms from. Optional.
			It is assumed that any passed state is compatible with initial state, i.e.
			has the same depth, number of rooms and hallway length. If not provided,
			inner representation of initial state is used.

		Returns:
			str: rooms string representation

		Example:
			([2, 1], [3, 4], [2, 3], [4, 1]) -> ###B#C#B#D###
												  #A#D#C#A#
												  #########
		'''
		if state is not None:
			rooms = state.rooms
		else:
			rooms = self.state_current.rooms
		appendix_len = (self.hallway_length - (len(rooms) * 2 + 1)) / 2
		if (not appendix_len.is_integer()):
			raise ValueError("The number of rooms and the hallway length do not much the required pattern")

		rooms_string = ""
		prefix = ''
		suffix = ''

		for current_depth in range(self.depth):
			if current_depth == 0:
				prefix = '#' * (int(appendix_len) + 1) + '#'
				suffix = '#' + '#' * int(appendix_len)
			else:
				prefix = ' ' * (int(appendix_len) + 1) + '#'
				suffix = ''

			rooms_string += prefix
			for room in rooms:
				rooms_string += self._decode(room[current_depth]) + '#'
			rooms_string += suffix + '\n'

		rooms_string += prefix + "#"*(len(rooms) * 2) + suffix + "\n"

		return rooms_string
	
	def get_state_goal(self) -> State:
		hallway_goal = tuple([0] * self.hallway_length)
		rooms_goal = []
		for room_index in range(self.rooms_count):
			rooms_goal.append(tuple([(room_index + 1)] * self.depth))
		rooms_goal = tuple(rooms_goal)
		return State(hallway_goal, rooms_goal)
	
	def display_state_goal(self):
		return self.get_state_graphical(self.state_goal)

	#--------------------------------
	# SEARCH
	# Not used
	def is_hallway_clear(self, state: State) -> bool:
		if any(cell != 0 for cell in state.hallway):
			return False
		return True

	def is_goal(self, state: State) -> bool:		
		return state == self.state_goal
	
	def heuristic(self, state: State) -> int:
		'''
		Estimates the lowest cost to reach the goal. Assumes that:
		  1. There are no other objects in the labyrinth
		  2. We exit/enter the closest cell in both rooms

		Results of this function are later added to the cost of the current state
		to put an estimate on completion score. This estimate is then used to
		order a heap containing possible moves.

		Args:
			state [State]: The state to compute the cost from.
			It is assumed that any passed state is compatible with initial state, i.e.
			has the same depth, number of rooms and hallway length.

		Returns:
			int: estimated cost to reach the goal (lowest margin)
		'''
		total = 0
		for position, obj in enumerate(state.hallway):
			if obj != 0:
				# Should be -1, 'cause objects are encoded starting from 1
				target_pos = self.rooms_positions[obj - 1]
				# +1, cause we need to enter the room as well. But don't consider depth yet
				distance = abs(position - target_pos) + 1
				total += distance * self.energy_cost[obj]
		
		for room_index, room in enumerate(state.rooms):
			correct_type = room_index + 1					# For room 0 it would be 1: 'A'
			room_pos = self.rooms_positions[room_index]
			
			for obj in room:
				if obj != 0 and obj != correct_type:
					target_pos = self.rooms_positions[obj - 1]
					# +2, cause we need to exit AND enter the room as well.
					# But don't consider depth yet
					distance = abs(room_pos - target_pos) + 2
					total += distance * self.energy_cost[obj]
		
		return total
	
	def _is_path_clear(self, state: State, start: int, end: int, from_hallway: bool = False) -> bool:
		'''Check if hallway path is clear between start and end.'''
		if start > end:
			start, end = end, start
			if from_hallway:				# Need to exclude the cell that
				end -= 1					# the object is standing on
		elif from_hallway:
			start += 1

		for cell in range(start, end + 1):
			if state.hallway[cell] != 0:
				return False
		
		return True
	
	def _is_room_accessible(self, room: Tuple[int], obj: int) -> bool:
		for resident in room:
			if resident == 0:				# Empty cell
				continue					# But the search continues
			elif resident != obj:		# Resident does not belong
				return False
			
		return True
	
	def _get_resident_to_move(self, room: Tuple, room_index: int, directly_to_room: bool = False) -> Optional[Tuple[int, int]]:
		is_found: bool = False				# = is_found (same purpose + 1 more)
		is_proper: bool = True
		obj: int
		cell: int
		for cell_index, resident in enumerate(room):
			if resident == 0:				# Empty cell
				continue					# But the search continues

			if not is_found:
				obj = resident				# Remember the first object for now
				cell = cell_index
			is_found = True

			if resident - 1 != room_index:	# If not a resident, the room isn't 'proper'
				is_proper = False
				break

		if ((is_proper) or not is_found or obj is None or
			(directly_to_room and obj - 1 == room_index)):
			return None
		return obj, cell

	def _try_move_to_destination(self, state: State, obj: int, 
								 starting_pos: int, from_hallway: bool = False) -> Optional[Tuple[int, int, int]]:
		'''
		Check if object can move directly to its destination room.
		
		Args:
			state [State]: current state (contains two tuples: hallway and rooms)
			obj [int]: integer representation of an object ('A': 1, 'B': 2...)
			starting_pos [int]: index of the starting position
			from_hallway [bool]: if set, the starting position will be ignored for
				possible target destinations.

		Returns:
			(room_room_index, target_cell, cost) 
			None
		'''
		target_room_index = self.rooms_positions[obj - 1]
		target_room = state.rooms[obj - 1]
		
		# Check if room is accepting new residents:
		found = False
		target_cell = self.depth - 1
		for cell_index, resident in enumerate(target_room):
			# See if residents of the room are either proper objects or an empty cell
			if resident != obj and resident != 0:
				return None
			if not found and resident == obj:
				found = True
			if not found and resident == 0:
				target_cell = cell_index

		# Check if path is clear
		if not self._is_path_clear(state, starting_pos, target_room_index, from_hallway):
			return None

		# Calculate cost
		steps_in_hallway = abs(starting_pos - target_room_index)
		steps_into_room = target_cell + 1		
		cost = (steps_in_hallway + steps_into_room) * self.energy_cost[obj]
		
		return (target_room_index, target_cell, cost)

	def _moves_hallway_to_room(self, state: State) -> List[Tuple[State, int]]:
		'''
		Move residents from hallway to their destination rooms.

		Args:
			state [State]: current state (contains two tuples: hallway and rooms)

		Returns:
			List[Tuple[State, int]]

		'''
		moves = []
		
		for hall_pos, resident in enumerate(state.hallway):
			if resident == 0:
				continue
			
			result = self._try_move_to_destination(state, resident, hall_pos, from_hallway = True)
			if result is None:
				continue
			
			target_room_index, target_cell, cost = result
			
			# Create new state
			new_hallway = list(state.hallway)
			new_hallway[hall_pos] = 0
			
			new_rooms = list(state.rooms)
			new_rooms[resident - 1] = list(new_rooms[resident - 1])
			new_rooms[resident - 1][target_cell] = resident			# Add to destination
			new_rooms[resident - 1] = tuple(new_rooms[resident - 1])

			new_state = State(tuple(new_hallway), tuple(new_rooms))
			moves.append((new_state, cost))
		
		return moves

	def _moves_room_to_room(self, state: State) -> List[Tuple[State, int]]:
		'''
		Move objects directly from one room to their destination room.
		Should be called before the _moves_room_to_hallway method.
		'Cause this is invariably a better move.

		Args:
			state [State]: current state (contains two tuples: hallway and rooms)

		Returns:
			List[Tuple[State, int]]
		'''
		moves = []
		
		for room_index, room in enumerate(state.rooms):
			result = self._get_resident_to_move(room, room_index, directly_to_room = True)
			if result is None:
				continue
			obj, cell = result

			starting_pos = self.rooms_positions[room_index] 
			steps_out = cell + 1
			
			# Try to move directly to destination
			result = self._try_move_to_destination(state, obj, starting_pos, from_hallway = False)
			if result is None:
				continue
			
			target_room_index, target_cell, target_room_move_cost = result
			
			# Total cost includes exiting current room
			total_cost = (steps_out * self.energy_cost[obj]) + target_room_move_cost

			new_rooms = list(state.rooms)
			
			new_rooms[room_index] = list(new_rooms[room_index])
			new_rooms[room_index][cell] = 0							# Remove from source
			new_rooms[obj - 1] = list(new_rooms[obj - 1])
			new_rooms[obj - 1][target_cell] = obj					# Add to destination
			new_rooms[room_index] = tuple(new_rooms[room_index])
			new_rooms[obj - 1] = tuple(new_rooms[obj - 1])
			
			new_state = State(state.hallway, tuple(new_rooms))
			moves.append((new_state, total_cost))
		
		return moves

	# TODO: Figure out how to avoid calling it when not needed.
	def _moves_room_to_hallway(self, state: State) -> List[Tuple[State, int]]:
		'''
		Move objects from rooms to hallway.
		Only do this if they can't go home directly.
		So it should be called last.
		
		Args:
			state [State]: current state (contains two tuples: hallway and rooms)

		Returns:
			List[Tuple[State, int]]
		'''
		moves = []
		
		for room_index, room in enumerate(state.rooms):
			if all(resident == room_index + 1 for resident in room):
				continue
			
			result = self._get_resident_to_move(room, room_index, directly_to_room = False)
			if result is None:
				continue
			obj, cell = result

			room_starting_pos = self.rooms_positions[room_index]
			room_destination_pos = self.rooms_positions[obj - 1]

			if self._is_path_clear(state, room_starting_pos, room_destination_pos) is not None:
				if self._is_room_accessible(state.rooms[obj - 1], obj):
					# Don't move to hallway - room-to-room move will handle this
					continue

			# Steps to exit room
			steps_out = cell + 1
			
			for target_hall_pos in self.valid_hallway_positions:
				# Check if path is clear
				if not self._is_path_clear(state, room_starting_pos, target_hall_pos, from_hallway = False):
					continue

				# Calculate cost
				steps_in_hallway = abs(room_starting_pos - target_hall_pos)
				cost = (steps_out + steps_in_hallway) * self.energy_cost[obj]
				
				# Create new state
				new_hallway = list(state.hallway)
				new_hallway[target_hall_pos] = obj
				
				new_rooms = list(state.rooms)
				new_rooms[room_index] = list(new_rooms[room_index])
				new_rooms[room_index][cell] = 0
				new_rooms[room_index] = tuple(new_rooms[room_index])
				
				new_state = State(tuple(new_hallway), tuple(new_rooms))
				moves.append((new_state, cost))
		
		return moves

	def get_possible_moves(self, state: State) -> List[Tuple[State, int]]:
		'''Generate all valid next states.'''
		moves = []
		
		# PRIORITY 1: Move from hallway to destination room
		# These dudes MUST go home - they can't move to hallway again
		moves.extend(self._moves_hallway_to_room(state))
		
		# PRIORITY 2: Move from room to room directly (if possible)
		# Why second? Because it's a rare thing, you know
		moves.extend(self._moves_room_to_room(state))
		
		# PRIORITY 3: Move from room to hallway (only if can't go home directly)
		moves.extend(self._moves_room_to_hallway(state))
		
		return moves

	# @profiler
	# def solve(self, verbose: bool = False) -> Optional[tuple[Dict[State, Optional[State]], int]]:
	def solve(self, verbose: bool = False) -> Optional[int]:
		'''
		A* search algorhythm.
		Estimates the lowest cost to reach the goal with heuristic function. It assumes that:
		  1. There are no other objects in the labyrinth
		  2. We exit/enter the closest cell in both rooms

		Results of the heuristic function are then added to the cost of the current state
		to put an estimate on minimal completion cost. This estimate is then used to
		order a heap containing possible moves.

		f_score (best_estimate): Is a total estimate of cost so far + best approximation
		At the start:
		f_score = 0 + initial_heuristic = initial_heuristic

		Afterwards:
		f_score = g_score + h_score
		Which translates to:
		cost_best_possible = cost_current + cost_heuristic
		
		move_counter: Second in priority. Servers as an ID, basically

		cost_current: Actual cost of reaching current state

		state: State representation

		Returns:
			- came_from: Dict[State, Optional[State]]
			- cost_current: int
		'''
		move_counter = 0
		cost_initial = self.heuristic(self.state_initial)
		heap = [(cost_initial, move_counter, 0, self.state_initial)]
		# Keeping track of the track, so to speak
		came_from: Dict[State, Optional[State]] = { self.state_initial: None }
		cost_so_far: Dict[State, int] = { self.state_initial: 0 }
		visited: Set[State] = set()
		
		iterations = 0
		
		while heap:
			try:
				cost_best_possible, _, cost_current, state_current = heapq.heappop(heap)
				iterations += 1
				
				if verbose and iterations % 10000 == 0:
					print(f"Iter: {iterations:7,}; Visited: {len(visited):6,}; " +
					f"Cost best possible: {cost_best_possible:6,}; Cost current: {cost_current:6,}; " +
					f"Heap count: {len(heap):6,}; Memory: {getsizeof(heap):6,} B")
				
				if state_current in visited:
					continue
				
				visited.add(state_current)
				
				if self.is_goal(state_current):
					if verbose:
						print(f"\n✓ Solution found!")
						print(f"{'Iterations: ':>20} {iterations:>8,}")
						print(f"{'States visited: ':>20} {len(visited):>8,}")
						print(f"{'Min cost: ':>20} {cost_current:>8,}")
					self.cost_min = cost_current
					return cost_current
					# TODO: Add history of the moves
					# return came_from, cost_current
				
				for state_next, move_cost in self.get_possible_moves(state_current):
					cost_next = cost_current + move_cost
					
					if state_next in cost_so_far and cost_next >= cost_so_far[state_next]:
						continue
					
					came_from[state_next] = state_current
					cost_so_far[state_next] = cost_next
					cost_heuristic_next = self.heuristic(state_next)
					cost_best_possible_next = cost_next + cost_heuristic_next
					
					move_counter += 1
					heapq.heappush(heap, (cost_best_possible_next, move_counter, cost_next, state_next))

			except Exception as ex:
				print("Iterations: {0:,}".format(iterations))
				print("move_counter: {0:,}".format(move_counter))
				print("cost_best_possible_next: {0:,}".format(cost_best_possible_next))
				raise ex
		
		if verbose:
			print(f"\n DEADLOCK")
			print(f" No solution was found!")
			print(f"{'Iterations: ':>20} {iterations:>8,}")
			print(f"{'States visited: ':>20} {len(visited):>8,}")
			print(f"{'Final cost: ':>20} {cost_current:>8,}")
		
		return None
		

#---------------------------------------
# Generator class
class Generator():
	'''
	Generator can be used to generate a new Labyrinth State.
	Use Labyrinth class to conver it into a string representation or find the min cost.
	
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

	Retrieve the last generated state:
	>>> last_state = generator.state_history[-1]
	>>> labyrinth_last = Labyrinth(last_state)
	>>> labyrinth_last
	Hallway: (0, 0, 0, 0, 0, 0, 0)
	Rooms: ((3, 2, 1, 1), (4, 2, 3, 4))
	>>> labyrinth_last.state_current == labyrinth.state_current
	True
	>>> labyrinth_last == labyrinth
	True
	>>> labyrinth.solve()
	296
	'''
	
	DEPTH_MIN: int = 2
	DEPTH_MAX: int = 4
	DEPTH_DEF: int = 2
	HALLWAY_LEN_MIN: int = 5
	HALLWAY_LEN_MAX: int = 19
	HALLWAY_LEN_DEF: int = 11
	DEFAULT_ROOM_COUNT: int = 4
	# Tuple[int, int, int] = (DEFAULT_ROOM_COUNT, DEPTH_DEF, HALLWAY_LEN_DEF)
	initial_parameters: List[Dict] = []
	states_history: List[State] = []
	parameters: Dict = {
		"rooms_count": DEFAULT_ROOM_COUNT,
		"depth": DEPTH_DEF,
		"hallway_length": HALLWAY_LEN_DEF
	}

	def __init__(self, rooms: Optional[int] = None,
					   depth: Optional[int] = DEPTH_DEF,
			  hallway_length: Optional[int] = HALLWAY_LEN_DEF,
				 ):
		self.parameters = self._get_valid_initial_parameters(rooms, depth, hallway_length)
		self.initial_parameters: List[Dict] = []
		self.states_history: List[State] = []
		self.initial_parameters.append(self.parameters)
		self.hallway = ()
		self.rooms = ()
		self.state = State(self.hallway, self.rooms)

	def _get_valid_depth(self, depth: Optional[int] = None):
		if depth and depth in range(self.DEPTH_MIN, self.DEPTH_MAX + 1):
			return depth
		else:
			return self.DEPTH_DEF
		
	def _get_valid_hallway_length(self, hallway_length: Optional[int] = None):
		if hallway_length is not None and (hallway_length in range(self.HALLWAY_LEN_MIN, self.HALLWAY_LEN_MAX + 1) and
			hallway_length % 2 == 1):
			return hallway_length
		else:
			return self.HALLWAY_LEN_DEF
	
	def _get_valid_rooms_count(self, rooms_count: Optional[int] = None,
								 hallway_length: Optional[int] = None):
		try:
			self.rooms_count = self._calculate_rooms_count(hallway_length)
			if rooms_count and 0 < rooms_count <= self.rooms_count:
				return rooms_count
			else:
				return self.rooms_count
		except Exception as ex:
			return self.DEFAULT_ROOM_COUNT
		
	def _get_valid_initial_parameters(self, rooms_count: Optional[int] = None,
												  depth: Optional[int] = None,
										 hallway_length: Optional[int] = None) -> Dict:
		if depth is None:
			if self.parameters and self.parameters["depth"] is not None:
				depth = self.parameters["depth"]
			else:
				depth = self.DEPTH_DEF
		else:
			depth = self._get_valid_depth(depth)
		
		if hallway_length is None:
			if self.parameters and self.parameters["hallway_length"] is not None:
				hallway_length = self.parameters["hallway_length"]
			else:
				hallway_length = self._get_valid_hallway_length(hallway_length)
		else:
			hallway_length = self._get_valid_hallway_length(hallway_length)

		# hallway_length = self._get_valid_hallway_length(hallway_length)
		
		if rooms_count is None:
			if self.parameters and self.parameters["rooms_count"] is not None:
				rooms_count = self.parameters["rooms_count"]
			else:
				rooms_count = self.DEFAULT_ROOM_COUNT
		else:
			rooms_count = self._get_valid_rooms_count(rooms_count, hallway_length)

		return {"rooms_count": rooms_count, "depth": depth, "hallway_length": hallway_length}

	def _calculate_rooms_count(self, hallway_length: Optional[int] = None) -> int:
		hallway_length = self._get_valid_hallway_length(hallway_length)
		if hallway_length in range(5, 7):
			return (hallway_length - 1) // 2
		else:
			return (hallway_length - 3) // 2

	def generate_random_list(self) -> list:
		'''
		Generates a heap with all the valid objects for the labyrinth and sorts it randomly.
		Pop an element to fill up a room.

		Args:
			None

		Returns:
			list: List of randomly sorted generated ints, each representing an object.
		'''
		objects_list = []
		for room in range(self.parameters["rooms_count"]):
			objects_list.extend([room + 1] * self.parameters["depth"])
		# objects_list = [i + 1 for i in range(rooms_count) for _ in range(self.depth)]

		random.shuffle(objects_list)

		return objects_list

	def generate_new_state(self, rooms_count: Optional[int] = None,
									   depth: Optional[int] = None,
							  hallway_length: Optional[int] = None) -> State:
		'''
		Generates a random initial State representation of a labyrinth.
		Hallway is always empty.
		Rooms are filled at random.
		If any of the input parameters are not present, the ones set during init are used.
		If those are not found, default parameters are used.

		Args:
			rooms_count (int): Number of rooms in the labyrinth.
			depth (int): Number of objects in each room.
			hallway_length (int): Length of the hallway.

		Returns:
			State: The State representation of an initial position.
		'''
		self.parameters = self._get_valid_initial_parameters(rooms_count, depth, hallway_length)
		# self.initial_parameters.append(self.parameters)
		self.parameters = self.initial_parameters[-1]
		self.rooms = []
		self.hallway = []
		self.labyrinth = []
		
		self.hallway = tuple([0 for _ in range(self.parameters["hallway_length"])])
		
		rooms = [[] for _ in range(self.parameters["rooms_count"])]
		objects_random_list = self.generate_random_list()
		for count, room in enumerate(rooms):
			rooms[count] = [objects_random_list.pop() for _ in range(self.parameters["depth"])]

		self.rooms = tuple(tuple(room) for room in rooms)
		self.state = State(self.hallway, self.rooms)
		self.states_history.append(self.state)

		return self.state

	def new_labyrinth(self, rooms: Optional[int] = None,
							depth: Optional[int] = None,
							hallway_length: Optional[int] = None,
							state: Optional[State] = None) -> Labyrinth:
		'''
		Generates a new labyrinth with a random initial state.
		Hallway is always empty.
		Rooms are filled at random.

		Returns:
			Labyrinth: A labyrinth object with a random initial position.
		'''
		if state is None:
			self.state = self.generate_new_state(rooms, depth, hallway_length)
		else:
			self.state = state
		self.labyrinth = Labyrinth(self.state)
	
		return self.labyrinth

#------------------------------------------------------------------------------
# Decorators
# Profiler
profiler = Profiler()

#------------------------------------------------------------------------------
# Tests
def display_labyrinth_info(labyrinth: Labyrinth, debug: bool = False, verbose: bool = False):
	print("\nLabyrinth reconstructed from the inner state representation:\n")
	print(labyrinth)
	print(labyrinth.get_state_readable())
	if verbose:
		print(labyrinth.states[0])
		print("Hallway accessible: {0}".format(labyrinth.hallway_accessible))
		print("Depth: {0}".format(labyrinth.depth))
		print("Room positions: {0}".format(labyrinth.rooms_positions))
		print("Energy cost dictionary: {0}".format(labyrinth.energy_cost))
		print("Goal state: {0}".format(labyrinth.get_state_goal()))
	if debug:
		print("Alternative display methods:\n")
		labyrinth
		print("Hallway: {0}".format(labyrinth.hallway))
		print("Rooms: {0}".format(labyrinth.rooms))
		print("Hallway accessible: {0}".format(labyrinth.hallway_accessible))
		print("Depth: {0}".format(labyrinth.depth))
		print("\nGoal state graphical:")
		print(labyrinth.display_state_goal())
	print()

def test_labyrinth(input_path, debug: bool = False):
	try:
		print('{:-^40}'.format("Testing input from file as text"))
		print("\nInput file path: {0}\n".format(Path(input_path).resolve()))
		input_file = str(File.read(Path(input_path)))
		print("Labyrinth read directly from file:\n")
		print(input_file)
		
		labyrinth = Labyrinth(input_file)
		display_labyrinth_info(labyrinth, debug)

		if(str(labyrinth.get_state_graphical()).strip() == input_file.strip()):
			print("Cool\n")
		else:
			print("NOT COOL!\n")

	except Exception as ex:
		print("Exception occurred while testing labyrinth from file.\n\n  Path: {0}\n  Exception: {1}\n".format(
			Path(input_path).resolve(), ex))

def test_labyrinth_as_file(input_path, debug: bool = False):
	try:
		print('{:-^40}'.format("Testing input from file as path."))
		print("\nInput file path: {0}\n".format(Path(input_path).resolve()))
		input_file = str(File.read(Path(input_path)))
		print("Labyrinth read directly from file:\n")
		print(input_file)
		labyrinth = Labyrinth()
		labyrinth.import_from_file(input_path)
		display_labyrinth_info(labyrinth, debug)

		if(str(labyrinth.get_state_graphical()).strip() == input_file.strip()):
			print("Cool\n")
		else:
			print("NOT COOL!\n")
	
	except Exception as ex:
		print("Exception occurred while testing labyrinth from file.\n\n  Path: {0}\n  Exception: {1}\n".format(
			Path(input_path).resolve(), ex))

def get_input_paths(input_dir: str):
	input_paths = []
	parent_dir = Path(__file__).parent.resolve()
	input_dir_absolute = Path(parent_dir, input_dir)
	if not input_dir_absolute.exists() or not input_dir_absolute.is_dir():
		raise FileNotFoundError("Input directory not found: {0}".format(input_dir_absolute))
	for file_path in input_dir_absolute.iterdir():
		if file_path.is_file() and file_path.suffix == ".txt":
			input_paths.append(str(file_path.resolve()))
	return input_paths

def test_all_labyrinths(input_dir: str, debug: bool = False, verbose: bool = True):
	input_paths = get_input_paths(input_dir)
	for input_path in input_paths:
		test_labyrinth(input_path, debug)

def test_generator(parameters: Optional[Dict] = None, debug: bool = False):
	try:
		print('{:-^40}'.format("Testing generator"))
		params_allowed = {"rooms", "depth", "hallway_length"}
		if parameters is None:
			print("- No parameters")
			generator = Generator()
			print("Default parameters: {0}".format(generator.parameters))
		else:
			print("- Parameters provided")
			params_clean = {key: value for key, value in parameters.items() if key in params_allowed}
			generator = Generator(**params_clean)
			print("Provided parameters: {0}".format(parameters))
			print("Processed parameters: {0}".format(generator.parameters))
			
		labyrinth = generator.new_labyrinth()
		display_labyrinth_info(labyrinth, debug)
		if parameters and "test_reuse" in parameters and parameters["test_reuse"]:
			print("--Test_reuse was passed--")
			labyrinth = generator.new_labyrinth(rooms = 5, depth = 3, hallway_length = 13)
			display_labyrinth_info(labyrinth, debug)

		success_state = labyrinth.state_current == generator.states_history[-1]
		if(success_state):
			print("State: cool\n")
		else:
			print("State: NOT COOL!\n")

		success_repr = labyrinth == Labyrinth(generator.states_history[-1])
		if(success_repr):
			print("Representation: cool\n")
		else:
			print("Representation: NOT COOL!\n")

		success_set = str(labyrinth) == str(Labyrinth(generator.states_history[-1]))
		if(success_set):
			print("String: cool\n")
		else:
			print("String: NOT COOL!\n")
	
	except Exception as ex:
		print("Exception occurred while testing generator.\n\n  Parameters: {0}\n  Exception: {1}\n".format(
			parameters, ex))

def test_all_generators(debug: bool = False, verbose: bool = True):
	params_list = [
		None,
		{"rooms": 4, "depth": 2, "hallway_length": 11},
		{"rooms": 4, "depth": 4, "hallway_length": 11},
		{"rooms": 2, "depth": 4, "hallway_length": 11},
		{"rooms": 4, "hallway_length": 7},
		{"depth": 4, "hallway_length": 11},
		{"rooms": 2, "depth": 4},
		{"rooms": 2},
		{"depth": 4},
		{"hallway_length": 19},
		{"rooms": 2, "depth": 4, "hallway_length": 11, "test_reuse": True},
	]
	test_generator()					# No arguements passed whatsoever
	for parameters in params_list:
		test_generator(parameters)

@profiler
def test_search(input_path, debug: bool = False, verbose: bool = False):
	try:
		print('\n{:=^67}'.format(""))
		print('{:-^67}'.format("Testing search (input from file as text)"))
		print("\nInput file path: {0}\n".format(Path(input_path).resolve()))
		input_file = str(File.read(Path(input_path)))
		
		labyrinth = Labyrinth(input_file)
		display_labyrinth_info(labyrinth, debug, verbose)

		print('{:-^67}'.format("SEARCH"))
		try:
			verbose = True
			time_start = perf_counter()
			solved = labyrinth.solve(verbose)
			time_elapsed = perf_counter() - time_start
			if solved is not None:
				# print(solved[0])
				if not verbose:
					# print("\n  Min cost: {0:,}".format(solved[1]))
					print(f"\n{'Min cost: ':>20} {solved:>8,}")
			print(f"{'Time elapsed: ':>20} {time_elapsed:>8.3f} s")
		except Exception as ex:
			print("[ERROR] Search failed: {0}".format(ex))

	except Exception as ex:
		print("[ERROR] Exception occurred while testing labyrinth from file.\n\n  Path: {0}\n  Exception: {1}\n".format(
			Path(input_path).resolve(), ex))
		
def test_all_searches(input_dir: str, debug: bool = False):
	input_paths = get_input_paths(input_dir)
	for input_path in input_paths:
		test_search(input_path, debug)

@profiler
def test_generator_and_search(parameters: Optional[Dict] = None, debug: bool = False, verbose: bool = False):
	try:
		params_allowed = {"rooms", "depth", "hallway_length"}
		if parameters is None:
			print("- No parameters")
			generator = Generator()
			print("Default parameters: {0}".format(generator.parameters))
		else:
			print("- Parameters provided")
			params_clean = {key: value for key, value in parameters.items() if key in params_allowed}
			generator = Generator(**params_clean)
			print("Provided parameters: {0}".format(parameters))
			print("Processed parameters: {0}".format(generator.parameters))

		labyrinth = generator.new_labyrinth()
		display_labyrinth_info(labyrinth, debug, verbose)
		
		print('{:-^67}'.format("SEARCH"))
		try:
			# verbose = True
			# time_start = perf_counter()
			time_start = process_time()
			solved = labyrinth.solve(verbose)
			# time_elapsed = perf_counter() - time_start
			time_elapsed = process_time() - time_start
			if solved is not None:
				# print(solved[0])
				if not verbose:
					# print("\n  Min cost: {0:,}".format(solved[1]))
					print(f"\n{'Min cost: ':>20} {solved:>8,}")
			print(f"{'Time elapsed: ':>20} {time_elapsed:>8.3f} s")
		except Exception as ex:
			print("\n[ERROR] Search failed: {0}".format(ex))

	except Exception as ex:
		print("\n[ERROR] Exception occurred while testing generator.\n\n  Parameters: {0}\n  Exception: {1}\n".format(
			parameters, ex))

def test_all_generator_and_search(count: int = 1, parameters: Optional[Dict] = None, debug: bool = False, verbose: bool = False):
	# Use instead of the counter
	params_list = [
		None,
		{"rooms": 4, "depth": 2, "hallway_length": 11},
		{"rooms": 4, "depth": 4, "hallway_length": 11},
		{"rooms": 2, "depth": 4, "hallway_length": 11},
		{"rooms": 4, "hallway_length": 7},
		{"depth": 4, "hallway_length": 11},
		{"rooms": 2, "depth": 4},
		{"rooms": 2},
		{"depth": 4},
		{"hallway_length": 19},
		{"rooms": 2, "depth": 4, "hallway_length": 11, "test_reuse": True},
	]
	limit = 50
	if count >= limit:
		print(f"{limit} is a lot. Are you sure? Y/N")
		user_input = str(input("> ")).strip()
		if user_input == 'N':
			print("All right, calling it all off")
			return
	for i in range(count):
		print('\n{:=^67}'.format(""))
		print('{:-^67}'.format("Testing generator + search"))
		print(f"\nTest: {i + 1} / {count}")
		test_generator_and_search(random.choice(params_list), debug, verbose)
		
def run_tests(debug: bool = False):
	labyrinths_dir = "../labyrinths"
	parent_dir = Path(__file__).parent.resolve()
	input_dir_absolute = Path(parent_dir, labyrinths_dir)
	
	#-----------------------
	# NOTE: Uncomment the options below to run various sets of tests:
	# LABYRINTHS
	# Option 1.A
	# test_all_labyrinths(labyrinths_dir)

	# Option 1.B
	# test_file = Path(input_dir_absolute, "labyrinth_hallway_not_empty.txt")
	# test_labyrinth(test_file, debug)

	# Option 1.C
	# test_file = Path(input_dir_absolute, "labyrinth.txt")
	# test_labyrinth_as_file(test_file, debug)

	#-----------------------
	# GENERATORS
	# Option 2.A
	# test_all_generators()

	# Option 2.B
	# generator = Generator()
	# labyrinth = generator.new_labyrinth()
	# display_labyrinth_info(labyrinth, debug = False)

	# Option 2.C
	# generator = Generator(rooms = 4, depth = 4, hallway_length = 11)
	# labyrinth = generator.new_labyrinth()
	# display_labyrinth_info(labyrinth, debug = False)

	#-----------------------
	# SEARCH
	# NOTE: 
	# Option 3.A
	labyrinths_search_dir = "../labyrinths/search"
	parent_dir = Path(__file__).parent.resolve()
	# # input_search_dir_absolute = Path(parent_dir, labyrinths_search_dir)
	test_all_searches(labyrinths_search_dir, debug)

	#-----------------------
	# Option 3.B
	# test_file = Path(input_dir_absolute, "labyrinth.txt")
	# test_file_1 = Path(input_dir_absolute, "labyrinth_ident_1.txt")
	# test_file_2 = Path(input_dir_absolute, "labyrinth_ident_2.txt")
	# test_file_goal = Path(input_dir_absolute, "labyrinth_goal.txt")
	# lab_1 = Labyrinth(path_to_labyrinth = str(test_file_1))
	# lab_2 = Labyrinth(path_to_labyrinth = str(test_file_2))
	# lab_goal = Labyrinth(path_to_labyrinth = str(test_file_goal))

	# lab_1.state_current == lab_2.state_current
	# lab_goal.is_goal(lab_goal.state_current)
	
	# test_labyrinth_as_file(test_file, debug)
	# test_search(test_file)
	# test_search(test_file_goal)

	#-----------------------
	# GENERATOR+SEARCH
	# NOTE: Automatic
	# Option 4.A
	# count = 10
	# test_all_generator_and_search(count = count, debug = debug)

#------------------------------------------------------------------------------
# Arguments parsing
@profiler
def solve_from_text(input_as_text: str, debug: bool = False, verbose: bool = False):
# def solve_from_file(file_path: str, debug: bool = False, verbose: bool = False):
	try:
		# file_obj = str(Path(file_path).resolve())
		# # print(file_obj)
		# input_as_text = str(File.read(Path(file_obj)))

		if debug or verbose:
			print("Labyrinth read directly from stdin:\n")
			print(input_as_text)
		
		labyrinth = Labyrinth(input_as_text)
		if debug or verbose:
			display_labyrinth_info(labyrinth, debug, verbose)

		time_start = perf_counter()						# process_time()
		solved = labyrinth.solve(verbose)
		time_elapsed = perf_counter() - time_start
		if solved is not None and (debug or verbose):
			# print("\n  Min cost: {0:,}".format(solved[1]))
			print(f"\n{'Min cost: ':>20} {solved:<8,}")
			print(f"{'Time elapsed: ':>20} {time_elapsed:>8.3f} s")

		print(solved)
	
	except Exception as ex:
		raise ex

def solve_from_input(input_path: Optional[str] = None, debug: bool = False, verbose: bool = False):
	# From the cli:   ./run.py < input.txt
	#			 cat input.txt | ./run.py
	#			 ./run.py (manual)
	try:
		if not input_path:
			if os_name == 'nt':
				help_message = "To exit send Ctrl+Z then Enter"
			elif os_name == 'posix':
				help_message = "To exit send Ctrl+D then Enter"
			else:
				help_message = "To exit enter a new line and type END"
			
			print("Paste or type the graphical representation of the labyrinth below.")
			print(help_message)
			print()
		
			try:
				input_as_text = stdin.read().strip()
			except EOFError:				# Ctrl+D or Ctrl+Z
				if debug:
					print("\nInput read")
			except KeyboardInterrupt:		# Ctrl+C
				print("\nCancelled")
				exit(0)

		else:
			parent_dir = Path(__file__).parent.resolve()
			path_absolute = Path(parent_dir, input_path).resolve()
			if path_absolute.is_dir():
				input_paths = get_input_paths(str(path_absolute))
				for input_file_path in input_paths:
					file_obj = Path(input_file_path).resolve()
					input_as_text = str(File.read(file_obj))
					solve_from_text(input_as_text, debug, verbose)
				return

			elif Path(path_absolute).is_file() and Path(path_absolute).suffix == ".txt":
				file_obj = Path(path_absolute).resolve()
				input_as_text = str(File.read(file_obj))
				solve_from_text(input_as_text, debug, verbose)
				return

		solve_from_text(input_as_text, debug, verbose)

	except Exception as ex:
		raise ex

@profiler
def solve_from_generators(depth: Optional[int] = None,
						  rooms: Optional[int] = None,
				 hallway_length: Optional[int] = None,
						  count: Optional[int] = None,
						  		   debug: bool = False,
								 verbose: bool = False):
	'''Generates new labyrinths and solves them'''
	try:
		generator = Generator(depth = depth,
							rooms = rooms,
					hallway_length = hallway_length)
		generator_amount = 1
		if count is not None:
			generator_amount = count
		for i in range(generator_amount):
			labyrinth = generator.new_labyrinth()
			if debug or verbose:
				display_labyrinth_info(labyrinth, debug, verbose)
			time_start = perf_counter()						# process_time()
			solved = labyrinth.solve(verbose)
			time_elapsed = perf_counter() - time_start
			if solved is not None and (debug or verbose):
				# print("\n  Min cost: {0:,}".format(solved[1]))
				print(f"\n{'Min cost: ':>20} {solved:>8,}")
				print(f"{'Time elapsed: ':>20} {time_elapsed:>8.3f} s")
			print(solved)
	
	except Exception as ex:
		raise ex

def read_labyrinth_from_file(path):
	'''Read from file specified as argument.'''
	parent_dir = Path(__file__).parent.resolve()
	new_path = Path(parent_dir, path)
	try:
		return File.read(name = new_path, as_list = False)
	except Exception as ex:
		raise ex

def invoke_labyrinth_solver(args):
	try:
		if args.input_path:
			# From file argument: ./run.py input.txt
			# input_as_text = read_labyrinth_from_file(args.input_path)
			solve_from_input(input_path = args.input_path, debug = args.debug, verbose = args.verbose)
		elif args.depth or args.rooms or args.count or args.generate or args.generate_nonstandard:
			if args.generate:
				generator_amount = args.generate
			elif args.count:
				generator_amount = args.count
			else:
				generator_amount = 4
			if args.generate_nonstandard:
				test_all_generator_and_search(count = generator_amount,
								debug = args.debug, verbose = args.verbose)
			else:
				solve_from_generators(args.depth, args.rooms, args.hallway_length,
							generator_amount, args.debug, args.verbose)
		else:
			solve_from_input(debug = args.debug, verbose = args.verbose)
	
	except Exception as ex:
		raise ex	

def parse_arguments():
	parser = argparse.ArgumentParser(description = PATHFINDER_TITLE)
	parser.add_argument("input_path",        nargs = '?',
					 help = "Path to the input file or folder")
	
	parser.add_argument('-d', "--debug",     action = "store_true", 
					 help = "Debug output")
	parser.add_argument('-v', "--verbose",   action = "store_true",
					 help = "Verbose output")
	
	parser.add_argument('-G', "--generate",  type = int, 
					 help = "Number of labyrinths to generate")
	parser.add_argument('-N', "--generate_nonstandard", action = "store_true",
					 help = "Generate labyrinths with various depths, numbers of rooms and hallway lengths")
	
	# Generator options
	parser.add_argument('-C', "--count",     type = int, default = 1,
					 help = "Number of labyrinths to generate")
	parser.add_argument('-D', "--depth",     type = int,
					 help = "Room depth")
	parser.add_argument('-R', "--rooms",     type = int,
					 help = "Number of rooms to generate (has no effect if --generate_nonstandard is used)")
	parser.add_argument('-H', '-L', "--hallway_length", type = int, default = 11,
					 help = "Hallway length")

	# Tests and profiler
	parser.add_argument('-P', "--profiler",  action = "store_true",
					 help = "Enable profiler")
	parser.add_argument('-T', "--tests",     action = "store_true",
					 help = "Invoke standard tests from a pre-defined input folder (comes with the repo)")
	
	return parser.parse_args()

#------------------------------------------------------------------------------
# Main function
def main():
	args = parse_arguments()	
	# NOTE: Manual
	# debug = True
	debug = args.debug
	if debug:
		print('\n{:-^67}'.format("Labyrinth Pathfinder by El Daro"))
		print('  VERSION: {0}'.format(VERSION))

	if args.profiler:
		profiler.enabled = True
	else:
		profiler.enabled = False

	# Parse and solve
	if args.tests:
		run_tests(debug = False)
		if args.profiler:
			profiler.summary()
	
	else:
		invoke_labyrinth_solver(args)

if __name__ == "__main__":
	main()