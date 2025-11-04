#pylint:disable=W0312

#---------------------------------------------------------------------------#
# Version: 0.4.3															#
# Virus:Isolation															#
# Through tough though thorough thought.									#
#---------------------------------------------------------------------------#

#---------------------------------------------------------------------------#
#-------------------------COMMAND LINE ARGUMENTS----------------------------#
#---------------------------------------------------------------------------#
#	positional arguments:
#  		input_string			| Path to the input file or folder
#
#		options:
#  		-h, --help				| show this help message and exit
#		-d, --debug				| Debug output
#		-v, --verbose			| Verbose output
#		-P, --profiler			| Enable profiler
#		-T, --tests				| Invoke standard tests from a pre-defined
# 								|	input folder (comes with the repo)
#		-O, --option OPTION		| Defines what specific tests to run
#			OPTION				| DEFAULT, EXAMPLE, FROM_FILE, FROM_DIR
#---------------------------------------------------------------------------#
#---------------------------------CHANGELOG---------------------------------#
#---------------------------------------------------------------------------#
# v0.2.3
#	- Improved debugging and visual representation
# v0.3.0
#	- Reworked BFSResult class
#	- Added State class to keep track of the moves
#	- Made the game cycle repeatable by introducing reset functionality
#	- Fixed and improved agnostic BFS method
#	- Added the main game cycle, including:
#	  BFS search, target prioritization, edge severing and moves
# v0.4.0
#	- Fixed game loop failing to find the correct severance sequence
#	  when more than one node was connected to a single gateway
#	- Fixed tests with FROM_FILE and FROM_DIR options
#	- Added test cases in the `/graphs/tests` directory
#	- Added a new option to the command line arguments (`test_path`)
#	  to specify the path to file or directory for when the `--tests`
# 	  option is used.
#	- Added processing of a positional argument:
# 	  graph as text or path to file or directory
#	- Added default representation for the `__str__` method based on the
#	  current state of the graph
# v0.4.1
#	- Implemented test wtih pre-defined examples (in-code)
# v0.4.2
#	- Improved debug and verbose outputs
#	- Added state representation for any given step
#	- Added verbose output during the game loop to see the steps
# v0.4.3
#	- Further improved debug and verbose output
#	- Added a new feature that displays all the steps in a readable way 
#---------------------------------------------------------------------------#
# TODO:
#	- Corner cases
#---------------------------------------------------------------------------#

#---------------------------------------------------------------------------#
# Given a graph defined as edges between nodes, find the correct sequence of
# edges to sever in order to prevent the virus from reaching the gateways.
#
# The Virus finds the closes gateway and tries to reach it.
# The system acts first by severing the edge to the closest gateway.
# Then the Virus moves. 
#
# Input example:
#	a-b
#	a-c
#	b-d
#	b-A
#	c-f
#	d-e
#	d-B
#	e-f
#	f-C
#
# Visual representation:
#     A   B
#     |   |
# a---b---d
# |       |
# c---f---e
#     |
#     C
#
# Correct output:
#	A-b
#	B-d
#	C-f
# 
#---------------------------------------------------------------------------#

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from os import name as os_name
from pathlib import Path
from time import perf_counter, process_time, time
from typing import Dict, List, Optional, Tuple, Set, Union
from sys import getsizeof, stdin

import argparse, heapq, random, re, sys, tracemalloc

#---------------------------------------------------------------
# DEFAULTS
VERSION = "0.4.3"
ISOLATION_TITLE = "Virus:Isolation by El Daro"
DEFAULT_GRAPHS_DIR = "../graphs"
DEFAULT_SEARCH_DIR = "../graphs/search"
DEFAULT_INVALID_DIR = "../graphs/invalid"

DEFAULT_TESTS_DIR = DEFAULT_GRAPHS_DIR + "/tests"
DEFAULT_TESTS_FILE = DEFAULT_TESTS_DIR + "/graph_simple.txt"

ARGS_DEF_OPTIONS = { "DEFAULT", "EXAMPLE", "FROM_FILE", "FROM_DIR" }

# Global


#---------------------------------------------------------------------------
# Classes
#---------------------------------------------------------------------------


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
	'''
	A very basic profiler that analyzes execution time.
	Used to track memory usage as well, but it slows the process down dramatically.
	Unly uncomment it when you really NEED to analyze the memory usage.
	'''
	
	def __init__(self):
		self.enabled = True
		self.results = []
	
	def __call__(self, func):
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
		if not self.results:
			print("\n No profiling data collected\n")
			return
		
		times = [result['time'] for result in self.results]
		# memories = [result['memory_peak'] for result in self.results]
		
		print(f"\n{'='*67}")
		print(f"PROFILING SUMMARY ({len(self.results)} runs)")
		print(f"{'='*67}")
		print(f"{'Total time: ':>20} {sum(times):>8,.3f} s")
		print(f"{'Average time: ':>20} {sum(times) / len(times):>8,.3f} s")
		print(f"{'Min time: ':>20} {min(times):>8.3f} s")
		print(f"{'Max time: ':>20} {max(times):>8.3f} s")
		# print(f"Avg memory:   {sum(memories) / len(memories) / 1024 / 1024:.2f} MB")
		# print(f"Peak memory:  {max(memories) / 1024 / 1024:.2f} MB")
		print(f"{'='*67}")

@dataclass
class BFSResult:
	'''Result of a BFS search'''
	targets_found: set[str] | None
	distances: dict[str, int] | None
	distance_shortest: int | None
	parents: dict[str, str | None]

@dataclass
class State:
	position: str | None
	target: str | None
	priority_path_current: list[str] | None
	severed_edge: dict[str, str | None]
	priority_path_next: list[str] | None

class Graph:
	'''
	Describes a bidirectional graph and provides its basic methods.
	That's it.
	'''
	_node_edges_initial: defaultdict[str, set[str]]
	node_edges: defaultdict[str, set[str]]
	_gateway_edges_initial: defaultdict[str, set[str]]
	gateway_edges: defaultdict[str, set[str]]
	nodes: set[str]
	_gateways_initial: set[str]
	gateways: set[str]
	subgraphs: defaultdict[str, set[str]]

	def __init__(self, source: Optional[str | List[str] | list[str] | 'Graph'] = None, *,
			  import_as_text: Optional[str] = None, path: str = "", graph: Optional['Graph'] = None):
		self.node_edges = defaultdict(set)
		self.gateway_edges = defaultdict(set)
		self.nodes = set()
		self.gateways = set()
		self.subgraphs = defaultdict(set)
		self.graph_as_text = None
		if import_as_text is not None and import_as_text != "":
			file_try = Path(import_as_text)
			if file_try.is_dir() or file_try.is_file():
				self.import_from_file(str(file_try))
			else:
				self.import_from_text(import_as_text)
		elif path is not None and path != "":
			self.import_from_file(str(Path(path)))
		elif graph is not None:
			self.import_from_graph(graph)

		elif isinstance(source, str) and source != "":
			file_try = Path(source)
			if file_try.is_dir() or file_try.is_file():
				self.import_from_file(source)
			else:
				self.import_from_text(source)
		elif isinstance(source, List) or isinstance(source, list):
			self.graph_as_text = "\n".join(source)
			self.import_from_text(self.graph_as_text)
		elif isinstance(source, Graph):
			self.import_from_graph(source)
		else:
			raise ValueError("Invalid input type")

		self._post_init()
		
	def _post_init(self):
		self._node_edges_initial = deepcopy(self.node_edges)
		self._gateway_edges_initial = deepcopy(self.gateway_edges)
		self._gateways_initial = deepcopy(self.gateways)
		
	def import_from_graph(self, graph: 'Graph'):
		self.node_edges = graph.node_edges if graph.node_edges else defaultdict(set)
		self.gateway_edges = graph.gateway_edges if graph.gateway_edges else defaultdict(set)
		self.nodes = graph.nodes if graph.nodes else set()
		self.gateways = graph.gateways if graph.gateways else set()
		self.subgraphs = graph.subgraphs if graph.subgraphs else defaultdict(set)
		self.graph_as_text = graph.graph_as_text if graph.graph_as_text else None

	def import_from_text(self, graph_as_text):
		if not self._is_valid_input(graph_as_text):
			print("Invalid input graph format")
			raise ValueError("Invalid input graph format")
	
		self.graph_as_text = graph_as_text
		self._build()
	
	def import_from_file(self, file_path: str):
		with open(file_path, 'r') as file:
			self.graph_as_text = file.read()
			try:
				if not self._is_valid_input(self.graph_as_text):
					print(f"Invalid input graph format.\n  Path: {file_path}")
					return
			except Exception as ex:
				print(f"Error importing graph from file: {file_path}")
				print(f"  Exception: {ex}")
				raise ex
			self.import_from_text(self.graph_as_text)

	def import_from_list(self, list_of_strings: list[str]):
		self.graph_as_text = "\n".join(list_of_strings)
		self.import_from_text(self.graph_as_text)

	def _is_valid_input(self, graph_as_text):
		# Rule 1: a-b format
		# Rule 2: Only latin symbols and a gyphen
		# Rule 3: No duplicate edges
		# Rule 4: No self-loops
		# Rule 5: * No multiple connections between the same nodes
		# Rule 6: No edges between gateways
		# * Might allow it, since it's non-braking
		if graph_as_text is not None and graph_as_text.strip() != "":
			self.graph_as_text = graph_as_text

		if self.graph_as_text is None or self.graph_as_text == "":
			raise ValueError("No graph input provided")

		graph_rows = self.graph_as_text.replace("\t", "").replace(",", "").strip().splitlines()

		# Allowed formats: a-b, A-b, a-B
		regex_from = re.compile(r"(?:([A-Za-z]-[a-z]))")
		regex_to = re.compile(r"(?:([a-z]-[A-Za-z]))")
		# if not all(regex_from.fullmatch(row) or
		# 	 		 regex_to.fullmatch(row) for row in graph_rows):
		# 	return False
		
		for row in graph_rows:
			if not regex_from.fullmatch(row) and not regex_to.fullmatch(row):
				print(f"Invalid edge format: {row}")
				return False
			node_from, node_to = row.split("-")
			if node_from == node_to:
				print(f"Self-loop detected: {row}")
				return False

		return True
	
	# TODO: See if adding A-b edges to node_edges is actually necessary 
	def add_edge(self, node_from, node_to):
		'''
		Adds an edge with specific rules:
		  - If both nodes are generic nodes (represented as a lowercase letters),
		an edge is added to `node_edges`;
		  - If one of the nodes is a gateway (represented as an uppercase letter),
		an edge is added to both `gateway_edges` and `node_edges`.
		`gateway_edges` represents the edges in the format `a-A`,
		where `a` is a generic node and `A` is a gateway.
		  - If both nodes are gateways, an exception is raised.
		
		Nodes are also stored in `nodes` and `gateway` sets.
		Basic input validation happens before this function is called.

		Args:
			node_from (str): A node in a graph
			node_to (str): A node in a graph

		Raises:
			ValueError: If both nodes are gateways (uppercase letters)
		'''
		if node_from.isupper():
			if node_to.isupper():
				raise ValueError("Gateway-to-gateway connections are not allowed.\nReceived: {0}-{1}".format(
					node_from, node_to
				))
			self.node_edges[node_to].add(node_from)
			self.nodes.add(node_to)
		
			self.gateway_edges[node_from].add(node_to)
			self.gateways.add(node_from)
		
		elif node_to.isupper():
			if node_from.isupper():
				raise ValueError("Gateway-to-gateway connections are not allowed.\nReceived: {0}-{1}".format(
					node_from, node_to
				))
			self.node_edges[node_from].add(node_to)
			self.nodes.add(node_from)

			self.gateway_edges[node_to].add(node_from)
			self.gateways.add(node_to)

		else:
			self.node_edges[node_from].add(node_to)
			self.node_edges[node_to].add(node_from)
			self.nodes.add(node_from)
			self.nodes.add(node_to)
	
	def sever_gateway(self, gateway, node):
		self.gateway_edges[gateway].discard(node)
		self.node_edges[node].discard(gateway)
		# self.gateways.discard(gateway)

	def _build_graph(self):
		if self.graph_as_text is None or self.graph_as_text == "":
			print("Invalid input graph format")
			raise ValueError("Invalid input graph format")
		
		graph_rows = self.graph_as_text.replace("\t", "").replace(",", "").strip().splitlines()
		suffix = ", "
		self._graph_as_text_simple = ""

		for row in graph_rows:
			node_from, node_to = row.strip().split("-")
			self.add_edge(node_from, node_to)
			self._graph_as_text_simple += row + suffix
		
		self._graph_as_text_simple = self._graph_as_text_simple.strip(suffix)

	def _build_subgraphs(self):
		visited = set()
		# NOTE:  Format: dict{ < first_node: set_of_connected_nodes >, ... }
		self.subgraphs = defaultdict(set)
		self.subgraphs_amount = 0
		self.max_subgraph_size = 0
		
		for node_initial in self.node_edges:
			self.subgraphs[node_initial] = set(node_initial)
			queue = [node_initial]
			visited.add(node_initial)
			subgraph_size_current = 0
			while len(queue) > 0:
				node_current = queue.pop(0)
				if node_current in visited:
					continue
				subgraph_size_current += 1
				visited.add(node_current)
				self.subgraphs[node_initial].add(node_current)
				for neighbor in self.node_edges[node_current]:
					queue.append(neighbor)

			self.subgraphs_amount += 1
			self.max_subgraph_size = max(self.max_subgraph_size, subgraph_size_current)

	def _build(self, graph_as_text: Optional[str] = None):
		if graph_as_text is not None and graph_as_text.strip() != "":
			self.graph_as_text = graph_as_text
			if not self._is_valid_input(graph_as_text):
				print("Invalid input graph format")
				raise ValueError("Invalid input graph format")
		
		if self.graph_as_text is None or self.graph_as_text == "":
			print("Invalid input graph format")
			raise ValueError("Invalid input graph format")

		self._build_graph()

		self._build_subgraphs()

	def _reset(self):
		self.node_edges = deepcopy(self._node_edges_initial)
		self.gateway_edges = deepcopy(self._gateway_edges_initial)
		self.gateways = deepcopy(self._gateways_initial)

	def get_all_neighbors(self, node):
		return self.node_edges[node] | self.gateway_edges[node]

	def is_graph_connected(self) -> bool:
		if self.subgraphs_amount == 1:
			return True
		elif self.subgraphs_amount < 1:
			print("No subgraphs found")
			raise ValueError("No subgraphs found")

		return False
	
	# NOTE: What should this function return though?
	# NOTE: Dataclass with specific properties
	#		How to interpret them is up to the caller
	def bfs(self, node_start: str = 'a', node_targets: Optional[set[str]] = None, early_exit: bool = True):
		result = BFSResult(set(), None, 0, {node_start: None})
		# result.targets_found = {None}
		# result.distance = 0
		# result.parents = {node_start: None}

		if node_targets is None:
			targets = self.gateways
		else:
			targets = node_targets

		if len(targets) == 0:
			return result

		# queue = [node_start]
		# deque out of a list of tuples
		queue = deque([(node_start, 0)])
		visited = set()
		visited.add(node_start)

		while len(queue) > 0:
			node_current, depth = queue.popleft()

			if (early_exit and
	   			result.distance_shortest is not None and
				result.distance_shortest != 0 and
				# result.distances[node_current] is not None and
	   			# result.distances[node_current] != 0 and
	   			depth > result.distance_shortest):
				break

			if node_current in targets:
				if result.distances is None:
					result.distances = dict()
				result.distances[node_current] = depth
				if result.distance_shortest == 0:
					result.distance_shortest = depth
				if result.targets_found is None:
					result.targets_found = set()
				result.targets_found.add(node_current)

			for neighbor in sorted(self.node_edges[node_current]):
				# NOTE: This check ensures alphabetical order of the parent nodes
				#		Doing reverse lookup later on will only provide the priority path,
				#		since no other would be recorded
				if neighbor not in visited:
					visited.add(neighbor)
					result.parents[neighbor] = node_current
					queue.append((neighbor, depth + 1))

		return result

class Virus:
	'''
	
	'''
	DEFAULT_LIMITS = {
		'OUTPUT': 8
	}
	_graph: Graph
	pos_initial: str = 'a'
	profiler = Profiler()
	_result: BFSResult
	results_history: list[BFSResult]
	steps: list[State]

	def __init__(self, source: Optional[str | List[str] | list[str] | 'Graph'] = None, *,
			  import_as_text: Optional[str] = None, path: str = "", graph: Optional['Graph'] = None):
		if import_as_text is not None and import_as_text != "":
			self._graph = Graph(import_as_text)
		elif path is not None and path != "":
			self._graph = Graph(path)
		elif graph is not None:
			self._graph = Graph(graph)
		elif source is not None:
			self._graph = Graph(source)
		else:
			print("No input provided")
			raise ValueError("No input provided")
		
		self.steps = list()
		self.results_history = list()
		self._result = BFSResult(set(), None, 0, {self.pos_initial: None})


	# PROPERTIES
	@property
	def graph(self):
		return self._graph

	@graph.setter
	def graph(self, graph: Graph):
		if (	graph.nodes is not None and len(graph.nodes) > 0
	  			and graph.node_edges is not None and len(graph.node_edges) > 0):
			self._graph = graph
		else:
			print("Could not find graph nodes or edges", file = sys.stderr)
			return None
	
	@property
	def nodes(self):
		return sorted(self._graph.nodes)
	
	@property
	def node_edges(self):
		edges_string = "" if len(self._graph.node_edges) < self.DEFAULT_LIMITS['OUTPUT'] else "{ "
		suffix = "\n" if len(self._graph.node_edges) < self.DEFAULT_LIMITS['OUTPUT'] else ", "
		postfix = "" if len(self._graph.node_edges) < self.DEFAULT_LIMITS['OUTPUT'] else " }"
		for node, neighbors in self._graph.node_edges.items():
			edges_string += f"'{node}': {neighbors}" + suffix
		return (edges_string.strip(", ") + postfix).strip()
	
	@property
	def gateways(self):
		return sorted(self._graph.gateways)
	
	@property
	def gateway_edges(self):
		edges_string = "" if len(self._graph.gateway_edges) < self.DEFAULT_LIMITS['OUTPUT'] else "{ "
		suffix = "\n" if len(self._graph.gateway_edges) < self.DEFAULT_LIMITS['OUTPUT'] else ", "
		postfix = "" if len(self._graph.gateway_edges) < self.DEFAULT_LIMITS['OUTPUT'] else " }"
		for node, neighbors in self._graph.gateway_edges.items():
			edges_string += f"'{node}': {neighbors}" + suffix
		return (edges_string.strip(", ") + postfix).strip()
	
	@property
	def result(self):
		if self.result is not None:
			return self._result
		else:
			print("No results were found", file = sys.stderr)
			return None
		
	# REPRESENTATION
	def __repr__(self):
		return self.get_state_readable()

	def __str__(self):
		return self.get_graph_graphical()

	def get_state_readable(self, step: int = -1):
		if self.steps is None or len(self.steps) == 0:
			return None

		state_as_string = ""
		suffix			= " | "
		state_as_string += str(step) + suffix


		if (self.steps[step].position is None):
			state_as_string += "..." + suffix
		else:
			state_as_string += str(self.steps[step].position) + suffix

		if (self.steps[step].priority_path_current is None):
			state_as_string += "..." + suffix
		else:
			state_as_string += "{0:<25}".format(str(self.steps[step].priority_path_current)) + suffix
			# state_as_string += str(self.steps[step].priority_path_current) + suffix

		if (self.steps[step].severed_edge is None):
			state_as_string += "..."
		else:
			state_as_string += str(self.steps[step].severed_edge)

		# state_as_string += str(self.steps[step].position) + suffix
		# state_as_string += str(self.steps[step].priority_path_current) + suffix
		# state_as_string += str(self.steps[step].severed_edge) + suffix
		# # state_as_string += str(self.steps[step].priority_path_next)

		return state_as_string

	def get_bfsresult_readable(self):
		if self._result is not None:
			return(f"Targets: {self._result.targets_found}"
				   f"Distance: {self._result.distance_shortest}"
				   f"Parents: {self._result.parents}")
		else:
			return ("Targets: N/D\nDistance: N/D\nParents: N/D")

	def get_graph_graphical(self):
		counter = 0
		state = ""
		visited = set(tuple())
		suffix = "\n"
		for node, neighbors in self._graph.node_edges.items():
			for neighbor in neighbors:
				if (node, neighbor) not in visited:
					counter += 1
					state += f"{node}-{neighbor}" + suffix
					visited.add((node, neighbor))
					visited.add((neighbor, node))

		if counter > self.DEFAULT_LIMITS['OUTPUT']:
			state = state.replace("\n", ", ").strip(", ")
		state = state.strip(suffix)

		return state

	def get_priority_string_width(self, width_min = 4, width_max = 53, width_item = 2, *, debug: bool = False):
		width = width_min
		reduction = 1
		for step in self.steps:
			if step.priority_path_current is not None:
				width = min(max(width,
									(len(step.priority_path_current) * width_item - reduction)
								),
				 				width_max
							)
				if reduction == 1:
					reduction = 3

		if debug:
			print(f"[DEBUG] [get_priority_string_width] Priority string width: {width}")

		return width

	def convert_from_list_to_str(self, lst: list):
		string = ""
		suffix = "-"
		if len(lst) > 1:
			for item in lst:
				string += item + suffix
		string = string.strip(suffix)

		return string

	def convert_from_dict_to_str(self, dictionary: Dict | dict | defaultdict):
		string = ""
		suffix = "-"
		postfix = ", "
		if len(dictionary) > 0:
			for key, value in dictionary.items():
				string += key + suffix + value
				if len(dictionary) > 1:
					string += postfix
		string = string.strip(postfix)

		return string

	def get_steps_history(self, debug: bool = False):
		if self.steps is None or len(self.steps) == 0:
			return None

		suffix					= " | "
		steps_as_text			= []
		priority_width_current	= self.get_priority_string_width(
				width_min = 4, width_max = 53, debug = debug
			)
		counter = 0
		for step in self.steps:
			steps_as_text.append(str(counter) + suffix)

			if (step.position is None):
				steps_as_text[counter] += "..." + suffix
			else:
				steps_as_text[counter] += str(step.position) + suffix

			if (step.priority_path_current is None):
				steps_as_text[counter] += "..." + suffix
			else:
				if counter == 0:
					priority_path = self.convert_from_list_to_str(step.priority_path_current)
				else:
					priority_path = self.convert_from_list_to_str(step.priority_path_current[1:])

				steps_as_text[counter] += "{0:<{w}}".format(
					priority_path,
					w = priority_width_current) + suffix
				# state_as_string += str(self.steps[step].priority_path_current) + suffix

			if (step.severed_edge is None):
				steps_as_text[counter] += "..." + suffix
			else:
				steps_as_text[counter] += self.convert_from_dict_to_str(step.severed_edge) + suffix

			if (step.priority_path_next is None):
				steps_as_text[counter] += "..."
			else:
				priority_path = self.convert_from_list_to_str(step.priority_path_next)
				if priority_path == "":
					priority_path = "END"
				
				steps_as_text[counter] += "{0:<{w}}".format(
					priority_path,
					w = priority_width_current)
				
				# steps_as_text[counter] += str(step.priority_path_next) + suffix

			counter += 1
		
		return steps_as_text
	
	def display_steps_history(self, debug: bool = False):
		steps_as_string = self.get_steps_history(debug)
		if steps_as_string is None:
			print("No steps history found")
		else:
			print("\n".join(steps_as_string))

	# FUNCTIONAL
	def _get_priority_target(self):
		if self._result is None or self._result.targets_found is None:
			raise Exception("No current BFS were found")

		return sorted(self._result.targets_found)[0]

	def _get_priority_path(self, target: str):
		if (self._result is None or
			self._result.parents is None or
			self._result.distances is None or len (self._result.distances) == 0 or
			target is None or target == ""):
			raise Exception(f"No priority path found for target: {self._target}")
		
		node_current = target
		priority_path = []
		priority_path.append(node_current)
		for step in range(self._result.distances[target]):
			if node_current == self.pos_current:
				break

			# NOTE: Annoying particularity of Python not realizing that 'None'
			#		will never be reached because of the check above
			if node_current is not None:
				node_current = self._result.parents[node_current]
				priority_path.append(node_current)

		priority_path.reverse()
		return priority_path

	def move(self, priority_path: list | List):
		if priority_path is None or len(priority_path) < 2:
			return None
		else:
			return priority_path[1]
	
	def _update_history(self, step_counter, *, debug: bool = False, verbose: bool = False):
		self.results_history.append(self._result)
		self.steps.append(State(
			position = self.pos_current,
			target = self._target,
			priority_path_current = self._priority_path,
			severed_edge = self._severed_edge,
			priority_path_next = None
		))
		# Retrofit the new priority path into the previous record
		# (Handles the special case of the 0th step)
		if step_counter > 0 and len(self.steps) > 0:
			self.steps[step_counter - 1].priority_path_next = self._priority_path

		if debug:
			print(self.get_state_readable(step_counter))

	def game_loop(self, debug: bool = False, verbose: bool = False):
		self.pos_current = self.pos_initial
		self._graph._reset()
		self._output = ""
		self._result = self._graph.bfs(self.pos_current)
		self.results_history = []
		self.steps = list()
		step_counter = 0
		
		while (self._result.targets_found is not None and
			  len(self._result.targets_found) > 0):
			# Step 1: Look for the nearest gateway in a changed graph
			self._result = self._graph.bfs(self.pos_current)

			if (self._result.targets_found is None or
			  len(self._result.targets_found) == 0):
				break
			
			# Step 2: Get the closest gateway
			self._target = self._get_priority_target()

			# Step 3: Get the priority path to it
			self._priority_path = self._get_priority_path(self._target)

			# Step 4: Move the virus
			# NOTE: Skipping the first move because of the specific starting condition
			if step_counter != 0:
				self.pos_current = self.move(self._priority_path)
	
			if self.pos_current is None:
				raise Exception(f"Could not execute the move for the following path: {self._priority_path}")

			if self.pos_current in self._graph.gateways:
				print("You died")
				return False

			# Step 5: Sever one of the gateway edges (based on priority)
			self._severed_edge = { self._target: self._result.parents[self._target] }
			self._graph.sever_gateway(self._target, self._result.parents[self._target])
			
			self._update_history(step_counter, debug = debug)
			step_counter += 1

		if step_counter > 0 and len(self.steps) > 0:
			self.steps[step_counter - 1].priority_path_next = [""]
		
		return True
	
	def solve(self, debug: bool = False, verbose: bool = False):
		if not self.game_loop(debug, verbose):
			result = "Game over"
		else:
			result = ""
			for step in self.steps:
				if len(step.severed_edge) > 1:
					raise Exception(f"Too many severed edges in a step: {step.severed_edge}")
				gateway, node = next(iter(step.severed_edge.items()))
				result += f"{gateway}-{node}\n"

		self._output = result.strip()

		return self._output

#---------------------------------------------------------------------------
# Decorators
# Profiler
profiler = Profiler()

#---------------------------------------------------------------------------
# Utils
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

#---------------------------------------------------------------------------
# Tests
def display_graph_info(virus: Virus, debug: bool = False, verbose: bool = False):
	# print(virus)
	# print(graph.get_state_readable())
	if debug or verbose:
		print("\n Graph reconstructed from the inner state representation:\n")
		# print(virus)
		print(virus)
	if debug:
		print(f"{'Nodes: ':>16} {virus.nodes}")
		print(f"{'Gateways: ':>16} {virus.gateways}")
		print(f"{'Node edges: ':>16}\n{virus.node_edges}")
		print(f"\n{'Gateway edges: ':>16}\n{virus.gateway_edges}")
		# print(f"{'Output: ':>16}\n")
		# if virus._output is not None and virus._output != "":
		# 	print(virus._output)
	# if verbose:
	# 	if debug:
	# 		print()
	# 	print("Alternative display methods:\n")
		
	print()

@profiler
def test_agnostic(title: str = "NO TITLE", subtitle: str = "", input_text: str = "", debug: bool = False, verbose: bool = False):
	# if debug or verbose:
	print()
	print('{:-^67}'.format(title))
	if subtitle is not None and subtitle != "":
		print("\n" + subtitle)
	if debug:
		print("\n [DEBUG] Input:")
		print(input_text)
	
	virus = Virus(input_text)
	display_graph_info(virus, debug, verbose)

	result = virus.solve(debug = debug, verbose = verbose)
	print(f"\n{'Steps: ':>10}")
	virus.display_steps_history(debug = debug)
	print(f"\n{'Output: ':>10}")
	print(result)
	print(f"\n{'-'*67}")

@profiler
def test_default(debug: bool = False, verbose: bool = False):
	try:
		input_text = """
		a-b,
		a-c,
		b-d,
		b-A,
		c-f,
		d-e,
		d-B,
		e-f,
		f-C
		"""
		test_agnostic(title    = "DEFAULT TEST",
					subtitle   = "TESTS: Testing pre-defined input",
					input_text = input_text,
					debug      = debug,
					verbose    = verbose)
		
	except Exception as ex:
		print("Exception occurred while running default test.\n\n  Input text: {0}\n  Exception: {1}\n".format(
			input_text, ex))

@profiler
def test_example(debug: bool = False, verbose: bool = False):
	try:
		example_graphs = {
			'example_1': "a-b\na-c\nb-D\nc-D",
			'example_2': "a-b\nb-c\nc-d\nb-A\nc-B\nd-C",
			'example_3': "a-b\nb-c\nc-d\nc-e\nA-d\nA-e\nc-f\nc-g\nf-B\ng-B",
			'disconnected': "a-b\nb-A\nb-B\nc-C",
			'loop': "a-b\na-c\nb-d\nb-A\nc-f\nd-e\nd-B\ne-f\nf-C"
		}
		example_descriptions = {
			'example_1': "Task Example 1",
			'example_2': "Task Example 2",
			'example_3': "Task Example 3",
			'disconnected': "Example with a disconnected graph",
			'loop': "Example with a loop"
		}
		for example_title, example_graph in example_graphs.items():
			test_agnostic(title    = "EXAMPLES TEST",
						subtitle   = example_descriptions[example_title],
						input_text = example_graph,
						debug      = debug,
						verbose    = verbose)
		
	except Exception as ex:
		print("Exception occurred while running default test.\n\n  Exception: {0}\n".format(
			ex))

@profiler
def test_from_text(input_text: str, debug: bool = False, verbose: bool = False):
	try:
		test_agnostic(title    = "TEST FROM TEXT",
					subtitle   = "TESTS: Testing from text input",
					input_text = input_text,
					debug      = debug,
					verbose    = verbose)

	except Exception as ex:
		print("Exception occurred while testing a graph.\n\n  Graph edges (input): {0}\n  Exception: {1}\n".format(
			input_text, ex))

@profiler
def test_from_file_as_content(input_path: str, debug: bool = False, verbose: bool = False):
	try:
		parent_dir = Path(__file__).parent.resolve()
		input_path_absolute = Path(parent_dir, input_path)
		
		if debug or verbose:
			print("\nInput file path: {0}\n".format(input_path_absolute))
			input_file = str(File.read(input_path_absolute))
			print("Graph read directly from file:\n")
			print(input_file)
		
		test_agnostic(title    = "TEST FROM FILE",
					subtitle   = "TESTS: Testing from file",
					input_text = input_file,
					debug      = debug,
					verbose    = verbose)

	except Exception as ex:
		print("Exception occurred while testing graph from file.\n\n  Path: {0}\n  Exception: {1}\n".format(
			Path(input_path).resolve(), ex))

@profiler
def test_from_file_as_path(input_path: str, debug: bool = False, verbose: bool = False):
	try:
		parent_dir = Path(__file__).parent.resolve()
		input_path_absolute = Path(parent_dir, input_path)

		if debug or verbose:
			print("\nInput file path: {0}\n".format(input_path_absolute))
			input_file = str(File.read(input_path_absolute))
			print("Graph read directly from file:\n")
			print(input_file)
		
		test_agnostic(title    = "TEST FROM FILE",
					subtitle   = "TESTS: Testing from file",
					input_text = str(input_path_absolute),
					debug      = debug,
					verbose    = verbose)

	except Exception as ex:
		print("Exception occurred while testing graph from file.\n\n  Path: {0}\n  Exception: {1}\n".format(
			Path(input_path).resolve(), ex))
		
@profiler
def test_from_dir(input_path: str, debug: bool = False, verbose: bool = False):
	try:
		if debug or verbose:
			print("\nInput dir path: {0}\n".format(Path(input_path).resolve()))

		parent_dir = Path(__file__).parent.resolve()
		path_absolute = Path(parent_dir, input_path).resolve()
		if path_absolute.is_dir():
			input_paths = get_input_paths(str(path_absolute))
			for input_file_path in input_paths:
				file_obj = Path(input_file_path).resolve()
				input_as_text = str(File.read(file_obj))
				test_from_file_as_path(input_file_path, debug = debug, verbose = verbose)
			return

		elif Path(path_absolute).is_file() and Path(path_absolute).suffix == ".txt":
			file_obj = Path(path_absolute).resolve()
			input_as_text = str(File.read(file_obj))
			test_from_file_as_path(file_obj, debug = debug, verbose = verbose)
			return

	except Exception as ex:
		print("Exception occurred while testing graph from file.\n\n  Path: {0}\n  Exception: {1}\n".format(
			Path(input_path).resolve(), ex))

@profiler
def run_tests(args, *, debug: bool = False, verbose: bool = False):
	if args.option not in ARGS_DEF_OPTIONS:
		if args.input_string is not None and args.input_string != "":
			test_from_file_as_content(args.input_string)
		else:
			print(f"[ERROR] TESTS: Invalid option: {args.option}")
			raise ValueError(f"Invalid option: {args.option}")
	else:
		if debug:
			print(f"[DEBUG] TESTS: Parameter passed: {args.option}")

		match args.option:
			case "DEFAULT":
				test_default(debug = debug, verbose = verbose)
			case "EXAMPLE":
				test_example(debug = debug, verbose = verbose)
			case "FROM_FILE":
				if args.test_path is None or args.test_path == "":
					path = DEFAULT_TESTS_FILE
					# print("[ERROR] TESTS: No input path provided")
					# raise ValueError("No input path provided")
				else:
					path = args.test_path
				test_from_file_as_path(path, debug = debug, verbose = verbose)
			case "FROM_DIR":
				if args.test_path is None or args.test_path == "":
					path = DEFAULT_TESTS_DIR
					# print("[ERROR] TESTS: No input path provided")
					# raise ValueError("No input path provided")
				else:
					path = args.test_path
				test_from_dir(path, debug = debug, verbose = verbose)

	return None

#------------------------------------------------------------------------------
# Arguments processing
@profiler
def solve_from_text(input_as_text: str, debug: bool = False, verbose: bool = False):
	try:

		if not input_as_text or len(input_as_text) == 0 or input_as_text == "":
			print("Warning: Empty input received", file = sys.stderr)
			raise Exception("Warning: Empty input received")

		if debug or verbose:
			print("Graph read directly from stdin:\n")
			print(input_as_text)
		
		virus = Virus(input_as_text)
		if debug or verbose:
			display_graph_info(virus, debug, verbose)

		time_start = perf_counter()						# process_time()
		solved = virus.solve(debug = debug, verbose = verbose)
		time_elapsed = perf_counter() - time_start
		if solved is not None and (debug or verbose):
			# print("\n  Min cost: {0:,}".format(solved[1]))
			print(f"\n{'Time elapsed: ':>20} {time_elapsed:>8.3f} s")
			print(f"\n{'Steps: ':>10}")
			virus.display_steps_history(debug = debug)
			print(f"\n{'Output: ':>10}")
			
		print(solved)
	
	except EOFError as ex:
		print("\nEOF received (empty input)", file = sys.stderr)
		return ""

	except Exception as ex:
		raise ex

def solve_from_input(input_path: Optional[str] = None, debug: bool = False, verbose: bool = False):
	# From the cli:   ./isolation.py < input.txt
	#			 cat input.txt | ./isolation.py
	#			 ./isolataion.py (manual)
	try:
		if not input_path:
			if os_name == 'nt':
				help_message = "To exit send Ctrl+Z then Enter"
			elif os_name == 'posix':
				help_message = "To exit send Ctrl+D then Enter"
			else:
				help_message = "To exit enter a new line and type END"
			
			if verbose and stdin.isatty():
				print("Paste or type the set of graph edges below (one per line).")
				print(help_message)
				print()
		
			try:
				input_as_text = stdin.read().strip().split("\n")
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

def invoke_virus_isolation(args):
	try:
		if args.input_string is not None and args.input_string != "":
			# python ./isolation.py graph.txt
			file_try = Path(args.input_string)
			if file_try.is_dir() or file_try.is_file():
				solve_from_input(input_path = args.input_string, debug = args.debug, verbose = args.verbose)
			else:
				print("Input is not a path to file or directory: {0}".format(args.input_string), file = sys.stderr)
				raise Exception("Input is not a path to file or directory: {0}".format(args.input_string))
			
		else:
			# python .\isolation.py
			solve_from_input(debug = args.debug, verbose = args.verbose)
	
	except Exception as ex:
		raise ex

#------------------------------------------------------------------------------
# Arguments parsing
def parse_arguments():
	parser = argparse.ArgumentParser(description = ISOLATION_TITLE)
	parser.add_argument("input_string",		nargs = '?',
					 help = "Path to the input file or folder")
	
	parser.add_argument('-d', "--debug",	 action = "store_true", default = False, 
					 help = "Debug output")
	parser.add_argument('-v', "--verbose",   action = "store_true", default = False,
					 help = "Verbose output")

	# Tests and profiler
	parser.add_argument('-P', "--profiler",  action = "store_true", default = False,
					 help = "Enable profiler")
	parser.add_argument('-T', "--tests",	 action = "store_true", default = False,
					 help = "Invoke standard tests. You can specify what type of tests to run with the `--option` parameter")
	parser.add_argument('-F', "--test_path", type = str,
					 help = "Path to the test file or directory")
	parser.add_argument("-O", "--option",
					choices = ARGS_DEF_OPTIONS,
					default = "DEFAULT",
					help = "Defines what specific tests to run")
	
	return parser.parse_args()

#------------------------------------------------------------------------------
# Main function
def main():
	args = parse_arguments()	
	# NOTE: Manual
	# debug = True
	debug = args.debug
	if args.debug or args.verbose:
		print('\n{:-^67}'.format(ISOLATION_TITLE))
		print('  VERSION: {0}\n'.format(VERSION))

	if args.profiler:
		profiler.enabled = True
	else:
		profiler.enabled = False

	# Some pre-defined tests
	if args.tests:
		run_tests(args = args,
				 debug = args.debug,
			   verbose = args.verbose)
		
	else:
		# The main solver
		invoke_virus_isolation(args)
	
	if args.profiler:
		profiler.summary()

if __name__ == "__main__":
	main()