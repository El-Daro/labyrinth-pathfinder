#pylint:disable=W0312

#---------------------------------------------------------------------------#
# Version: 0.2.2															#
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
#			OPTION				| DEFAULT, EXAMPLE, FROM_FILE
#---------------------------------------------------------------------------#
#---------------------------------CHANGELOG---------------------------------#
#---------------------------------------------------------------------------#
# v0.2.X
#	- 
#---------------------------------------------------------------------------#
# TODO:
#	- Everything
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
VERSION = "0.2.2"
ISOLATION_TITLE = "Virus:Isolation by El Daro"
DEFAULT_LABYRINTHS_DIR = "../graphs"
DEFAULT_SEARCH_DIR = "../graphs/search"
DEFAULT_INVALID_DIR = "../graphs/invalid"

ARGS_DEF_OPTIONS = { "DEFAULT", "EXAMPLE", "FROM_FILE" }

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
		print(f"{'Average time: ':>20} {sum(times) / len(times):>7,.3f} s")
		print(f"{'Min time: ':>20} {min(times):>7.3f} s")
		print(f"{'Max time: ':>20} {max(times):>7.3f} s")
		# print(f"Avg memory:   {sum(memories) / len(memories) / 1024 / 1024:.2f} MB")
		# print(f"Peak memory:  {max(memories) / 1024 / 1024:.2f} MB")
		print(f"{'='*67}")

@dataclass
class BFSResult:
	'''Result of a BFS search'''
	targets_found: set[str | None]
	distance: int | None
	parents: dict[str, str | None]

class Graph:
	'''
	Describes a bidirectional graph and provides its basic methods.
	That's it.
	'''
	node_edges: defaultdict[str, set[str]]
	gateway_edges: defaultdict[str, set[str]]
	nodes: set[str]
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
	
	def sever_gateway(self, node, gateway):
		self.gateway_edges[gateway].discard(node)
		self.node_edges[node].discard(gateway)
		self.gateways.discard(gateway)

	def _build_graph(self):
		if self.graph_as_text is None or self.graph_as_text == "":
			print("Invalid input graph format")
			raise ValueError("Invalid input graph format")
		
		graph_rows = self.graph_as_text.replace("\t", "").replace(",", "").strip().splitlines()
		
		for row in graph_rows:
			node_from, node_to = row.strip().split("-")
			self.add_edge(node_from, node_to)

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

	def get_all_neighbors(self, node):
		return self.node_edges[node] | self.gateway_edges[node]

	def is_graph_connected(self) -> bool:
		if self.subgraphs_amount == 1:
			return True
		elif self.subgraphs_amount < 1:
			print("No subgraphs found")
			raise ValueError("No subgraphs found")

		return False
	
	# TODO: What should this function return though?
	# NOTE: Dataclass with specific properties
	#		How to interpret them is up to the caller
	def bfs(self, node_start: str = 'a', node_targets: Optional[set[str]] = None, early_exit: bool = True):
		result = BFSResult(set(), 0, {node_start: None})
		# result.targets_found = {None}
		# result.distance = 0
		# result.parents = {node_start: None}

		if node_targets is None:
			targets = self.gateways
		else:
			targets = node_targets

		# queue = [node_start]
		# deque out of a list of tuples
		queue = deque([(node_start, 0)])
		visited = set()

		while len(queue) > 0:
			node_current, depth = queue.popleft()

			if (early_exit and
	   			result.distance is not None and
	   			depth > result.distance):
				break

			if node_current in targets:
				if result.distance == 0:
					result.distance = depth
				result.targets_found.add(node_current)

			for neighbor in sorted(self.node_edges[node_current]):
				if neighbor not in visited:
					visited.add(neighbor)
					result.parents[neighbor] = node_current
					queue.append((neighbor, depth + 1))

		return result

class Virus:
	'''
	
	'''
	_graph: Graph
	pos_initial: str = 'a'
	profiler = Profiler()
	_result: BFSResult

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
		
		self.pos_current = self.pos_initial


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
		return self._graph.nodes
	
	@property
	def node_edges(self):
		return self._graph.node_edges
	
	@property
	def gateways(self):
		return self._graph.gateways
	
	@property
	def gateway_edges(self):
		return self._graph.gateway_edges
	
	@property
	def result(self):
		if self.result is not None:
			return self._result
		else:
			print("No results were found", file = sys.stderr)
			return None
		
	def __repr__(self):
		return self.get_state_readable()

	def __str__(self):
		return self.get_state_graphical()

	# TODO: Define for solved and not solved states
	def get_state_readable(self):
		if self._result is not None:
			print(f"Targets: {self._result.targets_found}")
			print(f"Distance: {self._result.distance}")
			print(f"Parents: {self._result.parents}")

	def get_state_graphical(self):
		print("Not implemented yet")

	def move(self):
		self._result = self._graph.bfs(self.pos_current)
		if self._result is not None and self._result.distance is not None:
			self._result
		return self._result
	
	def solve(self):
		return None

#---------------------------------------------------------------------------
# Decorators
# Profiler
profiler = Profiler()

#---------------------------------------------------------------------------
# Tests
def display_graph_info(virus: Virus, debug: bool = False, verbose: bool = False):
	print("\nGraph reconstructed from the inner state representation:\n")
	# print(virus)
	# print(graph.get_state_readable())
	if verbose:
		print(virus.nodes)
		print(f"Nodes: {virus.nodes}")
		print(f"Gateways: {virus.gateways}")
		print(f"Node edges: {virus.node_edges}")
		print(f"Gateway edges: {virus.gateway_edges}")
	if debug:
		print("Alternative display methods:\n")
		virus
	print()

def test_agnostic(title: str = "NO TITLE", subtitle: str = "", input_text: str = "", debug: bool = False, verbose: bool = False):
	if debug or verbose:
		print()
		print('{:-^40}'.format(title))
		if subtitle is not None and subtitle != "":
			print("\n" + subtitle)
	if debug:
		print("\n [DEBUG] Input:")
		print(input_text)
	
	virus = Virus(input_text)
	display_graph_info(virus, debug, verbose)

	virus.solve()
	display_graph_info(virus, debug, verbose)

	return None

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

# TODO: Define examples from the original task
@profiler
def test_example(debug: bool = False, verbose: bool = False):
	print("[ERROR] Examples aren't defined yet")

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
		if debug or verbose:
			print("\nInput file path: {0}\n".format(Path(input_path).resolve()))
			input_file = str(File.read(Path(input_path)))
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
		if debug or verbose:
			print("\nInput file path: {0}\n".format(Path(input_path).resolve()))
			input_file = str(File.read(Path(input_path)))
			print("Graph read directly from file:\n")
			print(input_file)
		
		test_agnostic(title    = "TEST FROM FILE",
					subtitle   = "TESTS: Testing from file",
					input_text = input_path,
					debug      = debug,
					verbose    = verbose)

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
				if args.input_string is None or args.input_string == "":
					print("[ERROR] TESTS: No input path provided")
					raise ValueError("No input path provided")
				test_from_file_as_path(args.input_string, debug = debug, verbose = verbose)

	return None

#------------------------------------------------------------------------------
# Arguments processing


#------------------------------------------------------------------------------
# Arguments parsing
def parse_arguments():
	parser = argparse.ArgumentParser(description = ISOLATION_TITLE)
	parser.add_argument("input_string",		nargs = '?',
					 help = "Path to the input file or folder")
	
	parser.add_argument('-d', "--debug",	 action = "store_true", default = True, 
					 help = "Debug output")
	parser.add_argument('-v', "--verbose",   action = "store_true", default = True,
					 help = "Verbose output")

	# Tests and profiler
	parser.add_argument('-P', "--profiler",  action = "store_true",
					 help = "Enable profiler")
	parser.add_argument('-T', "--tests",	 action = "store_true", default = True,
					 help = "Invoke standard tests. You can specify what type of tests to run with the `--option` parameter")
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
	debug = True
	# debug = args.debug
	if debug:
		print('\n{:-^67}'.format(ISOLATION_TITLE))
		print('  VERSION: {0}\n'.format(VERSION))

	if args.profiler:
		profiler.enabled = True
	else:
		profiler.enabled = False

	# Some pre-defined tests
	if args.tests:
		run_tests(args = args,
				 debug = True,
			   verbose = True)
	
	if args.profiler:
		profiler.summary()

if __name__ == "__main__":
	main()