#pylint:disable=W0312
#===========================================================================#
#  Version: 0.9.0                                                           #
#     Name: Labyrinth Pathfinder UI                                         #
#   Author: El Daro                                                         #
# Assuming hash can hash cache, how many cache hashes hash has?             #
#---------------------------------------------------------------------------#
#-------------------------COMMAND LINE ARGUMENTS----------------------------#
#---------------------------------------------------------------------------#
#  positional arguments:                                                    #
#    input_path                  | Path to the input file or folder         #
#                                                                           #
#  options:                                                                 #
#    -h, --help                  | show this help message and exit          #
#    -d, --debug                 | Debug output                             #
#    -v, --verbose               | Verbose output                           #
#                                                                           #
#   GENERATOR                                                               #
#    Use it to automatically generate and solve new labyrinths              #
#    -G, --generate GENERATE                                                #
#        Number of labyrinths to generate                                   #
#    -N, --generate_nonstandard                                             #
#        Generate labyrinths with various depths,                           #
#        numbers of rooms and hallway lengths                               #
#                                                                           #
#   GENERATOR PARAMETERS                                                    #
#    This group has no effect if --generate_nonstandard is used             #
#    -C, --count COUNT            | Number of labyrinths to generate        #
#    -D, --depth DEPTH            | Room depth                              #
#    -R, --rooms ROOMS            | Number of rooms to generate             #
#        (has no effect if --generate_nonstandard is used)                  #
#    -H, -L, --hallway_length HALLWAY_LENGTH                                #
#         Hallway length                                                    #
#    -P, --profiler               | Enable profiler                         #
#    -T, --tests                  | Invoke standard tests from a            #
#        pre-defined input folder (comes with the repo)                     #
#---------------------------------------------------------------------------#
#---------------------------------CHANGELOG---------------------------------#
#---------------------------------------------------------------------------#
# 2025.10.22                                                                #
# - Added decoding from compacted state to readable and graphical formats   #
#                                                                           #
# 2025.10.23                                                                #
# - Finalized input processing and parsing                                  #
# - Decided on the state representation (State class)                       #
# - Changed the processing rules to accomodate any valid initial state      #
# - Added simple test functions                                             #
# - Added main method of testing: put your labyrinths as files in the       #
#     ..\labyrinths folder                                                  #
# - Implemented user interface                                              #
# - Refactored UI to a separate file                                        #
# - Implemented labyrinth generator                                         #
# - Added support for different room sizes                                  #
# - Added support for different hallway lengths                             #
# - Added support for different start positions                             #
# - Added support for different room depth                                  #
#                                                                           #
# 2025.10.24                                                                #
# - Fixed Generator                                                         #
# - Improved generator                                                      #
# - Added history of generated states                                       #
# - Fixed generator once again                                              #
# - Added generator tests                                                   #
# - Implemented goal state                                                  #
# Slept a bit                                                               #
# - Added room positions                                                    #
# - Improved calculation of accessible hallway nodes                        #
# - SOLVER:                                                                 #
#   - Logic for pathfinding and cost calculation                            #
#   - Implemented solve() method                                            #
#   - Implemented get_possible_moves() method                               #
#   - Implemented _moves_room_to_hallway() method                           #
#   - Implemented _moves_room_to_room() method                              #
#   - Implemented _try_move_to_destination() method                         #
#   - Implemented _is_path_clear() method                                   #
#   - Modified _moves_room_to_hallway() method                              #
#   - Implemented _moves_hallway_to_room() method                           #
#                                                                           #
#   - Fixed energy cost computation and heuristics                          #
#   - Fixed move generation and search logic                                #
#   - Fixed it again                                                        #
#   - And again                                                             #
#   - It works                                                              #
#                                                                           #
# 2025.10.25                                                                #
#   - Added profiler                                                        #
#   - Added a simpler way to measure execution time (aside from profiler)   #
#   - Fixed some visual representation                                      #
#   - Cleaned up the code, deleted obsolete stuff                           #
#   - Fixed various issues                                                  #
#   - Implemented command line arguments                                    #
#---------------------------------------------------------------------------#
# TODO:                                                                     #
#    - REPO                                                                 #
#    - State history                                                        #
#    - Binary encoding                                                      #
#        Each node is represented by a 3-bit binary number                  #
#          b000 - . (empty)                                                 #
#          b001 - A                                                         #
#          b010 - B                                                         #
#          b011 - C                                                         #
#          b100 - D                                                         #
#---------------------------------------------------------------------------#

#---------------------------------------------------------------------------#
# Find the minimal cost of reaching the goal state, where goal state is:    #
# hallway: (0**hallway_length)                                              #
# rooms: roomN = decoded(N + 1)                                             #
#   0: always represents an empty cell                                      #
#   1: A                                                                    #
#   2: B                                                                    #
#   ...                                                                     #
#   N: index + 1                                                            #
#                                                                           #
# Weights per one node move                                                 #
# A:    1                                                                   #
# B:   10                                                                   #
# C:  100                                                                   #
# D: 1000                                                                   #
# ...                                                                       #
# N: encoded(N)**10                                                         #
#                                                                           #
# Other constraints:                                                        #
# Execution time < 10 s                                                     #
# Memory usage < 200 MB                                                     #
#                                                                           #
# Visual example of an initial state:                                       #
# #############                                                             #
# #...........#                                                             #
# ###B#C#B#D###                                                             #
#   #D#C#B#A#                                                               #
#   #D#B#A#C#                                                               #
#   #A#D#C#A#                                                               #
#   #########                                                               #
#                                                                           #
# Visual example of a goal state:                                           #
# #############                                                             #
# #...........#                                                             #
# ###A#B#C#D###                                                             #
#   #A#B#C#D#                                                               #
#   #A#B#C#D#                                                               #
#   #A#B#C#D#                                                               #
#   #########                                                               #
#                                                                           #
# Min cost: 44169                                                           #
#                                                                           #
#---------------------------------------------------------------------------#

from os import system, _exit, name as os_name, path as os_path
from pathlib import Path
from time import time
from msvcrt import getch

from pathfinder import *

#----------------------------------------
# DEFAULTS
VERSION = "0.9.0"
DEFAULT_LABYRINTHS_DIR = "../labyrinths"
ERROR_CODES = {
	"NotANumber": 1,
	"OutOfRange": 2
}

#---------------------------------------------------------------------------
# Classes
#---------------------------------------------------------------------------
# Log class
class Log():
	'''
	Creates a log file, measures performance
	
	Args:
		file_name (string): Path to the log file
	'''

	DEFAULT_FILE_NAME		= "stats.log"
	absolute_file_path		= "stats.log"
	logging					= True
	dir_name				= Path.cwd()
	
	def __init__(self, file_name = DEFAULT_FILE_NAME):
		self.file_name			= file_name
		self.dir_name			= Path.cwd()
		# self.file_name			= File.get_valid_name(self.file_name, self.DEFAULT_FILE_NAME, is_file = True, extension = ".log")
		# self.absolute_file_path = File.get_absolute_path(self.file_name, self.dir_name)
		self.logging			= True
		self.start_time			= None
		self.end_time			= None
		self.time_str			= ""
	
	def start(self):
		self.start_time		= time()

	def end(self):
		# self.end_time		= time() - self.start_time
		self.time_str		= '{:.2f} {:}'.format(self.end_time, "s")

	def display(self):
		'''
		Use this template to output logging information
		'''
		print("\n")
		print("{0:>20}: {1:<9}".format("Elapsed time",	str(self.time_str)))
		print("")

	def write_log(self):
		try:
			# self.file_name = name
			
			with open(self.absolute_file_path, "a", encoding = "utf-8") as f:
				f.write("\nElapsed time: " + self.time_str)
				f.write('\n\n{:-^20}'.format(""))
				f.write("\n")

		except Exception as e:
			print("\nLog file has not been updated")
			print(e)

#---------------------------------------------------------------------------
# User Interface
def clear_screen():
	if os_name == 'nt':			# Clears screen in normal way
		system('cls')			# For Windows-systems
	elif os_name == 'posix':	# Or for Unix-systems
		system('clear')
	else: 						# Clears screen in a desperate way
		print("\n" * 50)

def print_title(sub_title: str = ""):
	print('{:-^67}'.format("Labyrinth Pathfinder by El Daro"))
	if sub_title is not None and sub_title != "":
		print(str(sub_title))
	print("")

def display_menu_main():
	clear_screen()
	print_title()
	print("1 - Run tests with default input directory")
	print("2 - Run predefined example")
	print("3 - Process a labyrinth from file")
	print("4 - Process a manually entered labyrinth")
	print("5 - Generate input labyrinth (not implemented)")
	print("6 - Config")
	print("0 - Exit")

def display_sub_menu():
	print('\n{:-^120}'.format(""))
	print("\n1 - Back to main menu")
	print("0 - Exit")

def sub_menu():
	'''
	This function shows a sub menu and asks user
	whether to go back to main menu or exit the program
	'''
	display_sub_menu()
	incorrectInput = True
	while incorrectInput: 
		user_input = str(input("> ")).strip()
		if user_input == '1':						# Go back to main menu
			incorrectInput = False
			return None
		elif user_input == '0':						# Exit program
			_exit(0)
		else:
			print("Incorrect input. Try again")

# 1: Default tests from directory
def display_default_tests(labyrinths_dir: str):
	clear_screen()
	print_title("Running default tests from directory: {0}".format(labyrinths_dir))

# 2: Predefined example labyrinth
def display_predefined_example():
	clear_screen()
	print_title("Running predefined example labyrinth")

# 3: Manual path selection interface
def display_test_from_file():
	clear_screen()
	print_title("Test labyrinth from file")
	print("Input a file name or a full path to the file")
	print("To go back enter \"back\" without quotes\n")

# 4: Manual labyrinth input interface
def display_enter_input_and_run():
	clear_screen()
	print_title("Enter input labyrinth as text manually")
	print("Input the labyrinth line by line.")
	print("To finish input, enter an empty line.\n")

# 5: Generate labyrinth input interface
def display_generate_input_and_run():
	clear_screen()
	print_title("Generate a labyrinth and process it")

# 6: Configuration menu interface
def display_menu_config():
	clear_screen()
	print_title("Configuration menu")
	print("1 - Enable\\Disable logging")
	print("2 - Show log file")
	print("3 - Set room depth for generated labyrinths")
	print("4 - Back to main menu")
	print("0 - Exit")

def display_generator_set_room_depth(gen):
	clear_screen()
	print_title("Set room depth for generated labyrinths")
	print("Enter the depth of the rooms for labyrinths to generate with")
	print("Note that you can't set minimum to less than 2 and maximum to more than 4.")
	print("To go back enter \"back\" without quotes\n")
	print("Current min = " + str(gen.min_depth))
	print("Current max = " + str(gen.max_depth) + "\n")

#------------------------------------
# Option 1
def run_default_tests(labyrinths_dir: str = DEFAULT_LABYRINTHS_DIR):
	display_default_tests(labyrinths_dir)
	test_all_labyrinths(labyrinths_dir)
	sub_menu()

# Option 2
def run_predefined_example():
	'''Runs a predefined example labyrinth test.'''
	display_predefined_example()
	example_labyrinth = (
		"#############\n"
		"#...........#\n"
		"###B#C#B#D###\n"
		"  #A#D#C#A#  \n"
		"  #########  "
	)

	display_labyrinth_info(Labyrinth(example_labyrinth))
	sub_menu()

# Option 3
def test_from_file():
	'''Prompt for a file path and process that labyrinth file.'''
	parent_dir = Path(__file__).parent.resolve()
	user_input = ''

	while user_input == '':
		display_test_from_file()
		user_input = str(input(str(parent_dir) + os_path.sep)).strip()
	if user_input.lower() == "back":
		return None
	else:
		try:
			input_path = Path(parent_dir, user_input)
			test_labyrinth(input_path)
		except Exception as ex:
			print(f"Failed to process file: {ex}")
	
	sub_menu()

# Option 4
def enter_input_and_run():
	'''Prompts user to enter a labyrinth as text and processes it.'''
	display_enter_input_and_run()

	labyrinth_lines = []
	while True:
		line = str(input()).strip('\n')
		if line == "":
			break
		labyrinth_lines.append(line)
	
	labyrinth_as_text = "\n".join(labyrinth_lines)
	try:
		labyrinth = Labyrinth(labyrinth_as_text)
		display_labyrinth_info(labyrinth)
	except Exception as ex:
		print(f"Failed to process the labyrinth: {ex}")

	sub_menu()

# Option 5
def generate_input_and_run():
	'''Generates an input labyrinth and runs it'''
	display_generate_input_and_run()
	gen = Generator()
	labyrinth = gen.new_labyrinth()
	display_labyrinth_info(labyrinth)
	sub_menu()

# Option 6
# def invoke_menu_config():
# 	'''Invokes the configuration menu'''
# 	incorrect_input = False
# 	user_input = ''
# 	# log_shown = ''
# 	while True: 
# 		display_menu_config()
# 		if Log.logging:
# 			print("\nLogging enabled")
# 		else:
# 			print("\nLogging disabled")
		
# 		if (incorrect_input):
# 			print("\nIncorrect input. Try again")
# 			incorrect_input = False
# 		else:
# 			print("\n")

# 		user_input = str(input("> ")).strip()
# 		if user_input == '1':					# 
# 			Log.logging = not Log.logging		# Enable\\Disable logging
# 			continue						
# 		elif user_input == '2':					# 
# 			show_log(log)						# Show log file
# 		elif user_input == '3':					# 
# 			generator_set_room_depth(gen)		# Set room depth for generated labyrinths
# 		elif user_input == '4':
# 			incorrect_input = False
# 			break
# 		elif user_input == '0':
# 			_exit(0)
# 		else:
# 			incorrect_input = True
# 	sub_menu()

# Log
def show_log(log):
	clear_screen()
	print_title("Log")
	print("Not implemented yet")
	sub_menu()

# AutoGenerator
def generator_set_room_depth(gen):
	'''Set room depth for generated labyrinths'''
	# 0: No error | 1: Not a number | 2: Out of range
	number_error_code = 0
	user_input = ''
	while True:
		display_generator_set_room_depth(gen)
		if number_error_code == ERROR_CODES["NotANumber"]:
			print("Not a number!")
		elif number_error_code == ERROR_CODES["OutOfRange"]:
			print("Entered depth is out of allowed range: {0}..{1}".format(
				gen.DEPTH_MIN, gen.DEPTH_MAX))
		else:
			print("")
		user_input = str(input("New depth = ")).strip()
		if user_input.lower() == "back":
			return None
		if not user_input.isdigit():
			user_input = ''
			number_error_code = ERROR_CODES["NotANumber"]		# = 1
		elif int(user_input) < gen.DEPTH_MIN or int(user_input) > gen.DEPTH_MAX:
			user_input = ''
			number_error_code = ERROR_CODES["OutOfRange"]		# = 2
		else:
			gen.depth = int(user_input)
			break

def menu_main():
	log				= Log()
	log.logging		= True

	incorrect_input	= False
	main_loop		= True
	while main_loop:
		display_menu_main()
		if (incorrect_input):
			print("\nIncorrect input. Try again")
			incorrect_input = False
		else:
			# print("\n")
			print("")
		user_input = str(input("> ")).strip()
		if user_input == '1':
			run_default_tests()							# Run tests with default input directory
		elif user_input == '2':							# 
			run_predefined_example()					# Run a predefined labyrinth example
		elif user_input == '3':							# 
			test_from_file()							# Read a labyrinth from file and process it
		elif user_input == '4':							# 
			enter_input_and_run()						# Generate input labyrinth (as text) and run it
		elif user_input == '5':							# 
			generate_input_and_run()					# Generate input labyrinth (as text) and run it
		elif user_input == '6':							# 
			print("Not implemented yet")
			getch()
		# 	invoke_menu_config()						# Config menu
		elif user_input == '0':							# Exit program							
			main_loop = False
			_exit(0)
		else:
			incorrect_input = True


if __name__ == "__main__":
	menu_main()