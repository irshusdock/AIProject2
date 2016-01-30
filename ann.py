import argparse
import numpy

"Parse through the list of arguements and return the value of the passed argument"
"list_of_args is the list of arguments in the format [arg1, val1, arg2, val2, etc]"
"arg is the argument to search for (e.x. arg1, arg2 in above)"
def get_argument(list_of_args, arg):
    if(not list_of_args):
        return 0
    counter = 0
    for element in list_of_args:
        if element == arg:
            return list_of_args[counter+1]
        counter = counter + 1
    return 0

def generate_neural_network(list_of_points, hidden_nodes):
	hidden_layer = []
	for x in range(0, hidden_nodes):
		return
		

parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="The file name")
parser.add_argument("optional_arguments", help="The option arguments (hidden nodes and holdhout percent)", nargs='*')
args = parser.parse_args()

"Parse the arguments"
file_name = args.file_name
hidden_nodes = get_argument(args.optional_arguments, 'h')
holdout_percent = get_argument(args.optional_arguments, 'p')

"Read the file and store it as a list of lines"
with open(file_name) as f:
	file_content = f.readlines()

"Store the lines as a list of dictionaries"
points = []
for line in file_content:
	temp = line.split(" ");
	"Skip all blank lines"
	if(temp[0] == "\n"):
		continue;

	"The last character of the line will be \n so it must be removed"
	points.append({"x_val": temp[0], "y_val": temp[1], "class": temp[2][:-1]})
 