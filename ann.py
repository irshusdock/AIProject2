import argparse
import numpy as num

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

"Generate the hidden layer of the neural network with random weights on each node"
"hidden_nodes is the number of hidden nodes to use"
def generate_neural_network(hidden_nodes):
	hidden_layer = []
	for x in range(0, hidden_nodes):
		hidden_layer.append({"weightX": num.random.random(), "weightY": num.random.random()})
	return hidden_layer

"Run the forward pass of the neural network for a given point"
"hidden_layer is the layer of hidden nodes to use"
"output_node is the output node to us"
"point is the current point to use"
def run_forward_neural_network(hidden_layer, output_node, point):
	
	out_exp = 0
		
	"For each hidden node, send the output value to the output node"
	for node in hidden_layer:
		"Compute the exponent in the sigmoid function for the node"
		node_exp = node["weightX"] * point["x_val"] + node["weightY"] * point["y_val"]
		"Compute the value of the sigmoid function for the node"
		node_val = 1/(1+num.exp(-node_exp))
		"Add the sigmoid value of the node to the exponent of the output node"
		out_exp = out_exp + node_val
		
	"Compute the sigmoid value for the output node"	
	out_val = 1/(1+num.exp(-out_exp))
	"Round to the nearest number"
	out_val = num.round(out_val)

	return out_val
		

parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="The file name")
parser.add_argument("optional_arguments", help="The option arguments (hidden nodes and holdhout percent)", nargs='*')
args = parser.parse_args()

"Parse the arguments"
file_name = args.file_name
hidden_nodes = int(get_argument(args.optional_arguments, 'h'))
holdout_percent = int(get_argument(args.optional_arguments, 'p'))

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
	points.append({"x_val": float(temp[0]), "y_val": float(temp[1]), "category": float(temp[2][:-1])})

hidden_layer = generate_neural_network(hidden_nodes)
output_node = {"weightX": num.random.random(), "weightY": num.random.random()}

