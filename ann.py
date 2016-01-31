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

"Used to split a list into two based on a percentage"
def split_list(list, percentage):
	length = len(list)
	divider = int(percentage*length)
	list1 = list[0:divider]
	list2 = list[divider:length]
	return list1, list2

"Class definition for Artifical Neural Network"
class ANN():
	def __init__(self, numInputNodes, numHiddenNodes, holdOutPercentage):
		#Define parameters
		self.numInputNodes = numInputNodes
		self.numHiddenNodes = numHiddenNodes
		self.numOutputNodes = 1   #This will never be different
		self.holdOut = holdOutPercentage

		#Create synapse weights
		#The first layer will be a matrix of weights that has rows = numInputNodes, cols = numHiddenNodes
		self.synapseLayer1 = num.random.uniform(0, 1, [self.numInputNodes, self.numHiddenNodes])
		print("synapseLayer1")
		print(self.synapseLayer1)
		#The second layer will be a matrix of weights that has rows = numHiddenNodes, cols = numOutputNodes
		self.synapseLayer2 = num.random.uniform(0, 1, [self.numHiddenNodes, self.numOutputNodes])
		print("synapseLayer2")
		print(self.synapseLayer2)
	#Propagate input through to output
	#Using an input matrix rather than pair value allows us to process clusters of input if we wish
	#This can also be used for a 1x(number of values per point) matrix, handling a single point at a time
	def propagate(self, inputMatrixArray):
		#Apply synapse weighitng to all inputs to get values at hidden layer
		self.hiddenNodeInput = num.dot(inputMatrixArray, self.synapseLayer1)
		#Run activation function
		self.hiddenLayerOutput = self.sigmoid(self.hiddenNodeInput)
		#Propagate hidden layer output values through to last layer
		self.outputNodeInput = num.dot(self.hiddenLayerOutput, self.synapseLayer2)
		#Run activation function on output
		self.finalOutput = self.sigmoid(self.outputNodeInput)
		return self.finalOutput.item(0, 0)

	#Perform sigmoid activation function
	def sigmoid(self, x):
		return (1/1+num.exp(-x))

	def derivativeSigmoid(self, x):
		return num.exp(-x)/((1+num.exp(-x))**2)

	'''Define functions for error computation
	def modifierCompute(self, input, expectedOutput):
		actualOutput = neuralNet.propagate(input)
		actualOutputMatrix = num.matrix(actualOutput)

		errorMatrix2 = -(expectedOutput - actualOutputMatrix)
		backPropError2 =  num.multiply(errorMatrix2, self.finalOutput)
		modifierMatrix2 = num.dot(self.hiddenLayerOutput.T, backPropError2)

		errorMatrix1 = num.dot(backPropError2, self.synapseLayer2.T)*self.derivativeSigmoid(self.hiddenNodeInput)
		modifierMatrix1 = num.doit(input.T, errorMatrix1)

		return modifierMatrix1, modifierMatrix2
	'''
	def modifierCompute(self, input, expectedOutput):
		actualOutput = neuralNet.propagate(input)
		actualOutputMatrix = num.matrix(actualOutput)

		modifierMatrix2 = num.zeros(shape=(5, 1))
		modifierMatrix1 = num.zeros(shape=(5, 2))

		for x in range (0, self.numHiddenNodes):
			#print("Output")
			#print (actualOutputMatrix)

			totalErrorMatrix = num.subtract(expectedOutput, actualOutputMatrix)
			totalErrorMatrix = num.square(totalErrorMatrix)
			totalErrorMatrix = num.divide(totalErrorMatrix, 2)
			totalError = totalErrorMatrix.sum()
			print("Total error")
			print(totalError)

			dTotalToOut = -(expectedOutput-actualOutput)
			print("dTotalToOut")
			print (dTotalToOut)
			dOutToNet = actualOutput*(1-actualOutput)
			print("OutToNet")
			print (dOutToNet)
			dNetToSynapse = self.hiddenLayerOutput.item(0, x)
			print("dNetToSynapse")
			print (dNetToSynapse)
			dTotalToSynapse = dTotalToOut*dOutToNet*dNetToSynapse
			print("dTotalToSynapse")
			print(dTotalToSynapse)

			modifierMatrix2[x, 0] = dTotalToSynapse

		print ("modifierMatrix2")
		print (modifierMatrix2)
		return modifierMatrix2

"Start processing input"
parser = argparse.ArgumentParser()
parser.add_argument("file_name", help="The file name")
parser.add_argument("optional_arguments", help="The option arguments (hidden nodes and holdhout percent)", nargs='*')
args = parser.parse_args()


"Parse the arguments"
file_name = args.file_name
"If no arguments are given, assign default values"
input_nodes = 2
if (int(get_argument(args.optional_arguments, 'h'))):
	hidden_nodes = (int(get_argument(args.optional_arguments, 'h')))/100
else:
	hidden_nodes = 5
if (int(get_argument(args.optional_arguments, 'p'))):
	holdout_percent = (int(get_argument(args.optional_arguments, 'p')))/100
else:
	holdout_percent = 0.20

"The neural net we will use for the rest of the script"
neuralNet = ANN(input_nodes, hidden_nodes, holdout_percent)
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
	points.append({"x_val": float(temp[0]), "y_val": float(temp[1]), "class": float(temp[2][:-1])})

"Separate data into training and test data"
testSet, trainingSet = split_list(points, holdout_percent)

"Go through the training set and train the neural net on each val"
neuralNet.modifierCompute(num.matrix([1.2, 0.4]), 0)
