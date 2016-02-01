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
"input_list is the list to split"
"percentage is the decimal representation of the percentage to split"
def split_list(input_list, percentage):
	length = len(input_list)
	divider = num.floor(percentage*length)
	divider = int(divider)
	list1 = []

	x = 0
	while(x < divider): 
		index = num.random.randint(0, divider)
		list1.append(input_list[index])
		del input_list[index]
		divider = divider-1
	return list1, input_list

"Class definition for Artifical Neural Network"
"numInputNodes is the number of input nodes"
"numOutputNodes is the number of output nodes"
class ANN():
	def __init__(self, numInputNodes, numHiddenNodes):
		#Define parameters
		self.numInputNodes = numInputNodes
		self.numHiddenNodes = numHiddenNodes
		self.numOutputNodes = 1   #This will never be different

		#Create synapse weights
		#The first layer will be a matrix of weights that has rows = numInputNodes, cols = numHiddenNodes
		#This is the matrix of weights from input nodes to hidden nodes
		self.synapseLayer1 = num.random.uniform(-1, 1, [self.numInputNodes, self.numHiddenNodes])
		#The second layer will be a matrix of weights that has rows = numHiddenNodes, cols = numOutputNodes
		#This is the matrix of weights from hidden nodes to output nodes
		self.synapseLayer2 = num.random.uniform(-1, 1, [self.numHiddenNodes, self.numOutputNodes])
	
	''' Propagate input through to output
		Using an input matrix rather than pair value allows us to process clusters of input if we wish
		This can also be used for a 1x(number of values per point) matrix, handling a single point at a time
		inputMatrixArray is the matrix of inputs to use (points with x y values)
	'''
	def propagate(self, inputMatrixArray):
		self.inputMatrixArray = inputMatrixArray
		
		#Apply synapse weighitng to all inputs to get values at hidden layer
		self.hiddenNodeInput = num.dot(inputMatrixArray, self.synapseLayer1)
		
		#Run activation function
		self.hiddenLayerOutput = self.sigmoid(self.hiddenNodeInput)
		
		#Propagate hidden layer output values through to last layer
		self.outputNodeInput = num.dot(self.hiddenLayerOutput, self.synapseLayer2)
		
		#Run activation function on output
		self.finalOutput = self.sigmoid(self.outputNodeInput)
		return self.finalOutput.item(0, 0)

	"Returns the result of the sigmoid function with p=1"
	"x is the exponent to raise e to"
	def sigmoid(self, x):
		return (1/(1+num.exp(-x)))

	"Compute the modifer matricies for back prop. These are the values to add to weights"
	"Note: this function runs propagate on the input point"
	"input is the point to train on"
	"expectedOutput is the actual class of the input point"
	def modifierCompute(self, input, expectedOutput):
		actualOutput = self.propagate(input)
		actualOutputMatrix = num.matrix(actualOutput)

		modifierMatrix2 = num.zeros(shape=(self.numHiddenNodes, self.numOutputNodes)) 
		modifierMatrix1 = num.zeros(shape=(self.numInputNodes, self.numHiddenNodes))

		#Generate mod matrix for weights going from hidden layer to output
		for x in range (0, self.numHiddenNodes):

			#Change in total output with respect to the output of the output of the output node
			dTotalToOut2 = -(expectedOutput-actualOutput)

			#Change in the output node with respect to the net input
			dOutToNet2 = actualOutput*(1-actualOutput)
			
			#Change in net input into the output node with respect to the weights into the output node
			dNetToSynapse2 = self.hiddenLayerOutput.item(0, x)

			#Change in total output with respect to the weights into the output node
			dTotalToSynapse2 = dTotalToOut2*dOutToNet2*dNetToSynapse2
			
			#The modifer matrix for weights into the output node
			modifierMatrix2[x, 0] = dTotalToSynapse2

		#Generate modifier matrix for weights going from the input nodes to the hidden layer
		for y in range (0, self.numInputNodes):
			for x in range (0, self.numHiddenNodes):

					#Change in total output with respect to the net input into the output node
					dTotalToNet2 = dTotalToOut2 * dOutToNet2
					
					#Change in total with respect to the output of the hidden layer
					dTotalToOut1 = dTotalToNet2 * self.synapseLayer1.item(y, x)
					
					#Change in output of hidden layer with respect to input into hidden layer
					dOutToNet1 = self.hiddenLayerOutput.item(0, x) * (1 - self.hiddenLayerOutput.item(0, x))

					#Change in input into the hidden layer with respect to the weights applied into the input of the hidden layer
					dNetToSynapse1 = self.inputMatrixArray.item(0, y)

					#Change in output with respect to the weights of the input layer
					dTotalToSynapse1 = dTotalToOut1 * dOutToNet1 * dNetToSynapse1
					modifierMatrix1[y, x] = dTotalToSynapse1

		return modifierMatrix2, modifierMatrix1

	"Update the weights of the neural network with the modifier matrix"
	"modifierMatrix1 is the modifier matrix to use for the weights into the hidden layer"
	"modifierMatrix2 is the modifier matrix to use for the weights into the output layer"
	def update_weights(self, modifierMatrix1, modifierMatrix2):
		self.synapseLayer1 = num.subtract(self.synapseLayer1, modifierMatrix1)
		self.synapseLayer2 = num.subtract(self.synapseLayer2, modifierMatrix2)
		return

	"Classify a set of points with the neural network and output the error rate"
	"testSet is the set of points to classify"
	def classify_set(self, testSet):
		incorrectClassifications = 0
	
		for point in testSet:
			actual = self.propagate(num.matrix([point['x_val'], point['y_val']]))
			actual = num.round(actual)
			if(actual != point['class']):
				incorrectClassifications = incorrectClassifications + 1
		return incorrectClassifications/len(testSet)

	"Train the neural network using a set of points"
	"training set is the set of points to use to train the neural network"
	def train_set(self, trainingSet):
		for point in trainingSet:
			modMatrix2, modMatrix1 = self.modifierCompute(num.matrix([point['x_val'], point['y_val']]), point['class'])
			self.update_weights(modMatrix1, modMatrix2)
		return


"The main program loop"
def ann_main():
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
		hidden_nodes = (int(get_argument(args.optional_arguments, 'h')))
	else:
		hidden_nodes = 5
	if (float(get_argument(args.optional_arguments, 'p'))):
		holdout_percent = (float(get_argument(args.optional_arguments, 'p')))
	else:
		holdout_percent = 0.20

	"The neural net we will use for the rest of the script"
	neuralNet = ANN(input_nodes, hidden_nodes)
	
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
	print(neuralNet.classify_set(testSet))
	neuralNet.train_set(trainingSet)

	"Determine the error rate of the testSet"
	errorRate = neuralNet.classify_set(testSet)

	print(errorRate)

if __name__ == '__main__':
	ann_main()