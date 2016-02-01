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
#TODO: Make the split random, not based on sequence
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
class ANN():
	def __init__(self, numInputNodes, numHiddenNodes, holdOutPercentage):
		#Define parameters
		self.numInputNodes = numInputNodes
		self.numHiddenNodes = numHiddenNodes
		self.numOutputNodes = 1   #This will never be different
		self.holdOut = holdOutPercentage

		#Create synapse weights
		#The first layer will be a matrix of weights that has rows = numInputNodes, cols = numHiddenNodes
		#This is the matrix of weights from input nodes to hidden nodes
		self.synapseLayer1 = num.random.uniform(-1, 1, [self.numInputNodes, self.numHiddenNodes])
		#The second layer will be a matrix of weights that has rows = numHiddenNodes, cols = numOutputNodes
		#This is the matrix of weights from hidden nodes to output nodes
		self.synapseLayer2 = num.random.uniform(-1, 1, [self.numHiddenNodes, self.numOutputNodes])
	#Propagate input through to output
	#Using an input matrix rather than pair value allows us to process clusters of input if we wish
	#This can also be used for a 1x(number of values per point) matrix, handling a single point at a time
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

	#Perform sigmoid activation function
	def sigmoid(self, x):
		return (1/(1+num.exp(-x)))

	def derivativeSigmoid(self, x):
		return num.exp(-x)/((1+num.exp(-x))**2)

	def modifierCompute(self, input, expectedOutput):
		actualOutput = self.propagate(input)
		actualOutputMatrix = num.matrix(actualOutput)

		modifierMatrix2 = num.zeros(shape=(self.numHiddenNodes, self.numOutputNodes)) 
		modifierMatrix1 = num.zeros(shape=(self.numInputNodes, self.numHiddenNodes))

		#Generate mod matrix for weights going from hidden layer to output
		for x in range (0, self.numHiddenNodes):

			#Calculate the total error at the output node
			totalErrorMatrix = num.subtract(expectedOutput, actualOutputMatrix)
			totalErrorMatrix = num.square(totalErrorMatrix)
			totalErrorMatrix = num.divide(totalErrorMatrix, 2)
			totalError = totalErrorMatrix.sum()

			dTotalToOut2 = -(expectedOutput-actualOutput)

			dOutToNet2 = actualOutput*(1-actualOutput)
			
			dNetToSynapse2 = self.hiddenLayerOutput.item(0, x)

			
			dTotalToSynapse2 = dTotalToOut2*dOutToNet2*dNetToSynapse2
			
			modifierMatrix2[x, 0] = dTotalToSynapse2

		for y in range (0, self.numInputNodes):
			for x in range (0, self.numHiddenNodes):

					dTotalToNet1 = dTotalToOut2 * dOutToNet2
					#Modify by weight
					dTotalToOut1 = dTotalToNet1 * self.synapseLayer1.item(y, x)
					#Find derivative of douth1 with respect to dneth1
					dOutToNet1 = self.hiddenLayerOutput.item(0, x) * (1 - self.hiddenLayerOutput.item(0, x))
					dNetToSynapse1 = self.inputMatrixArray.item(0, y)
					dTotalToSynapse1 = dTotalToOut1 * dOutToNet1 * dNetToSynapse1
					modifierMatrix1[y, x] = dTotalToSynapse1

		return modifierMatrix2, modifierMatrix1

	def update_weights(self, modifierMatrix1, modifierMatrix2):
		self.synapseLayer1 = num.subtract(self.synapseLayer1, modifierMatrix1)
		self.synapseLayer2 = num.subtract(self.synapseLayer2, modifierMatrix2)
		return

	def classify_set(self, testSet):
		incorrectClassifications = 0
	
		for point in testSet:
			actual = self.propagate(num.matrix([point['x_val'], point['y_val']]))
			actual = num.round(actual)
			if(actual != point['class']):
				incorrectClassifications = incorrectClassifications + 1
		return incorrectClassifications/len(testSet)

	def train_set(self, trainingSet):
		for point in trainingSet:
			modMatrix2, modMatrix1 = self.modifierCompute(num.matrix([point['x_val'], point['y_val']]), point['class'])
			self.update_weights(modMatrix1, modMatrix2)
		return


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
		holdout_percent = (float(get_argument(args.optional_arguments, 'p'))) #TODO CHECK THIS (dividing by 100)
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
	neuralNet.train_set(trainingSet)

	errorRate = neuralNet.classify_set(testSet)

	print(errorRate)

if __name__ == '__main__':
	ann_main()