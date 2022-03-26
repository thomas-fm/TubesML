import numpy as np
from random import seed
from random import random, uniform
from sklearn.datasets import load_iris

def linear(x):
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def softmax(x):
    net_h = np.array(x)
    numerator = np.exp(net_h)
    denominator = np.sum(np.exp(x))
    softmax_output = numerator / denominator
    return softmax_output

def linear_derivative(x):
    return 1

def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

# todo
def relu_derivative(x):
    x = np.array(x)
    return [0 if (el < 0) else 1 for el in x]

# todo
def softmax_derivative(x):
    return 1

class Layer:
    def __init__(self, n_input, n_nodes):
        self.weights = []
        self.n_input = n_input
        self.n_nodes = n_nodes
        self.activations = ""
        self.input = []
        self.output = []

class NeuralNetwork:
    def __init__(self, n_layers, n_neuron=[], activation=[], 
    learning_rate=0.1, err_threshold=0.01, 
    max_iter=100, batch_size=1, dataset=load_iris(),
    n_input=4, n_output=3):
        # Load iris dataset
        self.dataset = dataset # dataset
        self.input = dataset.data # input
        self.target = dataset.target # target
        self.target_names = dataset.target_names # target class name
        self.n_attr = n_input # n input attribute

        # Neural network
        self.n_layers = n_layers # how many hidden layers
        self.n_neuron = n_neuron # how many neuron for each hidden layer
        self.activation = activation # activation for each layer
        self.learning_rate = learning_rate 
        self.err_threshold = err_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.layers = []
        self.bias = 1
        self.output = [] # final output from forward propagate

        # Back prop
        self.error = []
        self.updated_weights = []
        self.error = 0 # current error?
        self.weights = [] #last updated weight

        for i in range(n_layers):
            # n input = n neuron in layer
            if i == 0:
                layer = Layer(self.n_attr+1, n_neuron[i])
                layer.weights = [[uniform(-0.5,0.5) for i in range(n_neuron[i])] for j in range(self.n_attr+1)]
            else:
                layer = Layer(n_neuron[i-1]+1, n_neuron[i])
                layer.weights = [[uniform(-0.5,0.5) for i in range(n_neuron[i])] for j in range(n_neuron[i-1]+1)]
            # initalize weight
            layer.activations = activation[i]

            self.layers.append(layer)
            # print(layer.weights)

    # todo
    def forward_propagation(self):
        # The first input layer
        self.layers[0].input = self.input

        # All hidden layers
        for i in range(self.n_layers):
            # add bias
            bias_input = self.bias
            if (i == 0): # if first hidden layer, convert to array from ndarray and then add bias in the last index
                temp_input = []
                for j in range(len(self.layers[i].input)):
                    input_row = []
                    for k in range(len(self.layers[i].input[j])):
                        input_row.append(self.layers[i].input[j][k])
                    input_row.append(bias_input)
                    temp_input.append(input_row)
                self.layers[i].input = temp_input
            else: # if not first layer the immediately add the bias in the last index
                for j in range(len(self.layers[i].input)):
                    self.layers[i].input[j].append(bias_input)

            # calculate sigma
            self.layers[i].output  = np.dot(self.layers[i].input, self.layers[i].weights)

            # print(f"------- Layer: {i+1} -------")
            # print(f"Activation : {self.layers[i].activations}")
            # print(f"Input : {self.layers[i].input}")
            # print(f"Weight : {self.layers[i].weights}")
            
            # activation function
            for j in range(len(self.layers[i].output)):
                input_next_layer = [] # temporary list to store the next layer's input
                for k in range(len(self.layers[i].output[j])):
                    x = self.layers[i].output[j][k]
                    result = 0
                    if (self.layers[i].activations.lower() == "linier"):
                        result = format(linear(x), ".2f")
                    elif (self.layers[i].activations.lower() == "sigmoid"):
                        result = format(sigmoid(x), ".2f")
                    elif (self.layers[i].activations.lower() == "relu"):
                        result = format(relu(x), ".2f")
                    elif (self.layers[i].activations.lower() == "softmax"):
                        result = format(softmax(x), ".2f")
                    else: # if activation is not linier, relu, sigmoid, or softmax
                        print(f"{self.layers[i].activations}: Invalid activation method!")
                        return
                    
                    self.layers[i].output[j][k] = result # append output, actually layers[i].output == layers[i+1].input
                    input_next_layer.append(float(result)) # append input for next layer in temporary list (input_next_layer)

                if (i+1 < self.n_layers): # if there is still the next layer
                    self.layers[i+1].input.append(input_next_layer) # append input for next layer in layers[i+1].input

                # print(f"Output : {self.layers[i].output}")

        # output in the last layer
        self.output = self.layers[self.n_layers-1].output.copy()

    # todo
    def error_output(self):
        return

    # todo 
    def error_hidden(self):
        return
    
    # todo
    def update_weights(self):
        return
    
    # todo
    def back_propagation(self):
        return

    # todo
    def train(self):
        it = 0
        while (it < self.max_iter) and (self.error < self.err_threshold):
            self.forward_propagation()
            self.back_propagation()
            it+=1
        return

    def predict(self, input):
        predict_class = np.dot(np.transpose(input), self.updated_weights)
        return

seed(1)
nn = NeuralNetwork(n_layers=2, n_neuron=[3,4], activation=["sigMOID", "RELU"])
nn.forward_propagation()