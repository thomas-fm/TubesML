import numpy as np
from random import seed
from random import random
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
                layer.weights = [[random() for i in range(self.n_attr+1)] for j in range(n_neuron[i])]
            else:
                layer = Layer(n_neuron[i-1]+1, n_neuron[i])
                layer.weights = [[random() for i in range(n_neuron[i-1]+1)] for j in range(n_neuron[i])]
            # initalize weight
            layer.activations = activation[i]

            self.layers.append(layer)
            print(layer.weights)

    # todo
    def forward_propagation(self):
        return

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
nn = NeuralNetwork(n_layers=3, n_neuron=[2, 3, 4], activation=["sigmoid","sigmoid","sigmoid"])