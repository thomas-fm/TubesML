from copyreg import constructor
from subprocess import CREATE_NEW_CONSOLE
import numpy as np
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
        self.error = []
        self.updated_weights = []

    def update_weights_backpropagate(self):
        self.weights = self.updated_weights.copy()
        self.updated_weights.clear()


class NeuralNetwork:
    def __init__(self, n_layers, n_neuron=[], activation=[],
                 learning_rate=0.1, err_threshold=0.01,
                 max_iter=100, batch_size=1, dataset=load_iris()):
        # Load iris dataset
        self.dataset = dataset  # dataset
        self.input = dataset.data  # input
        self.target = dataset.target  # target
        self.target_names = dataset.target_names  # target class name
        self.n_attr = len(self.input[0])  # n input attribute

        # Neural network
        self.n_layers = n_layers  # how many hidden layers
        self.n_neuron = n_neuron  # how many neuron for each hidden layer
        self.activation = activation  # activation for each layer
        self.learning_rate = learning_rate
        self.err_threshold = err_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.layers = []
        self.bias = 1
        self.output = []  # final output from forward propagate

        # Back prop
        self.error = []
        self.updated_weights = []
        self.error = 0  # current error?
        self.weights = []  # last updated weight

        for i in range(n_layers):
            # n input = n neuron in layer
            if i == 0:
                layer = Layer(4, n_neuron[i])
            else:
                layer = Layer(n_neuron[i-1], n_neuron[i])
            # initalize weight
            layer.weights = [1 for i in range(n_neuron[i-1])]
            layer.activations = activation[i]

            self.layers.append(layer)

    # todo
    def forward_propagation(self, activation):
        self.layers[0].input = self.input
        for i in range(self.n_layers):
            if (i != 0):
                for j in range(self.n_input):
                    bias_input = [1]
                    bias_input.extend(self.layers[i-1].output[j])
                    self.layers[i].input.append(bias_input)
            # print(self.layers[i].input)
            # print(self.layers[i].weights)
            self.layers[i].output = np.dot(
                self.layers[i].input, self.layers[i].weights)
            # print(self.layers[i].output)
            for j in range(len(self.layers[i].output)):
                for k in range(len(self.layers[i].output[j])):
                    self.layers[i].output[j][k] = format(self.sigmoid(  # !!hardcoding activation function should be replaced
                        self.layers[i].output[j][k]), ".2f")
            # print(self.layers[i].output)
        self.output = self.layers[self.n_layers-1].output.copy()

    # todo
    def error_output(self):
        # get output layer
        output_layer = self.layers[self.n_layers-1]
        activation_rule = output_layer.activations
        output_layer.error = output_layer.output - self.target

        # calculate error per output node
        for i in range(output_layer.n_nodes):
            if (activation_rule == "sigmoid"):
                output_layer.error[i] *= sigmoid_derivative(
                    output_layer.output[i])
            elif (activation_rule == "relu"):
                output_layer.error[i] *= relu_derivative(
                    output_layer.output[i])
            elif (activation_rule == "linear"):
                output_layer.error[i] *= linear_derivative(
                    output_layer.output[i])
            elif (activation_rule == "softmax"):
                output_layer.error[i] *= softmax_derivative(
                    output_layer.output[i])

        # calculate updated weights
        output_layer.updated_weights = np.dot(
            self.learning_rate, output_layer.error)
        output_layer.updated_weights = np.dot(
            output_layer.updated_weights, output_layer.weights)
        print(output_layer.weights)
        print(output_layer.updated_weights)
        return
        return

    # todo
    def error_hidden(self):
        return

    # todo
    def update_weights(self):
        for layer in self.layers:
            layer.update_weights()
        return

    # todo
    def back_propagation(self):
        # output layer
        self.error_output()

        # for all hidden layers
        return

    # todo
    def train(self):
        it = 0
        while (it < self.max_iter) and (self.error < self.err_threshold):
            self.forward_propagation()
            self.back_propagation()
            it += 1
        return

    def predict(self, input):
        predict_class = np.dot(np.transpose(input), self.updated_weights)
        return


nn = NeuralNetwork(n_layers=0)
