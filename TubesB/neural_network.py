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
        self.updated_weights = []

    def update_weights_back_propagation(self):
        self.weights = self.updated_weights.copy()
        self.updated_weights = []


class NeuralNetwork:
    def __init__(self, n_layers, n_neuron=[], activation=[],
                 learning_rate=0.1, err_threshold=0.01,
                 max_iter=100, batch_size=1, dataset=load_iris(),
                 n_input=4, n_output=3):
        # Load iris dataset
        self.dataset = dataset  # dataset
        self.input = dataset.data  # input
        self.target = dataset.target  # target
        self.target_names = dataset.target_names  # target class name
        self.n_attr = n_input  # n input attribute

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
                layer = Layer(self.n_attr+1, n_neuron[i])
                layer.weights = [
                    [uniform(-0.5, 0.5) for i in range(n_neuron[i])] for j in range(self.n_attr+1)]
            else:
                layer = Layer(n_neuron[i-1]+1, n_neuron[i])
                layer.weights = [
                    [uniform(-0.5, 0.5) for i in range(n_neuron[i])] for j in range(n_neuron[i-1]+1)]
            # initalize weight
            layer.activations = activation[i]

            self.layers.append(layer)

        # add last layer, last hidden to output
        layer = Layer(n_neuron[-1] + 1, n_output)
        layer.weights = [[uniform(-0.5, 0.5) for i in range(n_output)]
                         for j in range(n_neuron[-1] + 1)]
        layer.activations = activation[-1]
        self.layers.append(layer)

    # todo
    def forward_propagation(self):
        # The first input layer
        self.layers[0].input = self.input

        # All hidden layers
        for i in range(self.n_layers + 1):
            # add bias
            bias_input = self.bias
            if (i == 0):  # if first hidden layer, convert to array from ndarray and then add bias in the last index
                temp_input = []
                for j in range(len(self.layers[i].input)):
                    input_row = []
                    for k in range(len(self.layers[i].input[j])):
                        input_row.append(self.layers[i].input[j][k])
                    input_row.append(bias_input)
                    temp_input.append(input_row)
                self.layers[i].input = temp_input
            else:  # if not first layer the immediately add the bias in the last index
                for j in range(len(self.layers[i].input)):
                    self.layers[i].input[j].append(bias_input)

            # calculate sigma
            self.layers[i].output = np.dot(
                self.layers[i].input, self.layers[i].weights)

            # print(f"------- Layer: {i+1} -------")
            # print(f"Activation : {self.layers[i].activations}")
            # print(f"Input : {self.layers[i].input}")
            # print(f"Weight : {self.layers[i].weights}")

            # activation function
            for j in range(len(self.layers[i].output)):
                input_next_layer = []  # temporary list to store the next layer's input
                for k in range(len(self.layers[i].output[j])):
                    x = self.layers[i].output[j][k]
                    result = 0
                    if (self.layers[i].activations.lower() == "linier"):
                        result = format(linear(x))
                    elif (self.layers[i].activations.lower() == "sigmoid"):
                        result = format(sigmoid(x))
                    elif (self.layers[i].activations.lower() == "relu"):
                        result = format(relu(x))
                    elif (self.layers[i].activations.lower() == "softmax"):
                        result = format(softmax(x))
                    else:  # if activation is not linier, relu, sigmoid, or softmax
                        print(
                            f"{self.layers[i].activations}: Invalid activation method!")
                        return

                    # append output, actually layers[i].output == layers[i+1].input
                    self.layers[i].output[j][k] = result
                    # append input for next layer in temporary list (input_next_layer)
                    input_next_layer.append(float(result))

                if (i < self.n_layers):  # if there is still the next layer
                    # append input for next layer in layers[i+1].input
                    self.layers[i+1].input.append(input_next_layer)

            # print(f"Output : {self.layers[i].output}")

        # output in the last layer
        self.output = self.layers[-1].output.copy()

    # todo
    def error_output(self):
        # get output layer
        output_layer = self.layers[self.n_layers-1]
        activation_rule = output_layer.activations
        output_layer.error = output_layer.output.copy()
        for i in range(len(self.input)):
            for j in range(len(output_layer.output[i])):
                output_layer.error[i][j] -= self.target[i]

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
        return

    # todo
    def error_hidden(self):
        return

    # todo
    def update_weights(self, row_output, old_weight, error_term):
        new_weight_list = []

        # new weight for hidden layers
        for i in range(len(row_output)):
            new_weight = old_weight[i] + \
                self.learning_rate * error_term * row_output[i]
            new_weight_list.append(new_weight)

        # new weight for bias
        new_weight_list.append(
            old_weight[-1] + self.learning_rate * error_term * self.bias)

        # print(f"new wight: {new_weight_list}")
        return new_weight_list

    # todo
    def back_propagation(self):
        # output layer
        self.error_output()
        output_layer = self.layers[self.n_layers-1]
        self.error = np.mean(output_layer.error)
        updated_weights_temp = []
#         print(output_layer.input, "\n pokpok \n", output_layer.weights, "\n pokpok \n", output_layer.error)
        for i in range(len(self.input)):
            updated_weights_temp.append(self.update_weights(
                output_layer.input[i][0:self.n_attr-1], output_layer.weights, output_layer.error[i]))
        output_layer.updated_weights = output_layer.weights.copy()
        output_layer.updated_weights = np.mean(
            updated_weights_temp, axis=0)

#         print(output_layer.weights)
#         print(output_layer.updated_weights)

        # all hidden layer

        # update all weights
        for layer in self.layers:
            layer.update_weights_back_propagation()
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


seed(1)
nn = NeuralNetwork(n_layers=2, n_neuron=[3, 5], activation=["sigmoid", "relu"])
nn.forward_propagation()
nn.back_propagation()
# print(nn.output)

# # update weight between first neuron in output layer and first neuron in the last hidden layer
# test_weight = []
# for row_weight in nn.layers[2].weights:
#     test_weight.append(row_weight[0])
# row_output = nn.layers[1].output[0]
# # print(test_weight)
# # print(row_output)
# nn.update_weights(row_output, test_weight, 0.01)
