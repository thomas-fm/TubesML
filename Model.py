from audioop import bias
from tokenize import Double
import numpy as np


class Layer:
    def __init__(self, n_input, n_nodes):
        self.weights = []
        self.n_input = n_input
        self.n_nodes = n_nodes
        self.activations = []
        self.input = []
        self.output = []
        self.deltas = []
        # for i in range (len(n_nodes)) :
        #     self.deltas[i] = 0

    def getNode(self, i):
        return self.n_nodes[i]


class FFNN:
    def __init__(self, n_input, node_per_layer):  # node_per_layer = [5,5,2]
        self.n_layers = len(node_per_layer)
        self.n_input = n_input
        self.layers = []
        self.bias = 1
        self.input = []
        self.output = []
        self.loss = []
        self.error = []
        self.learning_rate = 0.001

        # create first layer (layer 0)
        self.layers += [Layer(n_input, node_per_layer[0])]

        # create layer hidden and output
        for i in range(1, self.n_layers):
            self.layers += [Layer(node_per_layer[i-1], node_per_layer[i])]

    def __init__(self, file_name):  # read model from file
        with open(file_name, "r") as f:
            # define attributes
            self.layers = []
            self.bias = 1
            self.input = []
            self.output = []
            self.loss = []
            self.error = []
            self.learning_rate = 0.001

            # read n_input
            self.n_input = int(f.readline()[0])

            # read node per layer
            line = f.readline().strip(" \n").split(" ")
            self.n_layers = len(line) - 1
            for i in range(1, self.n_layers + 1):
                self.layers.append(Layer(int(line[i-1]), int(line[i])))

            # read input
            f.readline()
            for i in range(self.n_input):
                input = []
                line = f.readline().strip(" \n").split(" ")
                input.append(1)  # bias
                for j in range(len(line)):
                    input.append(float(line[j]))
                self.input.append(input)

            # read weight for every layer
            for i in range(self.n_layers):
                f.readline()
                for j in range(self.layers[i].n_input + 1):
                    weight = []
                    line = f.readline().strip(" \n").split(" ")
                    for k in range(len(line)):
                        weight.append(float(line[k]))
                    self.layers[i].weights.append(weight)

    def showLayer(self, layer):
        for i in range(self.n_layers):
            print("layer", i, ":", self.layers[i])
        return self.layers

    # def forward_propagation(self, input):
    #     self.input = input
    #     self.output = []
    #     self.output.append(input)
    #     for i in range(1, self.n_layers):
    #         self.output.append(
    #             np.dot(self.output[i], self.layers[i-1].weights[i]) + self.bias)
    #     return self.output[-1]

    # !!activation function should be customable
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

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(x, 0)

    def linear(self, x):
        return x

    def softmax(self, x):
        net_h = np.array(x)
        numerator = np.exp(net_h)
        denominator = np.sum(np.exp(x))
        softmax_output = numerator / denominator
        return softmax_output

    def show_output(self):
        print(self.output)


if __name__ == "__main__":
    ffnn = FFNN("model.txt")
    ffnn.forward_propagation(["sigmoid"])
    ffnn.show_output()
