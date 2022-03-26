import numpy as np
from random import seed
from random import random, uniform
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import graphviz

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
    return [0 if (el < 0) else 1 for el in x]

# todo
def softmax_derivative(x):
    return [-(1-el) for el in x]


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

    def update_weights_back_propagation(self):
        self.weights = self.updated_weights.copy()
        self.updated_weights = []


class NeuralNetwork:
    def __init__(self, n_layers, n_neuron=[], activation=[],
                 learning_rate=0.1, err_threshold=0.01,
                 max_iter=100, batch_size=2, dataset=load_iris(),
                 n_input=4, n_output=3):
        # Load iris dataset
        self.dataset = dataset  # dataset
        # self.input = dataset.data  # input
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
        self.error_hidden_value = 0
        self.updated_weights = []
        self.error = 999  # current error?
        self.weights = []  # last updated weight
        self.predict = []

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
    def forward_propagation(self, type):
        for i in range(self.n_layers + 1):
            self.layers[i].input = []
        # The first input layer
        if type == "train":
            self.layers[0].input = self.input
        elif type == "predict":
            self.layers[0].input = self.predict

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
            # print(len(self.layers[i].input))
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
                    if (self.layers[i].activations.lower() == "linear"):
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
        output_layer = self.layers[self.n_layers]
        activation_rule = output_layer.activations.lower()
        output_layer.error = output_layer.output.copy()
        for i in range(len(self.input)):
            expected_target = []
            if (self.target[i] == 0):
                expected_target = [1, 0, 0]
            if (self.target[i] == 1):
                expected_target = [0, 1, 0]
            if (self.target[i] == 2):
                expected_target = [0, 0, 1]
            for j in range(3):
                output_layer.error[i][j] = expected_target[j] - output_layer.error[i][j]
        self.error = np.mean(output_layer.error)

        # calculate error per output node
        for i in range(len(output_layer.error)):
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
        #menghitung delta net j = delta k * weight j k
        delta_net_j = 0
        
        
        error_hidden = []
        for i in range(self.n_layers-1, -1, -1):
            delta_net_j_array = []
            # print( self.layers[i].error, "\n", self.layers[i+1].weights[i])
            for j in range(len(self.layers[i+1].error)):
                delta_net_j_data = []
                for k in range(len(self.layers[i+1].weights)-1):
                    # print(self.layers[i+1].error)
                    delta_net_j = np.dot(self.layers[i+1].error[j], np.transpose(self.layers[i+1].weights[k]))
                    delta_net_j_data.append(delta_net_j)
                    

                delta_net_j_array.append(delta_net_j_data)
                
            # print(delta_net_j_array)

            # print(delta_net_j_array)
            # print(len(self.layers[3].error))
            # print(self.layers[3].weights)
            # breaks
            self.layers[i].error = self.layers[i].output.copy()

            #menghitung error di hidden layer
            for j in range(len(self.layers[i].error)):
                if (self.layers[i].activations == "sigmoid"):
                    self.layers[i].error[j] = np.dot(sigmoid_derivative(
                        self.layers[i].output[j]),  delta_net_j_array[j])
                elif (self.layers[i].activations == "relu"):
                    self.layers[i].error[j] = np.dot(relu_derivative(
                        self.layers[i].output[j]), delta_net_j_array[j])
                elif (self.layers[i].activations == "linear"):
                    self.layers[i].error[j] = np.dot(linear_derivative(
                        self.layers[i].output[j]), delta_net_j_array[j])
                
            # error_hidden = delta_net_j * self.layers[i].output * (1 - self.layers[i].output)
            # print(self.layers[i].error)
        
        return

    # todo
    def update_weights(self, row_input, old_weight, error_term):
        new_weight_list = []

        # new weight for hidden layers
        for i in range(len(row_input)):
            new_weight = old_weight[i] - self.learning_rate * error_term * row_input[i]
            new_weight_list.append(new_weight)
        
        # print(new_weight_list)

        # new weight for bias
        # new_weight_list.append(
        #     old_weight[-1] + self.learning_rate * error_term * self.bias)

        # print(f"new wight: {new_weight_list}")
        return new_weight_list

    # todo
    def back_propagation(self):
        # output layer
        self.error_output()
        output_layer = self.layers[self.n_layers]
        self.error = np.mean(output_layer.error)
        updated_weights_temp = []
        
        for i in range(len(output_layer.output)):
            updated_weights_temp.append(self.update_weights(
                output_layer.input[i], output_layer.weights, output_layer.error[i]))
        output_layer.updated_weights = output_layer.weights.copy()
        output_layer.updated_weights = np.mean(
            updated_weights_temp, axis=0)

        self.error_hidden()
        for i in range(self.n_layers-1, -1 ,-1):
            self.error_hidden_value = np.mean(self.layers[i].error)
            updated_weights_temp_hidden = []
            for j in range(len(self.layers[i].input)):
                updated_weights_temp_hidden.append(self.update_weights(self.layers[i].input[j], self.layers[i].weights, self.layers[i].error[j]))
            self.layers[i].updated_weights = self.layers[i].weights.copy()
            self.layers[i].updated_weights = np.mean(updated_weights_temp_hidden, axis=0)

        # update all weights
        for layer in self.layers:
            layer.update_weights_back_propagation()
        return

    # todo
    def train(self):
        it = 0
        while ((it < self.max_iter) and (self.err_threshold < self.error)):
            output = []
            error = []
            for i in range(int(len(self.dataset.data)/self.batch_size)):
                idx = i * self.batch_size
                self.input = self.dataset.data[idx:idx+self.batch_size]
                self.forward_propagation(type="train")
                self.back_propagation()
                output.append(self.layers[self.n_layers].output)
                error.append(self.layers[self.n_layers].error)
            it += 1
            
        print(self.error)
        print(it)

        return

    def set_predict(self, input):
        self.predict = input

    def prediction(self):
        self.forward_propagation(type="predict")
        print(self.output)
        # return self.output
    def check_sanity(self):
        for layer in self.layers:
            print(layer.weights)
    
    def draw_model(self):
        f = graphviz.Digraph('Feed Forward Neural Network', filename="model")
        f.attr('node', shape='circle', width='1.0')
        f.edge_attr.update(arrowhead='vee', arrowsize='2')
        
        for i in range(self.n_layers):
            if i == 0:
                for j in range(len(self.layers[i].weights)): #count weights
                    for k in range(len(self.layers[i].weights[j])): #output node
                        if j==0:
                            f.edge(f"bx{j}", f"h{i+1}_{k}", 
                                   f"{self.layers[i].weights[j][k]:.2f}")
                        else:
                            f.edge(f"x{j}", f"h{i+1}_{k}", 
                                   f"{self.layers[i].weights[j][k]:.2f}")
            else:
                for j in range(len(self.layers[i].weights)): #count weights
                    for k in range(len(self.layers[i].weights[j])): #output node
                        if j==0:
                            f.edge(f"bh{i}", f"h{i+1}_{k}", 
                                   f"{self.layers[i].weights[j][k]:.2f}")
                        else:
                            f.edge(f"h{i}_{j-1}", f"h{i+1}_{k}", 
                                   f"{self.layers[i].weights[j][k]:.2f}")
        
        print(f.source)
        f.render(directory='model').replace('\\', '/')

seed(1)

# Normalize data
dataset = load_iris()

nn = NeuralNetwork(n_layers=3, dataset=dataset, batch_size=50, n_neuron=[3, 2, 5], activation=["sigmoid", "sigmoid", "sigmoid"])
# nn.forward_propagation()
nn.train()
nn.set_predict([[1.0, 6.5, 2.0, 3.5]])
nn.prediction()
nn.draw_model()
