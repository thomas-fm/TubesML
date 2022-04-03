import numpy as np
from random import seed
from random import random, uniform
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import graphviz
import copy
from random import randint
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Data:
    def __init__(self):
        self.data = []
        self.target = []
        self.target_names = []

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
    return x * (1 - x)

# todo
def relu_derivative(x):
    return [0 if (el < 0) else 1 for el in x]

# todo
def softmax_derivative(x):
    return [-(1-el) for el in x]

def split_dataset_90_10(dataset):
    data_length = len(dataset.target)
    n_train = round(data_length * 9/10)
    n_test = data_length - n_train
    
#     train = { "data": [], "target":[] }
#     test = { "data": [], "target":[] }
    train = Data()
    test = Data()
    train.target_names = dataset.target_names
    test.target_names = dataset.target_names

    test_idx = []
    while len(test_idx) < n_test:
        idx = randint(0, 149)
        try:
            test_idx.index(idx)
        except:
            test_idx.append(idx)
            test.data.append(dataset.data[idx])
            test.target.append(dataset.target[idx])
            
    for i in range(data_length):
        try:
            test_idx.index(i)
        except:
            train.data.append(dataset.data[i])
            train.target.append(dataset.target[i])
    return train, test

def confusionMatrix(y_test, y_pred):
    x = len(set(y_test))
    confusion_matrix = [[0 for i in range(x)] for j in range(x)]
    for i in range(len(y_test)):
        confusion_matrix[y_test[i]][y_pred[i]] += 1
    return np.array(confusion_matrix)

def accuracy(confusion_matrix):
    np.seterr(invalid='ignore')
    return np.nan_to_num(np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))

def precision(confusion_matrix):
    np.seterr(invalid='ignore')
    return np.nan_to_num(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0))

def recall(confusion_matrix):
    np.seterr(invalid='ignore')
    return np.nan_to_num(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))

def f1(confusion_matrix):
    np.seterr(invalid='ignore')
    return np.nan_to_num(2 * precision(confusion_matrix) * recall(confusion_matrix) / (precision(confusion_matrix) + recall(confusion_matrix)))

def summary(confusion_matrix):
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Accuracy:", accuracy(confusion_matrix))
    print("Precision:", precision(confusion_matrix))
    print("Recall:", recall(confusion_matrix))
    print("F1:", f1(confusion_matrix))

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
    def __init__(self, file_name, learning_rate=0.1, err_threshold=0.01, max_iter=100, batch_size=2, dataset=load_iris(), n_input=4, n_output=3):
        # Load iris dataset
        self.dataset = dataset  # dataset
        self.input = dataset.data  # input
        self.target = dataset.target  # target
        self.target_names = dataset.target_names  # target class name
        self.n_attr = n_input  # n input attribute

        # Neural network
        self.learning_rate = learning_rate
        self.err_threshold = err_threshold
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.layers = []
        self.bias = 1
        self.output = []  # final output from forward propagate
        self.mse = 999

        # Back prop
        self.error_hidden_value = 0
        self.updated_weights = []
        self.error = 999  # current error?
        self.weights = []  # last updated weight
        self.predict = []

        with open(file_name, "r") as f:
            line = f.readline().split()
            self.n_layers = len(line) - 1 # how many hidden layers + ouput
            
            for i in range(self.n_layers + 1):
                if i == 0:
                    self.layers.append(Layer(self.n_attr, int(line[i])))
                else:
                    self.layers.append(Layer(int(line[i-1]), int(line[i])))

            for i in range(self.n_layers + 1):
                f.readline()
                for j in range(self.layers[i].n_input + 2):
                    weight = []
                    line = f.readline().strip(" \n").split(" ")
                    for k in range(len(line)):
                        if (j == 0):
                            self.layers[i].activations = str(line[k])
                        else:
                            weight.append(float(line[k]))
                    if j!=0:
                        self.layers[i].weights.append(weight)

    # todo
    def save_model(self, filename):
        new_file = []
        with open(filename) as file:
            lines = [line.rstrip().split() for line in file]

            new_file.append(lines[0])
            new_file.append(lines[1])
            
            for i in range(self.n_layers + 1):
                new_file.append([self.layers[i].activations])
                for new_weight in self.layers[i].updated_weights:
                    new_file.append(new_weight)
                if i < self.n_layers - 1:
                    new_file.append('')
                    
            for line in range(len(new_file)):
                str_line = ''
                for i in range(len(new_file[line])):
                    str_line += str(new_file[line][i])
                    if i < len(new_file[line]) - 1:
                        str_line += ' '
                new_file[line] = str_line
                
        new_filename = filename.split(".")[0] + "_updated_weights"
        with open('model/' + new_filename, 'w') as f:
            for line in range(len(new_file)):
                f.write(new_file[line])
                if line < len(new_file) - 1:
                    f.write('\n')

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
        mse_sum = 0
        total = 0
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
                mse_sum += pow(expected_target[j] - output_layer.error[i][j], 2)
                total += 1
        self.mse = (1/total) * mse_sum
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
    
        # for i in range(len(self.layers)-1, 0, -1):
        #     next_layer = self.layers[i+1]
        #     weights = next_layer.weights
        #     nextWeights =
        
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
        input_temp = copy.copy(self.input)

        while ((it < self.max_iter) and (self.err_threshold < self.mse)):
            output = []
            error = []
            for i in range(int(len(self.input)/self.batch_size)):
                idx = i * self.batch_size
                self.input = input_temp[idx:idx+self.batch_size]
                self.forward_propagation(type="train")
                self.back_propagation()
                output.append(self.layers[self.n_layers].output)
                error.append(self.layers[self.n_layers].error)
            it += 1
            self.error = self.error
            
        # print(self.mse)
        # print(it)

        return

    def set_predict(self, input):
        self.predict = input

    def prediction(self, test_target):
        self.forward_propagation(type="predict")
        self.convert_output_to_class(test_target)

        return self.output
        # return self.output
        
    def prediction2(self):
        self.forward_propagation(type="predict")
        self.convert_output_to_class_2()
        return self.output
        
    def check_sanity(self):
        for layer in self.layers:
            print(layer.weights)

    def convert_output_to_class(self, test_target):
        self.output_predict = test_target.copy()
        for i in range(len(self.output)):
            if (self.output[i][0] > self.output[i][1]):
                self.output_predict[i] = 0 if (self.output[i][0] > self.output[i][2]) else 2
            elif (self.output[i][0] < self.output[i][1]):
                self.output_predict[i] = 1 if (self.output[i][1] > self.output[i][2]) else 2
    
    def convert_output_to_class_2(self):
        self.output_predict = self.target.copy()
        for i in range(len(self.output)):
            if (self.output[i][0] > self.output[i][1]):
                self.output_predict[i] = 0 if (self.output[i][0] > self.output[i][2]) else 2
            elif (self.output[i][0] < self.output[i][1]):
                self.output_predict[i] = 1 if (self.output[i][1] > self.output[i][2]) else 2

    def cross_validate(self):
        # shuffle dataset
        label = np.array(self.dataset.data)
        target = np.array(self.dataset.target)

        indices = np.arange(label.shape[0])
        np.random.shuffle(indices)

        label = label[indices]
        target = target[indices]

        # split into 10
        n = len(self.dataset.target)
        j = int(np.ceil(n / 10))

        total_mse = 0
        acc_score = 0
        prec_score = 0
        f1_score = 0
        rec_score = 0

        for it in range(10):
            data_train_label = copy.copy(label)
            data_train_target = copy.copy(target)

            data_train_label = np.concatenate((data_train_label[0:it*j], data_train_label[it*j:it*j+j]))
            data_train_target = np.concatenate((data_train_target[0:it*j], data_train_target[it*j:it*j+j]))

            self.input = data_train_label
            self.target = data_train_target

            data_test_label = label[it*j:it*j+j]
            data_test_target = target[it*j:it*j+j]
            self.predict = data_test_label

            # train and predict
            self.train()

            # calculate error
            pred = self.prediction()
            self.convert_output_to_class_2()
            expec = []

            # transform to [x,x,x]
            for i in range(len(self.output)):
                expected_target = []
                # print(data_test_label[i])
                if (data_test_target[i] == 0):
                    expected_target = [1, 0, 0]
                if (data_test_target[i] == 1):
                    expected_target = [0, 1, 0]
                if (data_test_target[i] == 2):
                    expected_target = [0, 0, 1]
                expec.append(expected_target)

            # calculate the MSE
            pred = np.concatenate(pred).ravel()
            expec = np.concatenate(expec).ravel()

            # calculate confusion matrix
            # print(self.output_predict)
            print(self.output_predict)
            confusion_matrix = confusionMatrix(data_test_target, self.output_predict)
            acc_score += accuracy(confusion_matrix)
            f1_score += f1(confusion_matrix)
            rec_score += recall(confusion_matrix)
            prec_score += precision(confusion_matrix)

            sum_mse_cv = 0

            for i in range(len(pred)):
                sum_mse_cv += pow(pred[i] - expec[i], 2)

            sum_mse_cv = float(sum_mse_cv/len(pred))
            total_mse += sum_mse_cv

        mse_cv = float(total_mse / 10)
        print(f"MSE Score: {1 - mse_cv}")
        print(f"Average accurarcy: {acc_score/10}")
        # print(f"Precision: {prec_score/10}")
        # print(f"F1: {f1_score/10}")
        # print(f"Recall: {rec_score/10}")



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

    def testForward(self):
        self.prediction_forward()
    
seed(1)
dataset = load_iris()
#
# nn = NeuralNetwork(file_name="model1.txt",dataset=dataset, batch_size=2)
# Split test
# dataset = load_iris()
train, test = split_dataset_90_10(dataset)
# nn.train()

nn = NeuralNetwork(file_name="model1.txt",dataset=dataset, batch_size=2)
nn.train()
# # # nn.set_input([[5.1, 3.5, 1.4, 0.2]])
# # # print(test.data)
nn.set_predict(test.data)
# # nn.set_predict(dataset.data)
nn.prediction(test.target)
# # print(nn.output_predict)

print("--- Split test ---")
summary(confusionMatrix(test.target, nn.output_predict))
# # nn.cross_validate()
# # nn.draw_model()

# ##NOMOR 2

# #UJI DENGAN MATRIX CONFUSION
# dataset = load_iris()
# nn = NeuralNetwork(n_layers=2, dataset=dataset, batch_size=2, n_neuron=[3, 2], activation=["sigmoid", "sigmoid"])
# nn.train()
# nn.set_predict(dataset.data)
# # nn.prediction2()
# nn.cross_validate()

# print("============ UJI DENGAN MATRIX CONFUSSION =================")
# summary(confusionMatrix(dataset.target, nn.output_predict))

# #UJI DENGAN SKLEARN
# print("============ UJI DENGAN SKLEARN =================")
# # Normalize
# scaler = StandardScaler()
# scaler.fit(dataset.data)

# train_data = scaler.transform(dataset.data)

# clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(3, 2), max_iter=1000, batch_size=2)
# clf.fit(train_data, dataset.target)  
# print(confusion_matrix(dataset.target, clf.predict(train_data)))
# print(accuracy_score(dataset.target, clf.predict(train_data), normalize=False)/float(150))
# print(precision_score(dataset.target, clf.predict(train_data), average=None))
# print(recall_score(dataset.target, clf.predict(train_data), average=None))
# print(f1_score(dataset.target, clf.predict(train_data), average=None))