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
    def __init__(self, n_input, node_per_layer): #node_per_layer = [5,5,2]
        self.n_layers = len(node_per_layer)
        self.n_input = n_input
        self.layers = []
        self.bias = 1
        self.input = []
        self.output = []
        self.loss = []
        self.error = []
        self.learning_rate = 0.001
        
        #create first layer (layer 0)
        self.layers += [Layer(n_input, node_per_layer[0])]
        
        #create layer hidden and output
        for i in range(1, self.n_layers):
            self.layers += [Layer(node_per_layer[i-1], node_per_layer[i])] 
    
    def showLayer(self, layer):
        for i in range(self.n_layers):
            print("layer", i, ":", self.layers[i])
        return self.layers
    
    def forward_propagation(self, input):
        self.input = input
        self.output = []
        self.output.append(input)
        for i in range(1, self.n_layers):
            self.output.append(np.dot(self.output[i], self.layers[i-1].weights[i]) + self.bias)
        return self.output[-1]
    
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

if __name__ == "__main__":
    n_input = 2
    node_per_layer = [5,5,2]
    # print(FFNN(n_input, node_per_layer).showLayer(node_per_layer))
    print(FFNN(n_input, node_per_layer).softmax(node_per_layer))