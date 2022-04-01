import numpy as np
from neural_network import NeuralNetwork
from sklearn import datasets
import random

iris = datasets.load_iris()
x = iris.data.tolist()
y = iris.target.tolist()

# print(x, y)

# target = []
# for i in range(len(y)):
#     if y[i] == 0:
#         target.append([1, 0, 0])
#     elif y[i] == 1:
#         target.append([0, 1, 0])
#     elif y[i] == 2:
#         target.append([0, 0, 1])
        
ffnn = NeuralNetwork(2, [3,2], ["linear", "sigmoid"])
ffnn.set_predict([[5.1, 3.5, 1.4, 0.2]])
ffnn.prediction()

#Confusion matrix untuk semua kelas
def confusionMatrix(y_test, y_pred):
    x = len(set(y_test))
    confusion_matrix = [[0 for i in range(x)] for j in range(x)]
    for i in range(len(y_test)):
        confusion_matrix[y_test[i]][y_pred[i]] += 1
    return np.array(confusion_matrix)

def accuracy(confusion_matrix):
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)

def precision(confusion_matrix):
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)

def recall(confusion_matrix):
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

def f1(confusion_matrix):
    return 2 * precision(confusion_matrix) * recall(confusion_matrix) / (precision(confusion_matrix) + recall(confusion_matrix))

def summary(confusion_matrix):
    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Accuracy:", accuracy(confusion_matrix))
    print("Precision:", precision(confusion_matrix))
    print("Recall:", recall(confusion_matrix))
    print("F1:", f1(confusion_matrix))

# print(confusionMatrix([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]))
# print(accuracy(confusionMatrix([2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1], [2, 2, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1])))
# print(precision(confusionMatrix([2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1], [2, 2, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1])))
# print(recall(confusionMatrix([2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1], [2, 2, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1])))
# print(f1(confusionMatrix([2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1], [2, 2, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1])))
summary(confusionMatrix([2, 1, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1], [2, 2, 0, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 2, 1]))