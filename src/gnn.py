import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pprint import pprint


# generating data using the blobs function :^)
CLASSES_COUNT = 4
FEATURES_COUNT = 2

X, y = make_blobs(200, cluster_std=1.3, centers=CLASSES_COUNT, n_features=FEATURES_COUNT)
y = np.array([[0 if j != i else 1 for j in range(np.max(y)+1)] for i in y])


# plotting the data
# plt.figure(figsize=(10, 8))
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()


# linear layer class
class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.w = np.random.normal(size=(in_features, out_features))
        self.b = np.random.normal(size=out_features)
    
    # one forward step (calling an instance of the class)
    def __call__(self, features: np.ndarray) -> np.ndarray:
        return np.dot(features, self.w) + self.b


# neural network class (with static parameters)
class NN:
    def __init__(self):
        self.linear1 = Linear(FEATURES_COUNT, 5)
        self.linear2 = Linear(5, CLASSES_COUNT)

    # forward propagation
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        return self.softmax(self.linear2(self.relu(self.linear1(x))))

    # ReLU activation function
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.max(np.array([x, np.zeros(x.shape)]), axis=0)

    # softmax activation function
    def softmax(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.e**i/np.sum(np.e**x) for i in x])

    # cross-entropy loss, fitness function respectively
    def cross_entropy(self, prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return -np.sum(labels*np.log(prediction) + (1-labels)*np.log(1-prediction))

    # get multiple predictions, 
    def __call__(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        pred = np.array([self.feedforward(i) for i in features])
        return pred, np.mean(np.array([self.cross_entropy(p, l) for p, l in zip(pred, labels)]))


if __name__ == '__main__':
    model = NN()
    pred, loss = model(X, y)
    pprint(loss)