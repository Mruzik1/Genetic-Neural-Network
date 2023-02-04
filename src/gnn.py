import numpy as np
import matplotlib.pyplot as plt
from random import random, choices
from sklearn.datasets import make_blobs


# generating data using the blobs function :^)
CLASSES_COUNT = 4
FEATURES_COUNT = 2

X, old_y = make_blobs(200, cluster_std=1.3, centers=CLASSES_COUNT, n_features=FEATURES_COUNT)
y = np.array([[0 if j != i else 1 for j in range(np.max(old_y)+1)] for i in old_y])


# linear layer class
class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.w = np.random.normal(size=(in_features, out_features))
        self.b = np.random.normal(size=out_features)
    
    # one forward step (calling an instance of the class)
    def __call__(self, features: np.ndarray) -> np.ndarray:
        return np.dot(features, self.w) + self.b

    # getting views of weights (flatten)
    def get_weights(self) -> np.ndarray:
        return self.w.view().reshape(self.w.size)


# neural network class (with static parameters)
class NN:
    def __init__(self):
        self.linear1 = Linear(FEATURES_COUNT, 8)
        self.linear2 = Linear(8, 8)
        self.linear3 = Linear(8, CLASSES_COUNT)
        
        self.weights = [self.linear1.get_weights(), self.linear2.get_weights(), self.linear3.get_weights()]
        self.biases = [self.linear1.b, self.linear2.b, self.linear3.b]

    # forward propagation
    def feedforward(self, x: np.ndarray) -> np.ndarray:
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.softmax(self.linear3(x))

    # ReLU activation function
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.max(np.array([x, np.zeros(x.shape)]), axis=0)

    # softmax activation function
    def softmax(self, x: np.ndarray) -> np.ndarray:
        return np.array([np.e**i/np.sum(np.e**x) for i in x])

    # cross-entropy loss, fitness function respectively
    def cross_entropy(self, prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return -np.sum(labels*np.log(prediction+1e-20) + (1-labels)*np.log(1-prediction+1e-20))

    # get multiple predictions, 
    def __call__(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        pred = np.array([self.feedforward(i) for i in features])
        return pred, np.mean(np.array([self.cross_entropy(p, l) for p, l in zip(pred, labels)]))


# GA class
class GA:
    def __init__(self, pop_size: int, features: np.ndarray, labels: np.ndarray,
                 selector: int, mutation_chance: float):
        self.population = [NN() for _ in range(pop_size)]
        self.pop_size = pop_size
        self.features = features
        self.labels = labels
        self.selector = selector
        self.mutation_chance = mutation_chance
    
    # performing uniform crossover on weights and biases
    def uniform_crossover(self, p1: NN, p2: NN, prob: float) -> NN:
        child = NN()

        for i, w in enumerate(child.weights):
            for j, _ in enumerate(w):
                child.weights[i][j] = p1.weights[i][j] if random() > prob\
                                      else p2.weights[i][j]
            for j, _ in enumerate(child.biases[i]):
                child.biases[i][j] = p1.biases[i][j] if random() > prob\
                                      else p2.biases[i][j]
        
        if self.mutation_chance > random():
            self.mutation(child)
        
        return child

    # performing mutation (randomly change some weights)
    def mutation(self, ind: NN):
        w_idx = np.random.randint(len(ind.weights))
        b_idx = np.random.randint(len(ind.biases))

        for i, _ in enumerate(ind.weights[w_idx]):
            if random() > 0.6:
                ind.weights[w_idx][i] = np.random.normal()

        for i, _ in enumerate(ind.biases[b_idx]):
            if random() > 0.6:
                ind.biases[b_idx][i] = np.random.normal()
    
    # getting fitness scores
    def get_scores(self) -> np.ndarray:
        return np.array([i(self.features, self.labels)[1] for i in self.population])

    # performs selection
    def selection(self):
        self.population.sort(key=lambda x: x(self.features, self.labels)[1], reverse=True)
        self.population = self.population[self.selector:]

    # choose 2 parents from the population
    def choose_parents(self) -> list[NN]:
        fit_scores = self.get_scores()
        w = [float(i)**(-1)/sum(fit_scores) for i in fit_scores]
        
        return choices(self.population, weights=w, k=2)

    # makes crossover k-times, generates offsprings
    def perform_crossover(self, k: int):
        for _ in range(k):
            parent1, parent2 = self.choose_parents()
            yield self.uniform_crossover(parent1, parent2, 0.5)

    # plotting
    def draw(self, axs: np.ndarray):
        pred = self.population[-1](self.features, self.labels)[0]
        pred = np.array([np.argmax(i) for i in pred])

        axs[0].scatter(X[:, 0], X[:, 1], c=old_y)
        axs[0].set_title('Actual')

        axs[1].scatter(X[:, 0], X[:, 1], c=pred)
        axs[1].set_title('Predictions')
        
        plt.draw()
        plt.pause(0.1)

    # starts evolving k-times
    def start(self, k: int) -> list[NN]: 
        _, axs = plt.subplots(2, 1, figsize=(10, 8))

        for i in range(k):
            print(f'{i}) Mean loss: {np.mean(self.get_scores())}')

            self.selection()
            self.draw(axs)

            offsprings = list(self.perform_crossover(self.pop_size-len(self.population)))
            self.population += offsprings

        return self.population


# returns decision boundary mapping
def get_decision_boundary(model: NN, features: np.ndarray) -> tuple[np.ndarray]:
    xx, yy = np.meshgrid(np.linspace(features[:, 0].min(), features[:, 0].max(), len(features)//2),
                         np.linspace(features[:, 1].min(), features[:, 1].max(), len(features)//2))

    new_features = np.column_stack((xx.flatten(), yy.flatten()))
    
    predictions = np.round(model(new_features))
    predictions = predictions.reshape(xx.shape).detach().numpy()

    return xx, yy, predictions


if __name__ == '__main__':
    ga = GA(20, X, y, 10, 0.15)
    best_pop = ga.start(30)