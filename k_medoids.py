"""
Module KMedoids
KMedoids clustering in pure Numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from get_data import get_data

K = 2
SEED = None
MAX_ITER = 100


class KMedoids:
    def __init__(self, data, k, max_iter):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.medoids = None
        self.cluster_assignment = None

    def fit(self):
        self.get_initial_medoids()
        prev_cost = self.get_cost()
        for i in range(self.max_iter):
            self.update_clusters()
            cost = self.get_cost()
            if cost >= prev_cost:
                break
            prev_cost = cost
        self.cluster_assignment = self.assign_clusters()

    # Random K elements
    def get_initial_medoids(self):
        medoids = self.data.copy()
        np.random.shuffle(medoids)
        self.medoids = medoids[:self.k]

    # Assigning nearest centriods
    def assign_clusters(self):
        distances = np.sqrt(
            ((self.data - self.medoids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    # Calculation of cost using broadcasting
    def get_cost(self):
        distances = np.sqrt(
            ((self.data - self.medoids[:, np.newaxis])**2).sum(axis=2))
        return np.min(distances, axis=0).sum()

    # Reducing cost
    def update_clusters(self):
        cost = self.get_cost()
        for d in self.data:
            if d.tolist() in self.medoids.tolist():
                continue
            for i in range(self.k):
                temp = self.medoids[i].copy()
                self.medoids[i] = d
                temp_cost = self.get_cost()
                if temp_cost > cost:
                    self.medoids[i] = temp
                    continue
                cost = temp_cost
                break


def main():
    if SEED is not None:
        np.random.seed(seed=SEED)
    data, ground_truth = get_data()
    k_medoids = KMedoids(data, K, MAX_ITER)
    k_medoids.fit()
    plt.subplot(121)
    plt.title("K Medoids")
    plt.scatter(data[:, 0], data[:, 1],
                c=k_medoids.cluster_assignment, s=50, cmap='viridis')
    plt.scatter(
        k_medoids.medoids[:, 0], k_medoids.medoids[:, 1], c='black', s=200, alpha=0.5)
    plt.subplot(122)
    plt.title("Ground Truth")
    plt.scatter(data[:, 0], data[:, 1], c=ground_truth, s=50, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    main()
