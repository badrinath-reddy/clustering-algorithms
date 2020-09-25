"""
Module KMedoids
KMedoids clustering in pure Numpy
"""

import numpy as np


class KMedoids:
    def __init__(self, data, k, max_iter, seed=None):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
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
        if self.seed is not None:
            np.random.seed(seed=self.seed)
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
        for _ in range(self.data.shape[0]):

            # Choosing random non-medoid
            idx = np.random.randint(0, self.data.shape[0] - 1)
            while(self.data[idx].tolist() in self.medoids.tolist()):
                idx = np.random.randint(0, self.data.shape[0] - 1)
            d = self.data[idx]

            for i in range(self.k):
                temp = self.medoids[i].copy()
                self.medoids[i] = d
                temp_cost = self.get_cost()
                if temp_cost > cost:
                    self.medoids[i] = temp
                    continue
                cost = temp_cost
                break
