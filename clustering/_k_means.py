"""
Module KMeans
KMeans clustering in pure Numpy
"""

import numpy as np


class KMeans:
    def __init__(self, data, k, max_iter):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.cluster_assignment = None
        self.centroids = None

    def fit(self):
        self.get_initial_centroids()
        previous_cluster_assignment = None
        for i in range(self.max_iter):
            self.assign_clusters()
            self.update_clusters()

            if previous_cluster_assignment is None:
                previous_cluster_assignment = self.cluster_assignment
            elif np.array_equal(previous_cluster_assignment, self.cluster_assignment):
                break
            else:
                previous_cluster_assignment = self.cluster_assignment

    # Random K elements
    def get_initial_centroids(self):
        centroids = self.data.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:self.k]

    # Assigning nearest centriods
    def assign_clusters(self):
        distances = (
            (self.data - self.centroids[:, np.newaxis])**2).sum(axis=2)
        self.cluster_assignment = np.argmin(distances, axis=0)

    # Calculating mean
    def update_clusters(self):
        self.centroids = np.array(
            [self.data[self.cluster_assignment == i].mean(axis=0) for i in range(self.k)])
