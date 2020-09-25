"""
Module DBScan
DBScan clustering in pure Numpy
"""

import numpy as np


class DBScan:
    def __init__(self, data, eps, min_points):
        self.data = data
        self.eps = eps
        self.min_points = min_points
        self.cluster_assignment = None

    def fit(self):
        # Finding core points
        core_points = np.array([((np.sqrt(((self.data - d)**2).sum(axis=1))
                                  < self.eps).sum()) > self.min_points for d in self.data])

        # Intial cluster assignment
        self.cluster_assignment = np.zeros(self.data.shape[0])

        num_clusters = 1
        for i, v in enumerate(core_points):

            # If a core point
            if v:

                # Assigning cluster to core point
                if self.cluster_assignment[i] == 0:
                    self.cluster_assignment[i] = num_clusters
                    num_clusters += 1

                # Assigning cluster to neighbouring points
                for j, d in enumerate(self.data):
                    if (self.cluster_assignment[j] == 0) and (np.sqrt(((d - self.data[i])**2).sum()) < self.eps):
                        self.cluster_assignment[j] = self.cluster_assignment[i]
