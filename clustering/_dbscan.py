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
        self.cluster_assignment = np.array(
            [0 for i in range(self.data.shape[0])])
        num_clusters = 1
        for idx, val in enumerate(core_points):

            # If not core point continue
            if not val:
                continue

            # Assigning cluster to core point
            if self.cluster_assignment[idx] == 0:
                self.cluster_assignment[idx] = num_clusters
                num_clusters += 1

            # Assigning cluster to neighbouring points
            for i, d in enumerate(self.data):
                if (np.sqrt(((d - self.data[idx])**2).sum()) < self.eps) and (self.cluster_assignment[i] == 0):
                    self.cluster_assignment[i] = self.cluster_assignment[idx]
