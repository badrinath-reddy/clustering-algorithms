"""
Module HierarchicalAgglomerative
Hierarchial clustering in pure Numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from get_data import get_data

K = 2

# 1 - Single Linkage
# 2 - Complete Linkage
# 3 - Average Linkage


class HierarchicalAgglomerative:
    def __init__(self, data, k, linkage):
        self.data = data
        self.k = k
        self.linkage = linkage
        self.cluster_assignment = None
        self.cluster_sizes = None  # For average linkage
        self._matrix = None

    def fit(self):
        self.initialize()
        for i in range(self.data.shape[0] - self.k):
            self.group()

    # Intializing matrix, cluster_assignment and cluster_sizes
    def initialize(self):
        self.cluster_assignment = []
        self.cluster_sizes = []
        self.calculate_matrix()

        for i in range(self.data.shape[0]):
            self.cluster_assignment.append(i)
            self.cluster_sizes.append(1)

        self.cluster_assignment = np.array(self.cluster_assignment)
        self.cluster_sizes = np.array(self.cluster_sizes)

    # Intial matrix calculation
    def calculate_matrix(self):
        ans = np.zeros((self.data.shape[0], self.data.shape[0]))
        ans += np.inf
        for i in range(self.data.shape[0]):
            for j in range(i):
                ans[i][j] = np.linalg.norm(self.data[i] - self.data[j])
        self._matrix = ans

    def merge(self, x, y, s_x, s_y):
        if self.linkage == 1:
            return min(x, y)
        elif self.linkage == 2:
            return max(x, y)
        else:
            return (x * s_x + y * s_y) / (s_x + s_y)

    def group(self):

        min_idx = np.unravel_index(
            np.argmin(self._matrix), self._matrix.shape)  # min index in matrix

        # Updating cluster assignment
        for i in range(self.data.shape[0]):
            if self.cluster_assignment[i] == min_idx[0]:
                self.cluster_assignment[i] = min_idx[1]
            if self.cluster_assignment[i] > min_idx[0]:
                self.cluster_assignment[i] -= 1

        # Updating row
        for i in range(self._matrix.shape[0]):
            if self._matrix[min_idx[1]][i] == np.inf:
                break
            self._matrix[min_idx[1]][i] = self.merge(
                self._matrix[min_idx[1]][i], self._matrix[min_idx[0]][i], self.cluster_sizes[min_idx[1]], self.cluster_sizes[min_idx[0]])

        # Updating column
        for i in range(self._matrix.shape[0]):
            if self._matrix[i][min_idx[1]] == np.inf:
                continue
            if self._matrix[min_idx[0]][i] != np.inf:
                self._matrix[i][min_idx[1]] = self.merge(
                    self._matrix[i][min_idx[1]],  self._matrix[min_idx[0]][i], self.cluster_sizes[min_idx[1]], self.cluster_sizes[min_idx[0]])
            else:
                self._matrix[i][min_idx[1]] = self.merge(
                    self._matrix[i][min_idx[1]],  self._matrix[i][min_idx[0]], self.cluster_sizes[min_idx[1]], self.cluster_sizes[min_idx[0]])

        self.cluster_sizes[min_idx[1]] += self.cluster_sizes[min_idx[0]]
        self.cluster_sizes = np.delete(self.cluster_sizes, min_idx[0])

        # Deletion
        self._matrix = np.delete(self._matrix, min_idx[0], 0)
        self._matrix = np.delete(self._matrix, min_idx[0], 1)


def main():
    data, ground_truth = get_data()
    dict = {1: "Single Linkage", 2: "Complete Linkage", 3: "Average Linkage"}
    for i in range(1, 4):
        ha = HierarchicalAgglomerative(data, K, i)
        ha.fit()
        plt.subplot(220 + i)
        plt.title("Hierarchical Agglomerative " + dict[i])
        plt.scatter(data[:, 0], data[:, 1],
                    c=ha.cluster_assignment, s=50, cmap='viridis')

    plt.subplot(224)
    plt.title("Ground Truth")
    plt.scatter(data[:, 0], data[:, 1], c=ground_truth, s=50, cmap='viridis')
    plt.show()


if __name__ == "__main__":
    main()
