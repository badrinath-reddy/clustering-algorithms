"""
The :mod:`clustering` module gathers some unsupervised clustering
algorithms.
"""
from ._k_means import KMeans
from ._k_medoids import KMedoids
from ._dbscan import DBScan
from ._hierarchical_agglomerative import HierarchicalAgglomerative

__all__ = ['KMeans', 'KMedoids', 'DBScan', 'HierarchicalAgglomerative']
