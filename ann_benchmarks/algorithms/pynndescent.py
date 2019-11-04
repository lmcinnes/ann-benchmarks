from __future__ import absolute_import
import pynndescent
from ann_benchmarks.algorithms.base import BaseANN
import numpy as np


class PyNNDescent(BaseANN):
    def __init__(self, metric, n_neighbors=10, pruning_degree_multiplier=2.0,
                 diversify_epsilon=0.5, leaf_size=30):
        self._n_neighbors = int(n_neighbors)
        self._queue_size = None
        self._n_search_trees = 0
        self._pruning_degree_multiplier = float(pruning_degree_multiplier)
        self._diversify_epsilon = float(diversify_epsilon)
        self._leaf_size = int(leaf_size)
        self._pynnd_metric = {'angular': 'cosine',
                              'euclidean': 'euclidean',
                              'hamming': 'hamming',
                              'jaccard': 'jaccard'}[metric]

    def fit(self, X):
        self._index = pynndescent.NNDescent(X.astype(np.float32, order='C'),
                                            n_neighbors=self._n_neighbors,
                                            metric=self._pynnd_metric,
                                            low_memory=False,
                                            leaf_size=self._leaf_size,
                                            pruning_degree_multiplier=self._pruning_degree_multiplier,
                                            diversify_epsilon=self._diversify_epsilon)
        self._index._init_search_graph()

    def set_query_arguments(self, epsilon=0.1, n_search_trees=1, queue_size=1.0):
        self._epsilon = float(epsilon)
        self._n_search_trees = int(n_search_trees)
        self._queue_size = float(queue_size)

    def query(self, v, n):
        ind, dist = self._index.query(
            np.expand_dims(v, axis=0).astype(np.float32, order='C'), k=n,
            queue_size=self._queue_size,
            n_search_trees=self._n_search_trees,
            epsilon=self._epsilon)
        return ind[0]

    def __str__(self):
        str_template = ('PyNNDescent(n_neighbors=%d, pruning_degree_multiplier=%.1f, '
                        'diversify_epsilon=%.2f, epsilon=%.3f)')
        return str_template % (self._n_neighbors, self._pruning_degree_multiplier,
                               self._diversify_epsilon, self._epsilon,
                               )

