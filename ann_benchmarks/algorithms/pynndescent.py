from __future__ import absolute_import
import pynndescent
from ann_benchmarks.algorithms.base import BaseANN


class PyNNDescent(BaseANN):
    def __init__(self, metric, n_neighbors=10, n_jobs=4):
        self._n_neighbors = int(n_neighbors)
        self._queue_size = None
        self._n_search_trees = 0
        self._n_jobs = int(n_jobs)
        self._pynnd_metric = {'angular': 'cosine',
                              'euclidean': 'euclidean',
                              'hamming': 'hamming',
                              'jaccard': 'jaccard'}[metric]

    def fit(self, X):
        self._index = pynndescent.NNDescent(X,
                                            n_neighbors=self._n_neighbors,
                                            metric=self._pynnd_metric,
                                            low_memory=True,
                                            n_jobs=self._n_jobs)
        self._index._init_search_graph()

    def set_query_arguments(self, queue_size, n_search_trees):
        self._queue_size = float(queue_size)
        self._n_search_trees = int(n_search_trees)

    def query(self, v, n):
        ind, dist = self._index.query(
            v.reshape(1, -1).astype('float32'), k=n,
            queue_size=self._queue_size,
            n_search_trees=self._n_search_trees)
        return ind[0]

    def __str__(self):
        str_template = ('PyNNDescent(n_neighbors=%d, n_search_trees=%d'
                        ', queue_size=%.2f)')
        return str_template % (self._n_neighbors, self._n_search_trees,
                               self._queue_size)
