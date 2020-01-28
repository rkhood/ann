import numpy as np
import pysparnn.cluster_index as ci
from scipy.sparse import csr_matrix


def nearest_neighbours(data, query, k=5):
    n = range(data.shape[0])
    cp = ci.MultiClusterIndex(data, n)

    return cp.search(query, k=k, k_clusters=2, return_distance=True)


if __name__ == '__main__':

    x = np.random.binomial(1, 0.01, size=(100, 10))
    x = csr_matrix(x)
    print(nearest_neighbours(x[:, :-1], x[:, -1]))
