import nmslib
import numpy as np


def nearest_neighbours(data, query, k=5):
    index = nmslib.init(method='hnsw', space='cosinesimil')

    index.addDataPointBatch(data)
    index.createIndex({'post': 2}, print_progress=True)
    index.saveIndex('nmslib.index')

    return index.knnQueryBatch(query, k=k)

if __name__ == '__main__':

    x = np.random.random((10, 100)) # make random pts (100 pts in 10D space)
    print(nearest_neighbours(x[:, :-1], x[:, -1]))
