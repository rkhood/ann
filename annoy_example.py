import annoy
import numpy as np


def nearest_neighbours(data, query, k=5):
    n = data.shape[0]
    index = annoy.AnnoyIndex(n, metric='euclidean')

    for i, v in enumerate(data.T):
        index.add_item(i, v.ravel())

    index.build(20) # build 20 trees (more trees more accurate/slower)
    index.save('annoy.index')

    return index.get_nns_by_vector(query, k, include_distances=True)


if __name__ == '__main__':

    x = np.random.random((10, 100)) # make random pts (100 pts in 10D space)
    print(nearest_neighbours(x[:, :-1], x[:, -1]))
