import faiss
import numpy as np


def nearest_neighbours(data, query, k=5):
    n = data.shape[0]
    index = faiss.IndexFlatL2(n)

    index.add(data)
    faiss.write_index(index, 'faiss.index')

    return index.search(query, k)


if __name__ == '__main__':

    x = np.random.random((10, 100)) # make random pts (100 pts in 10D space)
    print(nearest_neighbours(x[:, :-1], x[:, -1]))
