# Approximate Nearest Neighbours (ANNs)

A quick look at a few of the ANNs available.  ANNs are great for a scalable solution to nearest neighbours with lots of data.  I've looked at:

- Spotify's [Annoy](https://github.com/spotify/annoy)
	- Works better if you don't have too many dimensions (<100)
	- More trees gives higher precision, but more memory
- Facebook's [FAISS](https://github.com/facebookresearch/faiss/)
	- Indices stored in RAM
	- Can use compressed vectors, don't need to keep the original vectors. Less precise but scales to billions of vectors in main memory on a single server
	- GPU implementation option
- Non-Metric Space Library ([NMSLIB](https://github.com/nmslib/nmslib))
	- Only static data supported
	- Can use Hierarchical Navigable Small World Graph (HNSW), best method when [benchmarked](https://raw.githubusercontent.com/erikbern/ann-benchmarks/master/results/glove-100-angular.png)
- Facebook's [PySparnn](https://github.com/facebookresearch/pysparnn)
	- For sparse and high dimensional data