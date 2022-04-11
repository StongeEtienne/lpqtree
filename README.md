## Lpq Tree
Fast radius search for a list of MxN matrices, using Lpq Minkowski "mixed-norm" / "entry-wise matrix distance". 

Currently focussing on L21, for 3D points applications.

### Implementation
Modified version of [Nanoflann](https://github.com/jlblancoc/nanoflann) KD Tree with Lpq matrix metric.  
With python binder based of [pynanoflann](https://github.com/u1234x1234/pynanoflann).


### Installation
```
pip install lpqtree
```

### Python Usage
```python
import numpy as np
import lpqtree
import lpqtree.lpqpydist as lpqdist

# Create 3 matrix composed of four 2D points (4x2)
matrices_a = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]],
                       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                       [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]])

# Create 2 streamlines composed of four 2D points
matrices_b = np.array([[[0.1, 0.1], [1.5, 0.8], [1.8, 1.0], [2.5, 2.0]],
                      [[3.0, 2.2], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])

# Search metric Lpq apply the "p-norm" along the last axis and the "q-norm" after
lpq_metric = "l21" # L21 is equivalent to the Sum of L2 norm

# Compute the Lpq-norm of each matrix in "matrices_a"
lpqdist.lpq_str_switch(matrices_a, lpq_metric)
# == lpqdist.l1(lpqdist.l2(matrices_a))

# Compute all distances from each pair "matrices_a" to "matrices_b"
lpqdist.lpq_allpairs(matrices_a, matrices_b, p=2, q=1)

# Generate the k-d tree for the search (with "matrices_b")
mtree = lpqtree.KDTree(n_neighbors=1, radius=2.5, metric=lpq_metric)
mtree.fit(matrices_b)

# For each matrix in "matrices_a" search the nearest in "matrices_b"
nn_ids_b, nn_dist = mtree.query(matrices_a)

# For each matrix in "matrices_a" search the nearest in "matrices_b"
ids_a, ids_b, r_dists = mtree.radius_neighbors(matrices_a, radius=2.5)

# Get the adjacency matrix as scipycsr_matrix
adjacency_m = mtree.get_csr_matrix()
```

### Reference
In progress ...