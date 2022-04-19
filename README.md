## <span style="font-family: serif; font-size: 2em; font-style: italic;">L<sup><sup>p,q</sup></sup> Tree</span>
**LpqTree** is a C++ library that generalize k-d trees to *L<sup>p,q</sup>* Minkowski *mixed-norm* (*entry-wise matrix distance*).  
It can be used for, radius search and k-nearest neighbors search for a list of M×N matrices.


### Distance / Norm computation
The *L<sup>p,q</sup>* norm is defined like this :  
[![Lpq](https://latex.codecogs.com/svg.image?\large&space;\Vert&space;A&space;\Vert_{p,q}&space;=&space;&space;\bigg(\sum_{j=1}^M&space;\bigg(&space;\sum_{i=1}^N&space;|A_{ij}|^p&space;\bigg)^{\frac{q}{p}}\bigg)^{\frac{1}{q}})](https://en.wikipedia.org/wiki/Matrix_norm#%22Entry-wise%22_matrix_norms)  
In the **LpqTree** code, matrix row and column are swapped, for faster operation along the last axis:
```python
import numpy as np
def Lpq_norm(A, p, q):
    return np.sum( np.sum( np.abs(A)**p, axis=-1 )**(q/p), axis=-1)**(1.0/q)
```

Distance computation is optimized for on *L<sup>2,1</sup>* applications, with M×2, M×3 or M×4 structures.
Meanwhile, it also works for any M×N matrices using *L<sup>p,q</sup>*, where 1 ≤ p ≤ 9 and 1 ≤ q ≤ 9.


### K-d Tree Implementation
Modified version of [Nanoflann](https://github.com/jlblancoc/nanoflann) k-d trees to support *L<sup>p,q</sup>* norm.  
It includes python binder, pybind11 , based of [pynanoflann](https://github.com/u1234x1234/pynanoflann).


### Installation
```
pip install lpqtree
```

### Python Usage
```python
import numpy as np
import lpqtree
import lpqtree.lpqpydist as lpqdist

# Create 3 matrices composed of four 2D points (4x2)
matrices_a = np.array([[[0.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]],
                       [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                       [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]])

# Create 2 matrices composed of four 2D points (4x2)
matrices_b = np.array([[[0.1, 0.1], [1.5, 0.8], [1.8, 1.0], [2.5, 2.0]],
                       [[3.0, 2.2], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]]])

# Search metric Lpq apply the "p-norm" along the last axis and the "q-norm" after
lpq_metric = "l21" # L21 is equivalent to the Sum of L2 norm

# Compute the Lpq-norm of each matrix in "matrices_a"
lpqdist.lpq_str_switch(matrices_a, lpq_metric)
# lpqdist.l21(matrices_a) == lpqdist.l1(lpqdist.l2(matrices_a))

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
