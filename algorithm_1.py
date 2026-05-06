import numpy as np
from scipy.linalg import eigh
from similarity_matrix import load_similarity_matrix, get_degree_matrix
from pathlib import Path
import matplotlib.pyplot as plt

similarity_matrix_filename = Path(__file__).parent.parent / "code" / "data" / "periodically_forced_pendulum_similarity_matrix.npy"
eps = 0.45

#Input

W = load_similarity_matrix(similarity_matrix_filename)
print("dimensions of W: ", W.shape)
print("leading elements of W: \n", W[:5, :5], "\n\n")

#1. supposedly, sparsify, and remove nodes with degree 0. but i might not need to remove, matrix itself is just not that big in memory.

W[W < eps] = 0 #TODO: test in actually removing all the nodes with degree 0.

print("0s in W: ", np.sum(W == 0))
print("leading elements of W after sparsification: \n", W[:5, :5], "\n\n")

num_degree_zero = np.sum(np.sum(W - np.diag(np.diag(W)), axis=1) == 0)
print(f"Number of nodes with degree 0: {num_degree_zero}\n\n")

# Find the indices of nodes with non-zero degree (excluding diagonal elements)
nonzero_degree_mask = np.sum(W - np.diag(np.diag(W)), axis=1) != 0
W_sparse = W[nonzero_degree_mask][:, nonzero_degree_mask]
W_sparse_indices = np.where(nonzero_degree_mask)[0]
print("W_sparse.shape:", W_sparse.shape)
print("Leading elements of W_sparse: \n", W_sparse[:5, :5], "\n\n")
print("Indices of non-zero-degree nodes: \n", W_sparse_indices[:5], "\n\n")


#2. calculate the Laplacian matrix L = D - W and solve generalized eigenvalue problem Lu = lambda Du

#D = get_degree_matrix(W) #TODO: test in not including the huge term K in the diagonal.
#L = D - W
D = get_degree_matrix(W_sparse)
L = D - W_sparse

print("leading elements of L: \n", L[:5, :5], "\n\n")
print("leading elements of D: \n", D[:5, :5], "\n\n")

eigenvalues, eigenvectors = eigh(L, D)
print("leading eigenvalues: \n", eigenvalues[:5], "\n\n")

plt.figure(figsize=(8, 5))
plt.plot(np.arange(len(eigenvalues)), eigenvalues, marker='o')
plt.xlabel("Index of eigenvalue")
plt.ylabel("Eigenvalue")
plt.title("Sorted eigenvalues of Generalized Laplacian vs Index")
plt.grid()
plt.tight_layout()
plt.show()



#3. identify number k of coherent clusters by the gap between the eigenvalues, and select these first k eigenvectors as the coherent cluster indicators.

