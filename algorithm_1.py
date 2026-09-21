import numpy as np
from scipy.linalg import eigh
from periodically_forced_pendulum import load_and_plot_dataset
from similarity_matrix import load_similarity_matrix, get_degree_matrix, plot_similarity_matrix_heatmap
from pathlib import Path
import matplotlib.pyplot as plt

similarity_matrix_filename = Path(__file__).parent.parent / "code" / "data" / "periodically_forced_pendulum_similarity_matrix.npy"
distance_threshold_eps = 0.60

#Input

W = load_similarity_matrix(similarity_matrix_filename)
print("dimensions of W: ", W.shape)
print("leading elements of W: \n", W[:5, :5], "\n\n")

#1. supposedly, sparsify, and remove nodes with degree 0. but i might not need to remove, matrix itself is just not that big in memory.

print("Step 1: Sparsify W and remove nodes with degree 0.\n")

similarity_threshold = 1 / distance_threshold_eps
W[W < similarity_threshold] = 0

number_of_0s = np.sum(W == 0)
print("0s in W: ", number_of_0s)
print(f"percentage of non 0 elements (those that are kept) in W: {(W.size - number_of_0s) / W.size * 100:.2f}%")
print("leading elements of W after sparsification: \n", W[:5, :5], "\n\n")
plot_similarity_matrix_heatmap(W, title="W before sparsification")

num_degree_zero = np.sum(np.sum(W - np.diag(np.diag(W)), axis=1) == 0)
print(f"Number of nodes with degree 0: {num_degree_zero}\n\n")

# Find the indices of nodes with non-zero degree (excluding diagonal elements)
nonzero_degree_mask = np.sum(W - np.diag(np.diag(W)), axis=1) != 0
W_sparse = W[nonzero_degree_mask][:, nonzero_degree_mask]
W_sparse_indices = np.where(nonzero_degree_mask)[0]
print("W_sparse.shape:", W_sparse.shape)
print("Leading elements of W_sparse: \n", W_sparse[:5, :5], "\n\n")
print("Indices of leading non-zero-degree nodes: \n", W_sparse_indices[:5], "\n\n")

load_and_plot_dataset(mask_indices=W_sparse_indices) #, mask_title="after removing nodes with degree 0")

plot_similarity_matrix_heatmap(W_sparse, title="W after sparsification")

#2. calculate the Laplacian matrix L = D - W and solve generalized eigenvalue problem Lu = lambda Du

print("Step 2: Calculating Laplacian matrix L = D - W and solving generalized eigenvalue problem Lu = lambda Du.\n")

#D = get_degree_matrix(W)
#L = D - W
D = get_degree_matrix(W_sparse)
L = D - W_sparse

print("leading elements of L: \n", L[:5, :5], "\n\n")
print("leading elements of D: \n", D[:5, :5], "\n\n")

eigenvalues, eigenvectors = eigh(L, D) #eigenvalues are already sorted in ascending order
print("leading eigenvalues: \n", eigenvalues[:5], "\n\n")

plt.figure(figsize=(8, 5))
plt.plot(1 + np.arange(min(20, len(eigenvalues))), eigenvalues[: min(20, len(eigenvalues))], marker='o')
plt.xlabel(r"$i$, index of eigenvalue")
plt.ylabel(r"$\lambda_i$")
#plt.title("Sorted eigenvalues of Generalized Laplacian vs Index")
#plt.grid()
plt.xticks(1 + np.arange(min(20, len(eigenvalues))))
plt.tight_layout()
plt.show()



#3. identify number k of coherent clusters by the gap between the eigenvalues, and select these first k eigenvectors as the coherent cluster indicators.

print("Step 3: Identifying number k of coherent clusters by the gap between the eigenvalues, and selecting these first k eigenvectors as the coherent cluster indicators.\n")

def estimate_k_by_eigengap(eigenvalues, max_k=None):
    """
    Estimate the number of coherent clusters k using the eigengap heuristic.

    For sorted eigenvalues lambda_1 <= lambda_2 <= ...,
    choose k that maximizes (lambda_{k+1} - lambda_k) over a prefix.
    """
    n = len(eigenvalues)
    if n < 3:
        raise ValueError("Need at least 3 eigenvalues to apply eigengap heuristic.")

    # Keep the search in the low-frequency spectrum where cluster structure lives.
    if max_k is None:
        max_k = min(20, n - 2)
    max_k = max(1, min(max_k, n - 2))

    gaps = np.diff(eigenvalues[: max_k + 1]) #np.diff comuptes consecutive differences, i.e. lambda_{k+1} - lambda_k
    k = int(np.argmax(gaps) + 1)  # +1 to convert gap index to cluster count
    return k, gaps


k, gaps = estimate_k_by_eigengap(eigenvalues)
cluster_indicators = eigenvectors[:, :k] #TODO: why is the number of cluster indicatorss inconsistent when choosing differents eps?

print(f"Selected number of coherent clusters (k): {k}\n")
print("Leading eigengaps: \n", gaps[: min(10, len(gaps))], "\n")
print("cluster indicators shape: ", cluster_indicators.shape, "\n\n")

# Show eigengaps used to pick k, force integer ticks and step for x-axis (index counter)
plt.figure(figsize=(8, 4))
x_vals = np.arange(1, len(gaps) + 1)
plt.plot(x_vals, gaps, marker="o")
plt.axvline(k, color="r", linestyle="--", label=f"selected $k = {k}$")
plt.xlabel(r"$i$, index of eigenvalue")
plt.ylabel(r"$\lambda_{i+1} - \lambda_i$")
#plt.title("Eigengap heuristic for coherent cluster count")
#plt.grid()
plt.legend()
plt.xticks(x_vals)  # force integer ticks at each index value
plt.tight_layout()
plt.show()

#4 Apply k-means clustering to the k cluster indicators eigenvectors to get the coherent clusters. incoherent set is the complement of the resulting coherent sets.

print("Step 4: Applying K-means clustering to extract clusters.\n")

from scipy.cluster.vq import kmeans2

U = cluster_indicators
K_clusters = k #+ 1 #TODO: at this resolution, doing eps neighbrohood sparsing takes out all the orbits from the incoherent set, so there is no +1 to search for. thik about what can be conde. remove the +1 for lower reses?

# Normalize rows of U to unit length (often helps spectral clustering, 
# though basic Algorithm 1 doesn't strictly mandate it, it's standard practice)
# U_norm = U / np.linalg.norm(U, axis=1, keepdims=True) 

# K-means in spectral space R^k
centroids, labels = kmeans2(U, K_clusters, minit='++', iter=1000)


print(f"Clustering complete. Assigned {len(labels)} trajectories to K = {K_clusters} clusters.")


# To plot this back in physical space, we map the labels to the original grid indices
full_labels = np.full(nonzero_degree_mask.shape, -1, dtype=int) #TODO: this doesnt seem right. is the -1 supposed to be the uncoherent class, when it's already supposed to be assigned? to be adressed when fixing the algorithm.
full_labels[nonzero_degree_mask] = labels


print("Cluster labels (first 20):", labels[:20])

load_and_plot_dataset(
    ic_only=True,
    #mask_title="colored by clustering label",
    #mask_indices=W_sparse_indices,
    labels=full_labels,
    #labels=labels,
)