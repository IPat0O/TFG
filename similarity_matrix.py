import numpy as np
from pathlib import Path #solves issues with paths when i tried debugging
import os
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
dataset_path = BASE_DIR / "code" / "data" / "periodically_forced_pendulum.npz"
similarity_matrix_path = BASE_DIR / "code" / "data" / "periodically_forced_pendulum_similarity_matrix"

K = 1e6
eps = 0.45

def load_dataset(dataset_path):
    """
    Load the dataset from the given path
    Returns a tuple (t, trajectories)
    """
    dataset = np.load(dataset_path, allow_pickle=True)
    return dataset['t'], dataset['trajectories']

def get_dataset_dimensions(dataset):
    """
    Gets the dimensions (n, m, d) of the dataset,
    where n is the number of trajectories,
    m is the number of time steps,
    and d is the dimension of the configuration space.
    """
    return dataset.shape

def dynamical_distance(t, trajectory_i, trajectory_j):
    """
    Calculates the dynamical distance between two trajectories.
    Returns a scalar distance value.
    """

    t_end = t[-1]
    t_0 = t[0]

    sum = 0
    for k in range(len(t) - 1):
        xi_current = trajectory_i[k]
        xj_current = trajectory_j[k]
        xi_next = trajectory_i[k + 1]
        xj_next = trajectory_j[k + 1]

        sum += (t[k + 1] - t[k]) / 2 * (
            np.linalg.norm(xi_next - xj_next) + np.linalg.norm(xi_current - xj_current)
        )

    r_ij = sum / (t_end - t_0)

    return r_ij

#Since W is symmetric, i calculate only the upper triangular part and then mirror it. but, should i save only the upper triangular part instead? #TODO: think about this. its not that relevant at this scale.
def get_similarity_matrix(t, trajectories, dimensions):
    """
    Calculates the pairwise similarity scores w_ij between all trajectories in the dataset.
    Each w_ij is calculated as w_ij = 1/r_ij, where r_ij is a distance measure between the i-th and j-th trajectories.
    Returns a matrix of shape n x n, with elements the similarity scores w_ij.
    """
   
    n, m, d = dimensions
    W = np.zeros((n, n))

    total_pairs = (n * (n + 1)) // 2
    computed_pairs = 0

    for i in range(n):
        for j in range(i, n):
            if i == j:
                W[i, j] = K
            else:
                w_ij = 1 / dynamical_distance(t, trajectories[i], trajectories[j])

                W[i, j] = w_ij
                W[j, i] = w_ij

            computed_pairs += 1
            if computed_pairs % 10000 == 0 or computed_pairs == total_pairs:
                progress = computed_pairs / total_pairs * 100
                print(f"\rComputing similarity matrix: {computed_pairs}/{total_pairs} ({progress:.1f}%)", end='', flush=True)
    print("\nSimilarity matrix computation completed.")
    return W

def get_degree_matrix(W): #calculated after W is sparsified with eps-neighborhood strategy. is this okay? #TODO: think about this
    """
    Calculates the degree matrix D of the similarity matrix W.
    Returns a diagonal matrix of shape n x n, with elements the degrees of the nodes.
    """
    # Exclude the diagonal elements (W[i,i]) from the degree sum
    #W_no_diag = W.copy()
    #np.fill_diagonal(W_no_diag, 0)
    #return np.diag(np.sum(W_no_diag, axis=1))
    return np.diag(np.sum(W, axis=1))

def save_similarity_matrix(W, similarity_matrix_path):
    """
    Saves the similarity matrix to the specified path in the form of a numpy array.
    """
    np.save(similarity_matrix_path, W)

def load_similarity_matrix(similarity_matrix_filename):
    """
    Loads the similarity matrix from the specified path in the form of a numpy array.
    """
    return np.load(similarity_matrix_filename)


def _mask_to_indices(mask, n):
    """
    Convert a boolean mask or index array to integer indices for submatrix selection.
    """
    mask = np.asarray(mask)
    if mask.dtype == bool:
        if mask.shape != (n,):
            raise ValueError(f"Boolean mask must have length {n}, got {mask.shape}.")
        return np.where(mask)[0]
    return mask.astype(int)


def plot_similarity_matrix_heatmap(
    W,
    mask=None,
    title=None,
    cmap="viridis",
    exclude_diagonal_from_scale=True,
    show=True,
):
    """
    Plot a heatmap of the similarity matrix W, optionally restricted by a mask.

    Parameters
    ----------
    W : ndarray, shape (n, n)
        Similarity matrix.
    mask : array-like, optional
        Boolean mask of length n, or 1D integer indices selecting rows/columns.
        If None, the full matrix is plotted.
    title : str, optional
        Plot title.
    cmap : str
        Matplotlib colormap name.
    exclude_diagonal_from_scale : bool
        If True, color limits ignore diagonal entries (useful when W[i,i] = K is huge).
    show : bool
        If True, call plt.show().
    """
    W = np.asarray(W)
    n = W.shape[0]
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square 2D array.")

    if mask is not None:
        idx = _mask_to_indices(mask, n)
        W_plot = W[np.ix_(idx, idx)]
    else:
        W_plot = W

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_data = W_plot.copy()
    if exclude_diagonal_from_scale and plot_data.size > 0:
        off_diag = plot_data.copy()
        np.fill_diagonal(off_diag, np.nan)
        vmin = np.nanmin(off_diag) if np.any(~np.isnan(off_diag)) else None
        vmax = np.nanmax(off_diag) if np.any(~np.isnan(off_diag)) else None
    else:
        vmin = vmax = None

    # W[i, j]: row i (y-axis), column j (x-axis) in imshow convention
    im = ax.imshow(plot_data, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xlabel(r"$j$-th trajectory")
    ax.set_ylabel(r"$i$-th trajectory")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"Similarity matrix heatmap ({W_plot.shape[0]} × {W_plot.shape[0]})")
    fig.colorbar(im, ax=ax, label=r"$w_{ij}$")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def test_similarity_matrix():
    """
    Test the similarity matrix calculation
    """
    t, trajectories = load_dataset(dataset_path)
    dimensions = get_dataset_dimensions(trajectories)
    print(dimensions)
    W = get_similarity_matrix(t, trajectories, dimensions)
    print(W)

def generate_and_save_similarity_matrix():
    """
    Generates the similarity matrix from an existing dataset and saves the similarity matrix to a file.
    """

    if os.path.exists(similarity_matrix_path):
        print(f"Similarity matrix already exists at {similarity_matrix_path}")
        print("Overwrite? (y/n)")
        overwrite = input()
        if overwrite != "y":
            print("Similarity matrix not overwritten. Exiting.")
            return
        else:
            print("Overwriting similarity matrix...")
            os.remove(similarity_matrix_path)

    t, trajectories = load_dataset(dataset_path)
    dimensions = get_dataset_dimensions(trajectories)
    print("dimensions: ", dimensions)
    W = get_similarity_matrix(t, trajectories, dimensions)

    print("Similarity matrix generated successfully. Saving to file at {similarity_matrix_path}...")
    save_similarity_matrix(W, similarity_matrix_path)
    print(f"Similarity matrix saved successfully to file at {similarity_matrix_path}.")

if __name__ == "__main__":
    generate_and_save_similarity_matrix()