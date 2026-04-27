import numpy as np

dataset_path = "data/periodically_forced_pendulum.npz"
similarity_matrix_path = "data/periodically_forced_pendulum_similarity_matrix.npz"

K = 1e6
eps = 0.45

def load_dataset(dataset_path):
    """
    Load the dataset from the given path
    Returns a tuple (t, trajectories)
    """
    dataset = np.load(dataset_path, allow_pickle=True)
    return dataset['t'], dataset['trajectories'] #TODO: when generating dataset, use tbis naming convention

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
        sum += (t[k+1] - t[k])/2 * (np.abs(trajectory_i[k+1] - trajectory_j[k+1]) + np.abs(trajectory_i[k] - trajectory_j[k]))

    r_ij = sum / (t_end - t_0)

    return r_ij

#Since W is symmetric, i calculate only the upper triangular part and then mirror it. but, should i save only the upper triangular part instead? #TODO: think about this
def get_similarity_matrix(t, trajectories, dimensions, eps=eps):
    """
    Calculates the pairwise similarity scores w_ij between all trajectories in the dataset.
    Each w_ij is calculated as w_ij = 1/r_ij, where r_ij is a distance measure between the i-th and j-th trajectories.
    Returns a matrix of shape n x n, with elements the similarity scores w_ij.
    """
   
    n, m, d = dimensions
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                W[i, j] = K
            else:
                w_ij = 1 / dynamical_distance(t, trajectories[i], trajectories[j])
                if w_ij < eps:
                    W[i, j] = 0
                    W[j, i] = 0
                else:
                    W[i, j] = w_ij
                    W[j, i] = w_ij
    return W

def get_degree_matrix(W): #calculated after W is sparsified with eps-neighborhood strategy. is this okay? #TODO: think about this
    """
    Calculates the degree matrix D of the similarity matrix W.
    Returns a diagonal matrix of shape n x n, with elements the degrees of the nodes.
    """
    return np.diag(np.sum(W, axis=1)) #should i store only the diagonal elements? also, the huge term K is added to the diagonal. is this okay? #TODO: think about this, and the above in get_similarity_matrix, depends on hoe python stores matrices.

def save_similarity_matrix(W, similarity_matrix_path):
    """
    Saves the similarity matrix to a file
    """
    np.save(similarity_matrix_path, W)

def load_similarity_matrix(similarity_matrix_path):
    """
    Loads the similarity matrix from a file
    """
    return np.load(similarity_matrix_path)

def test_similarity_matrix():
    """
    Test the similarity matrix calculation
    """
    t, trajectories = load_dataset(dataset_path)
    dimensions = get_dataset_dimensions(trajectories)
    print(dimensions)
    W = get_similarity_matrix(t, trajectories, dimensions)
    print(W)

if __name__ == "__main__":
    test_similarity_matrix()