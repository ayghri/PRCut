import numpy as np
import scipy.sparse as SP
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import torch


def compute_sp_similarities(affinity_mat):
    maxes = affinity_mat.max(axis=1).data
    sigma = 1
    if len(maxes) > 0:
        sigma = np.median(maxes)
    similarity_mat = (affinity_mat + affinity_mat.getH()) / 2
    similarity_mat.data = np.exp(-(similarity_mat.data**2) / sigma**2)
    return similarity_mat


def get_knn_distances(x_left, x_right, num_neighbors, mode, metric) -> SP.csr_matrix:
    affinity_mat_1 = (
        NearestNeighbors(n_neighbors=num_neighbors, metric=metric)
        .fit(x_right)
        .kneighbors_graph(x_left, mode=mode)
    )

    # (num_right, num_left)
    affinity_mat_2 = (
        NearestNeighbors(n_neighbors=num_neighbors)
        .fit(x_left)
        .kneighbors_graph(x_right, mode=mode)
        .T
    )
    return 0.5 * (affinity_mat_1 + affinity_mat_2)


def sp_knn_similarity(
    x_left: np.ndarray,
    x_right: np.ndarray,
    num_neighbors: int,
    mode: str,
    metric: str,
):
    W = get_knn_distances(
        x_left, x_right, num_neighbors=num_neighbors, mode=mode, metric=metric
    )
    S = compute_sp_similarities(W)

    return S


def torch_knn_similarity(
    x_left: torch.Tensor,
    x_right: torch.Tensor,
    num_neighbors: int,
    mode="distance",
    metric="minkowski",
):
    return torch.tensor(
        sp_knn_similarity(
            x_left.cpu().numpy(),
            x_right.cpu().numpy(),
            num_neighbors,
            mode=mode,
            metric=metric,
        ).toarray()
    ).to(x_left)


def torch_pairwise_similarities(x1, x2, factor=1.0) -> torch.Tensor:
    distances = torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0)).squeeze(0)
    # distances = torch.nn.functional.pairwise_distance(x1, x2)
    sigma = torch.median(distances.max(1)[0])
    return torch.exp(-((distances / sigma / factor) ** 2))


def global_similarity_measure(similarity_matrix, left_indices, right_indices, device):
    """
    :param similarity_matrix:
    :param left_indices:
    :param right_indices:
    :param device:
    :return:
    """
    sim = torch.tensor(similarity_matrix[left_indices][:, right_indices].toarray()).to(
        device
    )
    return sim


def build_perfect_similarities(labels, num_neighbors=10):
    uni_labels = np.unique(labels)
    rows = []
    columns = []
    for lab in uni_labels:
        similar = np.where(labels == lab)[0]
        for s in similar:
            neighbors = np.random.choice(similar, num_neighbors, replace=False)
            for i in range(num_neighbors):
                rows.append(s)
                columns.append(neighbors[i])
                rows.append(neighbors[i])
                columns.append(columns)
    similarities = SP.coo_array(
        (np.ones(len(rows)), (rows, columns)), shape=(labels.shape[0], labels.shape[0])
    )
    return similarities


def compute_torch_similarities(affinity_mat):
    sigma = torch.median(affinity_mat.max(1)[0])
    similarity_mat = (affinity_mat + affinity_mat.t()) / 2
    similarity_mat = torch.exp(-(similarity_mat**2) / sigma**2)
    return similarity_mat


def compute_sp_laplacian(affinity_mat):
    similarity_mat = compute_sp_similarities(affinity_mat)
    degree_mat = SP.dia_array((similarity_mat.sum(0), [0]), shape=similarity_mat.shape)
    return degree_mat - similarity_mat
