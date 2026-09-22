from collections import deque

import numpy as np


def get_sgp_mat(num_in, num_out, link):
    A = np.zeros((num_in, num_out))
    for i, j in link:
        A[i, j] = 1
    A_norm = A / np.sum(A, axis=0, keepdims=True)
    return A_norm


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def get_k_scale_graph(scale, A):
    if scale == 1:
        return A
    An = np.zeros_like(A)
    A_power = np.eye(A.shape[0])
    for k in range(scale):
        A_power = A_power @ A
        An += A_power
    An[An > 0] = 1
    return An


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph_ensemble(num_node, self_link, inward, outward, nums):
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    A = I + In + Out
    A = np.repeat(A[np.newaxis, :], nums, axis=0)
    return A


def get_virtual_graph_ensemble(num_node, self_link, inward, outward, virtual, nums):
    I = np.pad(edge2mat(self_link, num_node), ((0, virtual), (0, virtual)))
    In = np.pad(edge2mat(inward, num_node), ((0, virtual), (0, virtual)))
    Out = np.pad(edge2mat(outward, num_node), ((0, virtual), (0, virtual)))
    for i in range(virtual):
        I[num_node + i, num_node + i] = 1
        In[:num_node, num_node + i] = 1
        Out[num_node + i, :num_node] = 1
    A = I + In + Out
    A = np.repeat(A[np.newaxis, :], nums, axis=0)
    return A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    A = np.stack((I, In, Out))
    return A


def get_normalized_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A_full = normalize_adjacency_matrix(edge2mat(self_link + inward + outward, num_node))
    return np.stack((I, In, Out, A_full))


def get_normalized_spatial_graph_k8(num_node, self_link, inward, outward):
    In1_raw = edge2mat(inward, num_node)
    Out1_raw = edge2mat(outward, num_node)

    I = edge2mat(self_link, num_node)
    In1 = normalize_digraph(In1_raw)
    Out1 = normalize_digraph(Out1_raw)
    Full1 = normalize_adjacency_matrix(edge2mat(self_link + inward + outward, num_node))

    In2 = normalize_digraph(k_adjacency(In1_raw, 2))
    Out2 = normalize_digraph(k_adjacency(Out1_raw, 2))
    Full2 = normalize_adjacency_matrix(k_adjacency(In1_raw + Out1_raw, 2))

    A_global = normalize_digraph(np.ones((num_node, num_node)))

    return np.stack((I, In1, Out1, Full1, In2, Out2, Full2, A_global)).astype(np.float32)


def get_random_graph(num_node, nx_node=1, num_filter=8, init_off=0.04, init_std=0.02):
    new_num_node = num_node * nx_node
    return np.random.randn(num_filter, new_num_node, new_num_node) * init_std + init_off


def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


def k_adjacency(A, k, with_self=False, self_factor=1):
    assert isinstance(A, np.ndarray)
    I = np.eye(len(A), dtype=A.dtype)
    if k == 0:
        return I
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    if with_self:
        Ak += self_factor * I
    return Ak


def get_multiscale_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    A1 = edge2mat(inward, num_node)
    A2 = edge2mat(outward, num_node)
    A3 = k_adjacency(A1, 2)
    A4 = k_adjacency(A2, 2)
    A1 = normalize_digraph(A1)
    A2 = normalize_digraph(A2)
    A3 = normalize_digraph(A3)
    A4 = normalize_digraph(A4)
    A = np.stack((I, A1, A2, A3, A4))
    return A


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_hdgcn_groups(dataset='NTU', CoM=21):
    if dataset == 'NTU':
        if CoM == 21:
            return [[21], [2, 3, 5, 9], [4, 6, 10, 1], [7, 11, 13, 17],
                    [8, 12, 14, 18], [22, 23, 24, 25, 15, 19], [16, 20]]
        elif CoM == 2:
            return [[2], [1, 21], [13, 17, 3, 5, 9], [14, 18, 4, 6, 10],
                    [15, 19, 7, 11], [16, 20, 8, 12], [22, 23, 24, 25]]
        elif CoM == 1:
            return [[1], [2, 13, 17], [14, 18, 21], [3, 5, 9, 15, 19],
                    [4, 6, 10, 16, 20], [7, 11], [8, 12, 22, 23, 24, 25]]
        else:
            raise ValueError(f'Unsupported CoM={CoM} for NTU')
    elif dataset == 'UCLA':
        if CoM == 3:
            return [[3], [2, 5, 9], [4, 6, 10, 1], [7, 11, 13, 17], [8, 12, 14, 18], [15, 19], [16, 20]]
        else:
            raise ValueError(f'Unsupported CoM={CoM} for UCLA')
    else:
        raise ValueError(f'Unsupported dataset={dataset}')


def get_hdgcn_edgeset(dataset='NTU', CoM=21):
    groups = [[i - 1 for i in g] for g in get_hdgcn_groups(dataset=dataset, CoM=CoM)]
    n = len(groups)
    identity, forward_h, reverse_h = [], [], []
    for i in range(n - 1):
        identity.append([(j, j) for j in groups[i] + groups[i + 1]])
        forward_h.append([(j, k) for j in groups[i] for k in groups[i + 1]])
        reverse_h.append([(j, k) for j in groups[-1 - i] for k in groups[-2 - i]])
    return [[identity[i], forward_h[i], reverse_h[-1 - i]] for i in range(n - 1)]


def get_hdgcn_hierarchical_graph(num_node, dataset='NTU', CoM=21):
    A = []
    for ident, fwd, rev in get_hdgcn_edgeset(dataset=dataset, CoM=CoM):
        ident_mat = edge2mat(ident, num_node)
        fwd_mat = normalize_digraph(edge2mat(fwd, num_node))
        rev_mat = normalize_digraph(edge2mat(rev, num_node))
        A.append(np.stack((ident_mat, fwd_mat, rev_mat)))
    return np.stack(A)


def get_anatomical_groups(dataset='NTU'):
    if dataset == 'NTU':
        return [
            [0, 1, 2, 3, 20],
            [4, 5, 6, 7, 21, 22],
            [8, 9, 10, 11, 23, 24],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
        ]
    elif dataset == 'UCLA':
        return [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
        ]
    else:
        raise ValueError(f'Unsupported dataset={dataset}')


def get_anatomical_groups_multiscale(dataset='NTU'):
    coarse = get_anatomical_groups(dataset)
    fine = []
    for group in coarse:
        mid = (len(group) + 1) // 2
        fine.append(group[:mid])
        fine.append(group[mid:])
    return coarse + fine


def get_hop_distance_matrix(num_node, edges):
    adj = [[] for _ in range(num_node)]
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    hop = np.full((num_node, num_node), -1, dtype=np.int64)
    for src in range(num_node):
        hop[src, src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if hop[src, v] == -1:
                    hop[src, v] = hop[src, u] + 1
                    queue.append(v)
    return hop


def get_anatomical_hyper_incidence(num_node, groups, edges, hop_decay=1.0, virtual=0):
    total_node = num_node + virtual
    num_groups = len(groups)
    H = np.zeros((total_node, num_groups), dtype=np.float32)

    hop = get_hop_distance_matrix(num_node, edges)

    for g, group in enumerate(groups):
        anchor = group[0]
        for v in group:
            H[v, g] = hop_decay ** hop[v, anchor]

    col_sum = H.sum(axis=0, keepdims=True) + 1e-6
    H = H / col_sum
    return H
