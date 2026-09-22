import numpy as np

from graph import tools

num_node = 20
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 2),
    (2, 3),
    (4, 3),
    (5, 3),
    (6, 5),
    (7, 6),
    (8, 7),
    (9, 3),
    (10, 9),
    (11, 10),
    (12, 11),
    (13, 1),
    (14, 13),
    (15, 14),
    (16, 15),
    (17, 1),
    (18, 17),
    (19, 18),
    (20, 19),
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    def __init__(self,
                 hyper_joints=0,
                 labeling_mode="spatial",
                 scale=1,
                 num_filter=8,
                 hyper_pa=False,
                 hop_decay=1.0):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.hyper_joints = hyper_joints
        self.num_filter = num_filter
        self.hop_decay = hop_decay
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.H_PA = self.get_hyper_incidence_prior() if hyper_pa else None

    def get_hyper_incidence_prior(self):
        groups = tools.get_anatomical_groups_multiscale('UCLA')
        H_PA = tools.get_anatomical_hyper_incidence(
            num_node, groups, edges=inward + outward, hop_decay=self.hop_decay,
            virtual=self.hyper_joints)
        return H_PA[np.newaxis, np.newaxis, :, :]

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == "spatial":
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == "spatial_ensemble":
            A = tools.get_spatial_graph_ensemble(num_node, self_link, inward, outward, self.num_filter)
        elif labeling_mode == "virtual_ensemble":
            A = tools.get_virtual_graph_ensemble(num_node, self_link, inward, outward, self.hyper_joints, 8)
        elif labeling_mode == "random":
            A = tools.get_random_graph(num_node, num_filter=self.num_filter)
        elif labeling_mode == "normalized_spatial":
            A = tools.get_normalized_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == "normalized_spatial_k8":
            A = tools.get_normalized_spatial_graph_k8(num_node, self_link, inward, outward)
        elif labeling_mode == "hierarchical":
            A = tools.get_hdgcn_hierarchical_graph(num_node, dataset='UCLA', CoM=3)
        else:
            raise ValueError()
        return A
