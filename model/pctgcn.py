import torch
import torch.nn as nn

from utils.common import import_class
from utils.nn_init import bn_init
from graph.tools import get_hop_distance_matrix

from .gcn.pctgcn_util import TCN_GCN_unit
from .head.gcn_head import GCNHead


class PCTGCN(nn.Module):

    def __init__(self,
                 A,
                 G_dist,
                 num_point,
                 num_person,
                 in_channels,
                 embedding_channels,
                 channels_scale,
                 num_subset=8,
                 rel_reduction=4,
                 topk=9,
                 route_window=1):
        super().__init__()

        self.num_point = num_point
        self.embedding_channels = embedding_channels

        self.data_bn = nn.BatchNorm1d(num_person * self.embedding_channels * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, self.embedding_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.embedding_channels))
        self.tanh = nn.Tanh()

        C = embedding_channels
        common = dict(num_subset=num_subset, rel_reduction=rel_reduction,
                      topk=topk, route_window=route_window)

        self.l1 = TCN_GCN_unit(C, C, num_point, A, G_dist, **common)
        self.l2 = TCN_GCN_unit(C, C, num_point, A, G_dist, **common)
        self.l3 = TCN_GCN_unit(C, C, num_point, A, G_dist, **common)
        self.l4 = TCN_GCN_unit(C, C * 2, num_point, A, G_dist, stride=2, **common)
        self.l5 = TCN_GCN_unit(C * 2, C * 2, num_point, A, G_dist, **common)
        self.l6 = TCN_GCN_unit(C * 2, C * 2, num_point, A, G_dist, **common)
        self.l7 = TCN_GCN_unit(C * 2, C * channels_scale, num_point, A, G_dist, stride=2, **common)
        self.l8 = TCN_GCN_unit(C * channels_scale, C * channels_scale, num_point, A, G_dist, **common)
        self.l9 = TCN_GCN_unit(C * channels_scale, C * channels_scale, num_point, A, G_dist, **common)

        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, _, T, V, M = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = self.to_joint_embedding(x)
        x += self.pos_embedding[:, :self.num_point]
        x = self.tanh(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(N, M * V * self.embedding_channels, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, self.embedding_channels, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, self.embedding_channels, T, V)

        x, orth1 = self.l1(x)
        x1 = x
        x, orth2 = self.l2(x)
        x, orth3 = self.l3(x + x1)

        x, orth4 = self.l4(x)
        x4 = x
        x, orth5 = self.l5(x)
        x, orth6 = self.l6(x + x4)

        x, orth7 = self.l7(x)
        x7 = x
        x, orth8 = self.l8(x)
        x, orth9 = self.l9(x + x7)

        c_new = x.size(1)
        x = x.view(N, M, c_new, -1).mean(3).mean(1)

        orth_loss = (orth1 + orth2 + orth3 + orth4 + orth5 + orth6 + orth7 + orth8 + orth9).view(N, M).mean(1)

        return x, orth_loss


class Model(nn.Module):

    def __init__(self,
                 num_class=60,
                 in_channels=3,
                 num_point=25,
                 num_person=2,
                 graph=None,
                 graph_args=None,
                 drop_out=0,
                 embedding_channels=128,
                 channels_scale=4,
                 num_subset=8,
                 rel_reduction=4,
                 topk=9,
                 route_window=1):
        super().__init__()

        if graph is None:
            raise ValueError("graph must be specified")
        Graph = import_class(graph)
        self.graph = Graph(**(graph_args or {}))

        G_dist = get_hop_distance_matrix(self.graph.num_node, self.graph.inward + self.graph.outward)

        self.backbone = PCTGCN(
            A=self.graph.A,
            G_dist=G_dist,
            num_point=num_point,
            num_person=num_person,
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            channels_scale=channels_scale,
            num_subset=num_subset,
            rel_reduction=rel_reduction,
            topk=topk,
            route_window=route_window,
        )
        self.head = GCNHead(
            num_class=num_class,
            in_channels=embedding_channels * channels_scale,
            dropout=drop_out,
        )

    def forward(self, x):
        x, orth_loss = self.backbone(x)
        return self.head(x), orth_loss
