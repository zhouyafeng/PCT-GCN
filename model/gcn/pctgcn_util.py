import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn_init import conv_init, bn_init, weights_init


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            padding_mode="replicate",
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class PCTGC(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        G_dist,
        num_subset=8,
        rel_reduction=4,
        topk=9,
        route_window=1,
    ):
        super().__init__()
        K, V, _ = A.shape
        assert K == num_subset, f"graph prior has {K} subsets but num_subset={num_subset}"
        self.num_subset = num_subset
        self.route_window = route_window
        self.num_point = V
        mid_in = in_channels // num_subset
        self.mid_out = out_channels // num_subset
        self.hidden = max(1, mid_in // rel_reduction)

        self.topk = min(topk, V - 1)
        self.register_buffer("PA", torch.from_numpy(A.astype(np.float32)))

        self.G_s = nn.Parameter(torch.randn(num_subset, V, V) * 0.02)
        self.alpha_s = nn.Parameter(torch.zeros(num_subset))

        num_hop_buckets = 7
        hop_bucket = np.clip(G_dist.astype(np.int64), 0, num_hop_buckets - 1)
        self.register_buffer("hop_bucket", torch.from_numpy(hop_bucket))
        self.hop_bias = nn.Parameter(torch.zeros(num_subset, num_hop_buckets))

        self.to_V_pos = nn.Conv1d(in_channels, num_subset * self.hidden, 1, groups=num_subset)
        self.to_V_vel = nn.Conv2d(in_channels, num_subset * self.hidden, 1, groups=num_subset)
        self.to_V_acc = nn.Conv2d(in_channels, num_subset * self.hidden, 1, groups=num_subset)
        self.to_W = nn.Sequential(
            nn.Conv1d(in_channels, num_subset * self.hidden, 1, groups=num_subset),
            nn.LeakyReLU(),
            nn.Conv1d(num_subset * self.hidden, num_subset, 1),
            nn.Tanh(),
        )
        alpha_init_raw = math.log(math.exp(1.0) - 1)
        self.alpha = nn.Parameter(torch.full((1,), alpha_init_raw))
        dynamic_convs = (self.to_V_pos, self.to_V_vel, self.to_V_acc, self.to_W[0], self.to_W[2])

        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_subset)

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in (*dynamic_convs, self.conv_d):
            conv_init(m)
        bn_init(self.bn, 1e-6)

    def _a_norm(self, A):
        d = A.abs().sum(dim=-2, keepdim=True) + 1e-8
        return A / d

    @staticmethod
    def _gram_orth(X, Y, sum_dims):
        X_n = X / (X.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-8)
        Y_n = Y / (Y.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-8)
        cross = torch.matmul(X_n, Y_n.transpose(-1, -2))
        return cross.pow(2).sum(dim=sum_dims)

    def _hyper_norm(self, H, W):
        w = W.unsqueeze(-2)
        if H.dim() == w.dim() + 1:
            w = w.unsqueeze(-3)
        norm_w = torch.norm(H, 1, dim=-2, keepdim=True) + 1e-8
        w_ = w / norm_w
        H_w = H * w
        norm_v = torch.norm(H_w, 1, dim=-1, keepdim=True) + 1e-8
        h_ = H_w / norm_v
        return (h_ * w_) @ H.transpose(-1, -2)

    @staticmethod
    def _vel_acc(x):
        vel = torch.zeros_like(x)
        vel[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        acc = torch.zeros_like(vel)
        acc[:, :, 2:] = vel[:, :, 2:] - vel[:, :, 1:-1]
        acc[:, :, 1] = acc[:, :, 2]
        acc[:, :, 0] = acc[:, :, 1]
        vel[:, :, 0] = vel[:, :, 1]
        return vel, acc

    def _forward_kahr(self, x):
        N, C, T, V = x.size()
        K = self.num_subset
        t_x = x.mean(dim=2)

        vel, acc = self._vel_acc(x)
        window = min(self.route_window, T)
        assert T % window == 0, (
            f"route_window={self.route_window} (clamped to {window}) must evenly divide T={T}"
        )
        Tr = T // window
        if window > 1:
            vel_r = vel.view(N, C, Tr, window, V).mean(dim=3)
            acc_r = acc.view(N, C, Tr, window, V).mean(dim=3)
        else:
            vel_r, acc_r = vel, acc
        v_e = self.to_V_pos(t_x).unsqueeze(2) + self.to_V_vel(vel_r) + self.to_V_acc(acc_r)
        v_e = v_e.view(N, K, self.hidden, Tr, V).permute(0, 1, 3, 4, 2)

        dist = torch.cdist(v_e, v_e, compute_mode="use_mm_for_euclid_dist")
        hop_bias = self.hop_bias[:, self.hop_bucket].unsqueeze(0).unsqueeze(2)
        score = -dist + hop_bias
        topk_vals, topk_idx = torch.topk(score, self.topk, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        H = torch.zeros_like(score).scatter(-1, topk_idx, topk_weights)

        W = self.to_W(t_x)
        G = self._hyper_norm(H, W)
        alpha = F.softplus(self.alpha)
        R_d = alpha * G
        if window > 1:
            R_d = R_d.repeat_interleave(window, dim=2)
        return R_d

    def forward(self, x):
        N, _, T, V = x.size()
        K = self.num_subset

        R_d = self._forward_kahr(x)

        alpha_s = F.softplus(self.alpha_s)
        R_s = alpha_s.view(K, 1, 1) * torch.tanh(self.G_s)

        pre_x = self.conv_d(x).view(N, K, self.mid_out, T, V)
        PA_b = self.PA.unsqueeze(0).unsqueeze(2)
        Rs_b = R_s.unsqueeze(0).unsqueeze(2)
        A = self._a_norm(PA_b + Rs_b + R_d)
        y = torch.einsum("nktvw,nkctv->nkctw", A, pre_x).contiguous().view(N, -1, T, V)
        orth_loss = self._gram_orth(Rs_b, R_d, (1, 3, 4)).mean(dim=1)

        y = self.relu(self.bn(y) + self.down(x))
        return y, orth_loss


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            padding_mode="replicate",
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, "# out channels should be multiples of # branches"

        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if isinstance(kernel_size, list):
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_size, dilations):
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, kernel_size=ks, stride=stride, dilation=dilation),
            ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels),
        ))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        self.apply(weights_init)

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class TCN_GCN_unit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_point,
        A,
        G_dist,
        stride=1,
        residual=True,
        kernel_size=5,
        dilations=[1, 2],
        num_subset=8,
        rel_reduction=4,
        topk=9,
        route_window=1,
    ):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = PCTGC(
            in_channels,
            out_channels,
            A,
            G_dist,
            num_subset=num_subset,
            rel_reduction=rel_reduction,
            topk=topk,
            route_window=route_window,
        )
        self.tcn1 = MultiScale_TemporalConv(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations, residual=False
        )  # noqa
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y, orth_loss = self.gcn1(x)
        y = self.relu(self.tcn1(y) + self.residual(x))
        return y, orth_loss
