import os

import numpy as np

from torch.utils.data import Dataset

from feeders import tools
from feeders.bone_pairs import get_bone_pairs


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path=None,
        p_interval=1,
        num_clips=1,
        split="train",
        random_choose=False,
        random_shift=False,
        random_move=False,
        random_rot=False,
        rot_theta=0.3,
        frame_sample="resize",
        align=False,
        spatial_flip=False,
        p_spatial_flip=0.5,
        drop_joint=False,
        drop_axis=False,
        window_size=-1,
        normalization=False,
        use_mmap=False,
        bone=False,
        vel=False,
        bone_topology="default",
        joint20_mode="per_frame",
    ):

        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_clips = num_clips
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.rot_theta = rot_theta
        self.align = align
        self.frame_sample = frame_sample
        self.spatial_flip = spatial_flip
        self.p_spatial_flip = p_spatial_flip
        self.drop_joint = drop_joint
        self.drop_axis = drop_axis
        self.bone = bone
        self.vel = vel
        self.bone_topology = bone_topology
        self.joint20_mode = joint20_mode
        self.bone_pairs = get_bone_pairs("ntu", bone_topology)
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        mmap = "r" if self.use_mmap else None
        if self.split == "train":
            self.data = np.load(os.path.join(self.data_path, "x_train.npy"), mmap_mode=mmap)
            self.label = np.where(np.load(os.path.join(self.data_path, "y_train.npy")) > 0)[1]
            self.sample_name = ["train_" + str(i) for i in range(len(self.data))]
        elif self.split == "test":
            self.data = np.load(os.path.join(self.data_path, "x_test.npy"), mmap_mode=mmap)
            self.label = np.where(np.load(os.path.join(self.data_path, "y_test.npy")) > 0)[1]
            self.sample_name = ["test_" + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError("data split only supports train/test")
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        if self.num_clips > 1:
            assert self.split == "test", "num_clips > 1 is only supported for test (TTA)"
            clips = [
                self._apply_modality(self._augment(self._sample_clip(data_numpy, valid_frame_num)))
                for _ in range(self.num_clips)
            ]
            return np.stack(clips, axis=0), label, index

        data_numpy = self._sample_clip(data_numpy, valid_frame_num)
        data_numpy = self._augment(data_numpy)
        return self._apply_modality(data_numpy), label, index

    def _sample_clip(self, data_numpy, valid_frame_num):
        if self.frame_sample == "clips":
            return tools.get_clips(data_numpy[:, :valid_frame_num], self.window_size)
        elif self.frame_sample == "resize":
            return tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        else:
            raise ValueError("frame_sample only supports clips or resize")

    def _augment(self, data_numpy):
        if self.align:
            data_numpy = tools.align(data_numpy)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy, self.rot_theta)
        if self.spatial_flip and self.split == "train" and np.random.random() < self.p_spatial_flip:
            flip_index = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 19, 12, 13, 14, 15, 20, 23, 24, 21, 22]
            data_numpy = data_numpy[:, :, flip_index, :]
        return data_numpy

    def _apply_modality(self, data_numpy):
        if self.bone:
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in self.bone_pairs:
                bone_data_numpy[:, :, v1] = data_numpy[:, :, v1] - data_numpy[:, :, v2]
            if self.joint20_mode == "per_frame":
                bone_data_numpy[:, :, 20] = data_numpy[:, :, 20]
            data_numpy = bone_data_numpy
        elif not self.vel:
            if self.joint20_mode == "zero":
                center = data_numpy[:, 0:1, 1:2, 0:1]
                data_numpy = data_numpy - center
            else:
                trajectory = data_numpy[:, :, 20]
                data_numpy = data_numpy - data_numpy[:, :, 20:21]
                data_numpy[:, :, 20] = trajectory

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        return data_numpy

    def top_k(self, score, top_k):
        rank = score.argsort()
        labels = self.label[: len(rank)]
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
