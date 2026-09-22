import math
import os
import random

import numpy as np
from torch.utils.data import Dataset

from feeders import tools
from feeders.bone_pairs import get_bone_pairs


class Feeder(Dataset):
    def __init__(self,
                 data_path='datasets/processed/NW_UCLA',
                 split='train',
                 window_size=64,
                 repeat=1,
                 num_persons=1,
                 normalization=False,
                 normalize_coordinates=True,
                 random_rot=False,
                 bone=False,
                 vel=False,
                 bone_topology='default',
                 p_interval=None,
                 frame_sample=None,
                 num_clips=1):
        self.data_path = data_path
        self.split = split
        self.window_size = window_size
        self.repeat = repeat
        self.num_persons = num_persons
        self.normalization = normalization
        self.normalize_coordinates = normalize_coordinates
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.bone_topology = bone_topology
        self.bone_pairs = get_bone_pairs('nw_ucla', bone_topology)
        self.p_interval = p_interval
        self.frame_sample = frame_sample
        self.num_clips = num_clips
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        if self.split == 'train':
            self.data = np.load(os.path.join(self.data_path, 'x_train.npy'))
            self.label = np.where(np.load(os.path.join(self.data_path, 'y_train.npy')) > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = np.load(os.path.join(self.data_path, 'x_test.npy'))
            self.label = np.where(np.load(os.path.join(self.data_path, 'y_test.npy')) > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')

        N, T, V, C = self.data.shape
        self.data = self.data.reshape((N, T, self.num_persons, V, C)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return int(len(self.label) * self.repeat)

    def __iter__(self):
        return self

    def _sample_one_clip(self, data_numpy, valid_frame_num):
        C, _, V, M = data_numpy.shape

        cropped_frames = valid_frame_num
        if self.p_interval is not None:
            if len(self.p_interval) == 2:
                p = random.uniform(self.p_interval[0], self.p_interval[1])
            else:
                p = self.p_interval[0]
            cropped_frames = max(1, int(valid_frame_num * p))

        center = data_numpy[:, 0, 1, :].reshape(C, 1, 1, M)
        value = data_numpy[:, :cropped_frames, :, :] - center

        use_resize = (self.frame_sample == 'resize') or (self.split != 'train')
        if not use_resize:
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)
            value = self.rand_view_transform(value, agx, agy, s)
            needed_repeats = (self.window_size + cropped_frames - 1) // cropped_frames
            idx = random.sample(list(range(cropped_frames)) * needed_repeats, self.window_size)
            idx.sort()
        else:
            idx = np.linspace(0, cropped_frames - 1, self.window_size).astype(np.int64)

        if self.normalize_coordinates:
            epsilon = 1e-8
            min_vals = np.min(value.reshape(C, -1), axis=1).reshape(C, 1, 1, 1)
            max_vals = np.max(value.reshape(C, -1), axis=1).reshape(C, 1, 1, 1)
            value = (value - min_vals) / (max_vals - min_vals + epsilon)
            value = value * 2 - 1

        clip = np.zeros((C, self.window_size, V, M))
        clip[:] = value[:, idx]

        if self.random_rot:
            clip = tools.random_rot(clip)

        if self.bone:
            bone_clip = np.zeros_like(clip)
            for v1, v2 in self.bone_pairs:
                bone_clip[:, :, v1] = clip[:, :, v1] - clip[:, :, v2]
            clip = bone_clip

        if self.vel:
            clip[:, :-1] = clip[:, 1:] - clip[:, :-1]
            clip[:, -1] = 0

        return clip

    def __getitem__(self, index):
        id = index % len(self.label)
        data_numpy = np.array(self.data[id])
        label = self.label[id]

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        if self.num_clips > 1:
            assert self.split == 'test', "num_clips > 1 is only supported for test (TTA)"
            clips = [self._sample_one_clip(data_numpy, valid_frame_num)
                     for _ in range(self.num_clips)]
            return np.stack(clips, axis=0), label, index

        return self._sample_one_clip(data_numpy, valid_frame_num), label, index

    def rand_view_transform(self, X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        transform_matrix = np.dot(Ry, np.dot(Rx, Ss))
        C, T, V, M = X.shape
        X = X.reshape(C, -1).T
        X = np.dot(X, transform_matrix)
        X = X.T.reshape(C, T, V, M)
        return X

    def top_k(self, score, top_k):
        rank = score.argsort()
        labels = self.label[:len(rank)]
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(labels)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
