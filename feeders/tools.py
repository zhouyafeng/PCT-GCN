import random

import numpy as np


def valid_crop_resize(data_numpy, valid_frame_num, p_interval, window, min_cropped_length=64):
    C, T, V, M = data_numpy.shape
    end = valid_frame_num if valid_frame_num > 0 else T
    valid_size = end
    assert valid_size > 0, "empty valid range"

    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1-p) * valid_size/2)
        data = data_numpy[:, bias:end-bias, :, :]
        cropped_length = data.shape[1]
    else:
        p = np.random.uniform(p_interval[0], p_interval[1])
        cropped_length = min(max(int(valid_size * p), min_cropped_length), valid_size)
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, bias:bias + cropped_length, :, :]
    cropped_length = data.shape[1]

    data = data.transpose(0, 2, 3, 1).reshape(C * V * M, cropped_length)
    data = data.astype(np.float32, copy=False)

    j = np.arange(window, dtype=np.float32)
    x_in = np.clip((j + 0.5) * cropped_length / window - 0.5, 0.0, cropped_length - 1.0)
    lo = np.floor(x_in).astype(np.int64)
    hi = np.minimum(lo + 1, cropped_length - 1)
    w = (x_in - lo)[None, :]
    data = data[:, lo] * (1.0 - w) + data[:, hi] * w
    return data.reshape(C, V, M, window).transpose(0, 3, 1, 2)


def downsample(data_numpy, step, random_sample=True):
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def _rot(rot):
    cos_r, sin_r = np.cos(rot), np.sin(rot)
    T = rot.shape[0]

    rx = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
    rx[:, 1, 1] = cos_r[:, 0]
    rx[:, 1, 2] = sin_r[:, 0]
    rx[:, 2, 1] = -sin_r[:, 0]
    rx[:, 2, 2] = cos_r[:, 0]

    ry = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
    ry[:, 0, 0] = cos_r[:, 1]
    ry[:, 0, 2] = -sin_r[:, 1]
    ry[:, 2, 0] = sin_r[:, 1]
    ry[:, 2, 2] = cos_r[:, 1]

    rz = np.tile(np.eye(3, dtype=np.float32), (T, 1, 1))
    rz[:, 0, 0] = cos_r[:, 2]
    rz[:, 0, 1] = sin_r[:, 2]
    rz[:, 1, 0] = -sin_r[:, 2]
    rz[:, 1, 1] = cos_r[:, 2]

    return np.matmul(np.matmul(rz, ry), rx)


def random_rot(data_numpy, theta=0.3):
    data = data_numpy.astype(np.float32)
    C, T, V, M = data.shape
    data = data.transpose(1, 0, 2, 3).reshape(T, C, V * M)
    rot = np.random.uniform(-theta, theta, 3).astype(np.float32)
    rot = np.tile(rot[None], (T, 1))
    R = _rot(rot)
    data = np.matmul(R, data)
    data = data.reshape(T, C, V, M).transpose(1, 0, 2, 3)
    return data.astype(np.float32)


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix(axis, theta):
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3, dtype=np.float32)
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                    [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                    [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]], dtype=np.float32)


def get_clips(full_kp, clip_len):
    C, T, V, M = full_kp.shape
    num_frames = int(T)

    if num_frames < clip_len:
        start = np.random.randint(0, num_frames)
        inds = (np.arange(start, start + clip_len) % num_frames)
        clip = full_kp[:, inds].copy()
    elif clip_len <= num_frames < 2 * clip_len:
        basic = np.arange(clip_len)
        inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
        offset = np.zeros(clip_len + 1, dtype=np.int64)
        offset[inds] = 1
        inds = basic + np.cumsum(offset)[:-1]
        clip = full_kp[:, inds].copy()
    else:
        bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
        bsize = np.diff(bids)
        bst = bids[:clip_len]
        offset = np.random.randint(bsize)
        inds = bst + offset
        clip = full_kp[:, inds].copy()

    return clip


def align(data_numpy):
    joint_rshoulder = data_numpy[:, 0, 8, 0]
    joint_lshoulder = data_numpy[:, 0, 4, 0]
    axis = np.cross(joint_rshoulder - joint_lshoulder, np.array([1, 0, 0]))
    angle = angle_between(joint_rshoulder - joint_lshoulder, np.array([1, 0, 0]))
    matrix_x = rotation_matrix(axis, angle)
    data_numpy = np.einsum('ctvm,kc->ktvm', data_numpy, matrix_x)

    return data_numpy


def drop_axis(data_numpy, axis=None, p=0.5):
    if axis is not None:
        axis_next = axis
    else:
        axis_next = random.randint(0, 2)

    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((T, V, M))
        temp[axis_next] = x_new
        return temp
    else:
        return data_numpy.copy()


def drop_joint(data_numpy, joint_list=None, time_range=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape

        if joint_list is not None:
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, joint_list)
            joint_list_ = sorted(joint_list_)
        else:
            random_int = random.randint(5, 15)
            all_joints = [i for i in range(V)]
            joint_list_ = random.sample(all_joints, random_int)
            joint_list_ = sorted(joint_list_)

        if time_range is not None:
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, time_range)
            time_range_ = sorted(time_range_)
        else:
            random_int = random.randint(16, 32)
            all_frames = [i for i in range(T)]
            time_range_ = random.sample(all_frames, random_int)
            time_range_ = sorted(time_range_)

        x_new = np.zeros((C, len(time_range_), len(joint_list_), M))
        temp2 = temp[:, time_range_, :, :].copy()
        temp2[:, :, joint_list_, :] = x_new
        temp[:, time_range_, :, :] = temp2
        return temp
    else:
        return data_numpy.copy()
