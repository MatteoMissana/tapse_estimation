from cupyx.scipy.ndimage import affine_transform
import cupy as cp
import numpy as np
import math as m


def slice_volume_x(vol, theta):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # unit vector
    e_xz = [[np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]

    center = cp.array([0, vol.shape[1] / 2, vol.shape[2] / 2], dtype=float)

    e_xz = cp.array(e_xz)
    e_xz = e_xz / cp.linalg.norm(e_xz, axis=1)[:, cp.newaxis]

    # R: rotation matrix: data_coords = center + r @ slice_coords
    ey = cp.cross(e_xz[0], e_xz[1])
    r = cp.array([e_xz[0], ey, e_xz[1]], dtype=cp.float32).T

    # free gpu mem
    del e_xz, ey

    # setup slice points P with coordinates (X, Y, 0)
    mx, mz = int(vol.shape[0]), int(vol.shape[2])
    xs = cp.arange(0.5, 0.5 + mx)
    zs = cp.arange(0.5 - mz / 2, 0.5 + mz / 2)
    pp = cp.zeros((3, mx, mz), dtype=cp.float32)
    pp[0, :, :] = xs.reshape(mx, 1)
    pp[2, :, :] = zs.reshape(1, mz)

    # Transform to data coordinates (x, y, z) - idx.shape == (3, mx, mx)
    # pure numpy solution with nearest-neighbor interpolation
    idx = cp.einsum('il,ljk->ijk', r, pp) + (0.5 + center.reshape(3, 1, 1))

    # free gpu mem
    del r, xs, zs, pp, center

    # computation of indices without einstein summation
    # idx = cp.zeros((3, mx, mz), dtype=cp.float32)
    # for i in range(r.shape[0]):
    #     for j in range(pp.shape[1]):
    #         for k in range(pp.shape[2]):
    #             idx[i, j, k] = r[i, 0]*pp[0, j, k] + r[i, 1]*pp[1, j, k] + r[i, 2]*pp[2, j, k]
    # idx += (0.5 + center.reshape(3, 1, 1))

    idx = idx.astype(cp.int16)
    # Find out which coordinates are out of range - shape (mx, mx)
    offpoints = cp.any(cp.array([idx[0, :, :] < 0,
                                 idx[0, :, :] >= vol.shape[0],
                                 idx[1, :, :] < 0,
                                 idx[1, :, :] >= vol.shape[1],
                                 idx[2, :, :] < 0,
                                 idx[2, :, :] >= vol.shape[2]
                                 ]), axis=0)

    idx[:, offpoints] = 0
    img = vol[idx[0], idx[1], idx[2]]

    # free gpu mem
    del offpoints, idx, vol

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return img


def slice_volume_y(vol, theta):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # unit vector
    e_xy = [[np.sin(theta), 0.0, np.cos(theta)], [0.0, 1.0, 0.0]]

    center = cp.array([0, vol.shape[1] / 2, vol.shape[2] / 2], dtype=float)

    e_xy = cp.array(e_xy)
    e_xy = e_xy / cp.linalg.norm(e_xy, axis=1)[:, cp.newaxis]

    # R: rotation matrix: data_coords = center + r @ slice_coords
    ez = cp.cross(e_xy[0], e_xy[1])
    r = cp.array([e_xy[0], e_xy[1], ez], dtype=cp.float32).T

    # free gpu mem
    del e_xy, ez

    # setup slice points P with coordinates (X, Y, 0)
    mx, my = int(vol.shape[0]), int(vol.shape[1])
    xs = cp.arange(0.5, 0.5 + mx)
    ys = cp.arange(0.5 - my / 2, 0.5 + my / 2)
    pp = cp.zeros((3, mx, my), dtype=cp.float32)
    pp[0, :, :] = xs.reshape(mx, 1)
    pp[1, :, :] = ys.reshape(1, my)

    # Transform to data coordinates (x, y, z) - idx.shape == (3, mx, mx)
    # pure numpy solution with nearest-neighbor interpolation
    idx = cp.einsum('il,ljk->ijk', r, pp) + (0.5 + center.reshape(3, 1, 1))

    # free gpu mem
    del r, xs, ys, pp, center

    idx = idx.astype(cp.int16)
    # Find out which coordinates are out of range - shape (mx, mx)
    offpoints = cp.any(cp.array([idx[0, :, :] < 0,
                                 idx[0, :, :] >= vol.shape[0],
                                 idx[1, :, :] < 0,
                                 idx[1, :, :] >= vol.shape[1],
                                 idx[2, :, :] < 0,
                                 idx[2, :, :] >= vol.shape[2]
                                 ]), axis=0)

    idx[:, offpoints] = 0
    img = vol[idx[0], idx[1], idx[2]]

    # free gpu mem
    del offpoints, idx, vol

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return img


def slice_volume_z(vol, theta):
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    # unit vector
    e_xy = [[1.0, 0.0, 0.0], [0.0, np.cos(theta), np.sin(theta)]]

    center = cp.array([vol.shape[0] / 2, vol.shape[1] / 2, vol.shape[2] / 2], dtype=float)

    e_xy = cp.array(e_xy)
    e_xy = e_xy / cp.linalg.norm(e_xy, axis=1)[:, cp.newaxis]

    # free gpu mem
    # R: rotation matrix: data_coords = center + r @ slice_coords
    ez = cp.cross(e_xy[0], e_xy[1])
    r = cp.array([e_xy[0], e_xy[1], ez], dtype=cp.float32).T

    # free gpu mem
    del e_xy, ez

    # setup slice points P with coordinates (X, Y, 0)
    mx, my = int(vol.shape[0]), int(vol.shape[1])
    xs = cp.arange(0.5 - mx / 2, 0.5 + mx / 2)
    ys = cp.arange(0.5 - my / 2, 0.5 + my / 2)

    pp = cp.zeros((3, mx, my), dtype=cp.float32)
    pp[0, :, :] = xs.reshape(mx, 1)
    pp[1, :, :] = ys.reshape(1, my)

    # Transform to data coordinates (x, y, z) - idx.shape == (3, mx, mx)
    # pure numpy solution with nearest-neighbor interpolation
    idx = cp.einsum('il,ljk->ijk', r, pp) + (0.5 + center.reshape(3, 1, 1))

    # free gpu mem
    del r, xs, ys, pp, center

    idx = idx.astype(cp.int16)
    # Find out which coordinates are out of range - shape (mx, mx)
    offpoints = cp.any(cp.array([idx[0, :, :] < 0,
                                 idx[0, :, :] >= vol.shape[0],
                                 idx[1, :, :] < 0,
                                 idx[1, :, :] >= vol.shape[1],
                                 idx[2, :, :] < 0,
                                 idx[2, :, :] >= vol.shape[2]
                                 ]), axis=0)

    idx[:, offpoints] = 0
    img = vol[idx[0], idx[1], idx[2]]

    # free gpu mem
    del offpoints, idx, vol

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return img


def rz(theta):  # true rx, but first dim in US volume is z-axis
    return np.array([[1, 0, 0, 0],
                     [0, m.cos(theta), -m.sin(theta), 0],
                     [0, m.sin(theta), m.cos(theta), 0],
                     [0, 0, 0, 1]])


def ry(theta):
    return np.array([[m.cos(theta), 0, m.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-m.sin(theta), 0, m.cos(theta), 0],
                     [0, 0, 0, 1]])


def rx(theta):  # true rz, but first dim in US volume is z-axis
    return np.array([[m.cos(theta), -m.sin(theta), 0, 0],
                     [m.sin(theta), m.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def t(tx, ty, tz):
    return np.array([[1, 0, 0, tx],
                     [0, 1, 0, ty],
                     [0, 0, 1, tz],
                     [0, 0, 0, 1]])


def rotate_about_axis(axis, sequence, rot_angles):
    print('Rotating volume about ' + axis + ' axis.')
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # translate to probe center, to rotate about centerline
    tm_center = t(sequence.shape[1] / 2, sequence.shape[2] / 2, sequence.shape[3] / 2)
    inv_tm_center = np.linalg.inv(tm_center)

    if axis == 'x':
        rot_imgs = np.ndarray((len(rot_angles), sequence.shape[0], sequence.shape[1], sequence.shape[3]))
    else:
        rot_imgs = np.ndarray((len(rot_angles), sequence.shape[0], sequence.shape[1], sequence.shape[2]))
    trf_mats = np.ndarray((len(rot_angles), 4, 4))

    for frame in range(sequence.shape[0]):

        print('Processing frame ' + str(frame + 1) + ' of ' + str(sequence.shape[0]) + '.')

        # rotate volume about z axis 
        for idx, angle in enumerate(rot_angles):
            theta = m.radians(angle % 360)
            vol_gpu = cp.array(sequence[frame][:][:][:])

            # extract image
            if axis == 'x':
                rot_img = slice_volume_x(vol_gpu, theta)
            elif axis == 'y':
                rot_img = slice_volume_y(vol_gpu, theta)
            elif axis == 'z':
                rot_img = slice_volume_z(vol_gpu, theta)

            # rot_imgs[idx, frame, :, :] = np.flipud(cp.asnumpy(rot_img))
            rot_imgs[idx, frame, :, :] = cp.asnumpy(rot_img)

            # free gpu mem
            del vol_gpu, rot_img

            # save transformation matrix for inverse transformation of landmarks
            if frame == 0:
                rm_z = rz(theta)
                trf_mats[idx, :, :] = np.linalg.multi_dot([tm_center, rm_z, inv_tm_center])

            # VisualizeVolume.display_rgb_image(rot_imgs[idx, frame, :, :], "extracted image after " + str(angle) +
            # " degrees rotation about z axis")

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return rot_imgs, trf_mats


def inverse_trf_z_rot(cors, trf_mats, sequence_sz):
    # number of rotations
    num_rot = cors.shape[0]
    num_frames = cors.shape[1]

    inv_cors = np.zeros((2 * num_rot, num_frames, 3))

    # for all rotations about x and y axis
    for rot_idx in range(num_rot):
        for frame_idx in range(num_frames):
            cor_left_3d = [cors[rot_idx][frame_idx][1], cors[rot_idx][frame_idx][0], sequence_sz[2] / 2, 1]
            cor_right_3d = [cors[rot_idx][frame_idx][3], cors[rot_idx][frame_idx][2], sequence_sz[2] / 2, 1]

            inv_trf = np.linalg.inv(trf_mats[rot_idx])

            inv_cor = inv_trf @ cor_left_3d
            inv_cors[rot_idx, frame_idx, :] = inv_cor[0:3]

            inv_cor = inv_trf @ cor_right_3d
            inv_cors[2 * num_rot - 1 - rot_idx, frame_idx, :] = inv_cor[0:3]

    return inv_cors


def rotate_about_long_axis(sequence, rot_angles, rot):
    print('Rotating volume estimated long axis')
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    rot_imgs = np.ndarray((len(rot_angles), sequence.shape[0], sequence.shape[1], sequence.shape[2]))
    trf_mats = np.ndarray((len(rot_angles), 4, 4))

    # translate to probe center, to rotate about centerline
    tm_center = t(0, sequence.shape[2] / 2, sequence.shape[3] / 2)
    inv_tm_center = np.linalg.inv(tm_center)

    alpha = np.radians(-(90-rot['y']))
    rm_y = ry(alpha)

    beta = np.radians((90-rot['x']))
    rm_x = rx(beta)

    # translate and rotate volume
    trf_mat = np.linalg.multi_dot([tm_center, rm_x, rm_y, inv_tm_center])
    trf_mat_gpu = cp.array(trf_mat)

    for frame in range(sequence.shape[0]):

        trf_vol = affine_transform(cp.array(sequence[frame][:][:][:]), trf_mat_gpu)

        for idx, angle in enumerate(rot_angles):
            theta = m.radians(angle)

            # extract image
            rot_img = slice_volume_z(trf_vol, theta)

            # save extracted image after rotation
            rot_imgs[idx, frame, :, :] = cp.asnumpy(rot_img)

            if frame == 0:
                rm_z = rz(theta)
                trf_mats[idx, :, :] = np.linalg.multi_dot([tm_center, rm_z, rm_x, rm_y, inv_tm_center])
                # trf_mats[idx, :, :] = np.linalg.multi_dot([tm_center, rm_z, inv_tm_center])

            # VisualizeVolume.display_rgb_image(cp.asnumpy(rot_img), "extracted image after " + str(angle) +
            # " degrees rotation about estimated long axis")

        # free gpu mem
        del trf_vol, rot_img

    del trf_mat_gpu
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return rot_imgs, trf_mats


def inverse_trf_long_axis_rot(cor_rots, trf_mats, sequence_sz, rot_angles):
    inv_cors = np.empty((360, sequence_sz[0], 3))
    inv_cors[:] = np.nan

    # for all rotations about long axis
    for idx, angle in enumerate(rot_angles):
        for frame_idx in range(sequence_sz[0]):
            cor_left_3d = [cor_rots[idx][frame_idx][1], cor_rots[idx][frame_idx][0], sequence_sz[2] / 2, 1]
            cor_right_3d = [cor_rots[idx][frame_idx][3], cor_rots[idx][frame_idx][2], sequence_sz[2] / 2, 1]

            inv_trf = np.linalg.inv(trf_mats[idx])
            
            inv_cor = inv_trf @ cor_right_3d
            inv_cors[angle, frame_idx, :] = inv_cor[0:3]

            inv_cor = inv_trf @ cor_left_3d
            left_angle = (angle + 180) % 360
            inv_cors[left_angle, frame_idx, :] = inv_cor[0:3]

    return inv_cors
