# -*-coding:utf-8 -*-
import math
import re
import struct
import time

import cv2
import numpy as np
import os
import dask.dataframe as dd
from ..exception import AbavaParameterException, AbavaNotImplementException


def read_pcd(pcd_path):
    """
    read pcd file
    :param pcd_path:
    :return:
    """
    global pc_points
    headers_lines = 11
    try:
        with open(pcd_path, 'r') as f:
            header = [next(f) for _ in range(headers_lines)]
    except UnicodeDecodeError:
        with open(pcd_path, 'rb') as f:
            header = [next(f).decode('ISO-8859-1') for _ in range(headers_lines)]

    headers = {}
    pattern = re.compile(r'(VERSION|FIELDS|SIZE|TYPE|COUNT|WIDTH|HEIGHT|VIEWPOINT|POINTS|DATA)')
    for i, line in enumerate(header):
        match = pattern.match(line)
        if match:
            fields = line.strip().split()
            key = match.group()
            if key == 'DATA':
                headers[key] = fields[1]
                headers['data_start'] = i + 1
            elif key == 'POINTS':
                headers[key] = int(fields[1])
            else:
                headers[key] = fields[1:]

    type_size_map = {('U', '1'): np.uint8, ('U', '2'): np.uint16, ('U', '4'): np.uint32,
                     ('F', '4'): np.float32,
                     ('I', '1'): np.int8, ('I', '2'): np.int16, ('I', '4'): np.int32}

    num_fields = len(headers['FIELDS'])
    if headers['DATA'] == 'ascii':
        dtypes = {idx: type_size_map[(type, headers['SIZE'][idx])] for idx, type in enumerate(headers['TYPE'])}
        df = dd.read_csv(pcd_path, skiprows=headers['data_start'], sep=" ", header=None, assume_missing=True,
                         dtype=dtypes)
        pc_points = df.to_dask_array(lengths=True).reshape((-1, num_fields)).compute()

    elif headers['DATA'] == 'binary':
        with open(pcd_path, 'rb') as f:
            for _ in range(headers['data_start']):
                _ = f.readline()
            dtype_list = [(name, type_size_map[(field_type, size)]) for name, field_type, size in
                          zip(headers["FIELDS"], headers['TYPE'], headers["SIZE"])]
            dt = np.dtype(dtype_list)
            data = np.fromfile(f, dtype=dt)

        names = headers["FIELDS"]
        counter_dict = {}
        new_names = []
        for i, el in enumerate(names):
            if names.count(el) > 1:
                if el not in counter_dict:
                    counter_dict[el] = 1
                else:
                    counter_dict[el] += 1
                new_names.append(el + str(counter_dict[el]))
            else:
                new_names.append(el)

        for old_name, new_name in zip(names, new_names):
            data.dtype.names = [name.replace(old_name, new_name) for name in data.dtype.names]

        pc_points = np.zeros((headers['POINTS'], len(new_names)), dtype=np.float32)
        for i, name in enumerate(data.dtype.names):
            pc_points[:, i] = data[name]
    elif headers['DATA'] == 'binary_compressed':
        # TODO: binary_compressed
        raise AbavaNotImplementException('Temporarily unable to read binary_compressed data.')
    else:
        raise 'Unknown pcd data type.'

    headers.pop('data_start')
    return pc_points, headers


def write_pcd(points, out_path, head=None, data_mode='ascii'):
    """
    write pcd file
    :param points: 2-d np.array
    :param out_path:
    :param head: {
        "FIELDS": ["x", "y", "z", "intensity"],
        "SIZE": ["4", "4", "4", "4"],
        "TYPE": ["F", "F", "F", "F"],
        "COUNT": ["1", "1", "1", "1"] }
    :param data_mode: ascii, binary
    :return:
    """
    if head is None:
        head = {
            "FIELDS": ["x", "y", "z", "intensity"],
            "SIZE": ["4", "4", "4", "4"],
            "TYPE": ["F", "F", "F", "F"],
            "COUNT": ["1", "1", "1", "1"]
        }
    point_num = points.shape[0]

    header = f'# .PCD v0.7 - Point Cloud Data file format\n' \
             f'VERSION 0.7\n' \
             f'FIELDS {" ".join(head["FIELDS"])}\n' \
             f'SIZE {" ".join(head["SIZE"])}\n' \
             f'TYPE {" ".join(head["TYPE"])}\n' \
             f'COUNT {" ".join(head["COUNT"])}\n' \
             f'WIDTH {point_num}\n' \
             'HEIGHT 1\n' \
             'VIEWPOINT 0 0 0 1 0 0 0\n' \
             f'POINTS {point_num}\n' \
             f'DATA {data_mode}'
    type_map = {('U', '1'): 'B', ('U', '2'): 'H', ('U', '4'): 'I',
                ('F', '4'): 'f',
                ('I', '1'): 'c', ('I', '2'): 'h', ('I', '4'): 'i'}

    if data_mode == 'ascii':
        handle = open(out_path, 'w')
        handle.write(header)
        for point in points:
            str_points = [str(p) for p in point]
            string = '\n' + ' '.join(str_points)
            handle.write(string)
        handle.close()
    elif data_mode == 'binary':
        with open(out_path, 'wb') as handle:
            handle.write(header.encode())
            handle.write(b'\n')

            pack_string = ''.join([type_map[(type, head['SIZE'][idx])] for idx, type in enumerate(head['TYPE'])])
            points_string = []

            for point in points:
                binary_data = [
                    struct.pack(pack, float(point[idx])) if pack == 'f' else struct.pack(pack, int(point[idx])) for
                    idx, pack in enumerate(pack_string)]
                points_string.append(b''.join(binary_data))

            handle.write(b''.join(points_string))

    elif data_mode == 'binary_compressed':
        # TODO: binary_compressed
        raise AbavaNotImplementException('Temporarily unable to read binary_compressed data.')
    else:
        raise 'Unknown pcd data type.'


def pcd2bin(pcd_path, out_path):
    """
    pcd convert to bin
    :param fields: pcd FIELDS
    :param pcd_folder:
    :param bin_folder:
    :return:
    """
    data, headers = read_pcd(pcd_path)
    data.tofile(out_path)


def bin2pcd(bin_path, pcd_path, head=None):
    """
    Convert point cloud bin format to pcd format
    :param bin_path: bin file path
    :param pcd_folder: pcd folder path
    :param head: {
        "FIELDS": ["x", "y", "z", "intensity"],
        "SIZE": ["4", "4", "4", "4"],
        "TYPE": ["F", "F", "F", "F"],
        "COUNT": ["1", "1", "1", "1"] }
    :return: pcd file
    """
    try:
        if not (os.path.isdir(pcd_path)):
            os.makedirs(os.path.join(pcd_path))
    except OSError as e:
        raise

    if head is None:
        head = {
            "FIELDS": ["x", "y", "z", "intensity"],
            "SIZE": ["4", "4", "4", "4"],
            "TYPE": ["F", "F", "F", "F"],
            "COUNT": ["1", "1", "1", "1"]
        }

    print("Converting Start!")
    points = np.fromfile(bin_path, dtype="float32").reshape((-1, len(head['FIELDS'])))
    write_pcd(points, pcd_path, head)


def pcd_ascii2binary(input_file, output_file):
    start = time.time()
    point_data, headers = read_pcd(input_file)
    end = time.time()
    print('read time:', end - start)
    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(point_data, output_file, head, data_mode='binary')

    print('Conversion complete!')


def pcd_binary2ascii(input_file, output_file):
    point_data, headers = read_pcd(input_file)
    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(point_data, output_file, head, data_mode='ascii')

    print('Conversion complete!')


def filter_points_in_boxes(pcd_file, boxes_list):
    """
    Given point cloud and a list of 3D boxes, remove the points inside the boxes.

    @param point_cloud: (N, 3) numpy.ndarray, N points.
    @param boxes_list: list of boxes with format [x, y, z, roll, pitch, yaw, length, width, height].
    """
    point_cloud, headers = read_pcd(pcd_file)
    try:
        intensity_point = point_cloud[:, 3]
    except:
        intensity_point = np.zeros(len(point_cloud))
    xyz_point = point_cloud[:, :3]
    for box in boxes_list:
        box_center = np.array(box[:3])
        rot = euler_to_rotation_matrix(box[3:6])
        box_size = np.array(box[6:])

        RT = np.eye(4)
        RT[:3, :3] = rot
        RT[:3, 3] = box_center

        box_pcd_np = np.linalg.inv(RT).dot(
            np.concatenate([xyz_point, np.ones((xyz_point.shape[0], 1))], axis=-1).T).T[:, :3]
        mask = np.all((box_pcd_np >= -0.5 * box_size) & (box_pcd_np <= 0.5 * box_size), axis=1)
        xyz_point = xyz_point[~mask]
        intensity_point = intensity_point[~mask]
    filtered_point = np.concatenate([xyz_point, intensity_point.reshape(-1, 1)], axis=1)
    return filtered_point


def voxel_subsample_keep_intensity(pcd_path, intensity, voxel_size, output_path='./subsampled.pcd'):
    """
    Retain points within the intensity information threshold and downsample the remaining points based on voxels
    :param pcd_path: pcd format point cloud path
    :param intensity: intensity range (list) example: [20, 200]
    :param voxel_size: voxel size
    :param output_path: Point cloud file save path
    :return:
    """
    pc_points, headers = read_pcd(pcd_path)[:, :4]
    # We default the first four columns of the point cloud to x, y, z, intensity
    points_intensity = pc_points[(pc_points[:, 3] >= intensity[0]) & (pc_points[:, 3] <= intensity[1])]
    points_other = pc_points[(pc_points[:, 3] < intensity[0]) | (pc_points[:, 3] > intensity[1])]
    voxel_coords = np.floor(points_other[:, 0:3] / voxel_size).astype(np.int32)
    voxel_indices = np.unique(voxel_coords, axis=0, return_index=True)[1]
    downsampled_points_other = points_other[voxel_indices]
    final_points = np.concatenate((points_intensity, downsampled_points_other), axis=0)
    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(final_points, output_path, head)


def random_subsample_keep_intensity(pcd_path, intensity, sampling_ratio, output_path='./subsampled.pcd'):
    """
    Retain points within the intensity information threshold and randomly downsample the remaining points
    :param pcd_path: pcd format point cloud path
    :param intensity: intensity range (list) example: [20, 200]
    :param sampling_ratio: downsampling rate
    :param output_path: Point cloud file save path
    :return:
    """
    pc_points, headers = read_pcd(pcd_path)[:, :4]
    points_intensity_20_200 = pc_points[(pc_points[:, 3] >= intensity[0]) & (pc_points[:, 3] <= intensity[1])]
    points_other = pc_points[(pc_points[:, 3] < intensity[0]) | (pc_points[:, 3] > intensity[1])]
    num_points = points_other.shape[0]
    num_sampled_points = int(sampling_ratio * num_points)
    sampled_indices = np.random.choice(num_points, num_sampled_points, replace=False)
    downsampled_points_other = points_other[sampled_indices]
    final_points = np.concatenate((points_intensity_20_200, downsampled_points_other), axis=0)
    head = {
        "FIELDS": headers['FIELDS'],
        "SIZE": headers['SIZE'],
        "TYPE": headers['TYPE'],
        "COUNT": headers['COUNT']
    }
    write_pcd(final_points, output_path, head)


def pnp_compute_Rt(obj_points, img_points, intrinsic, distortion):
    """
    Solve for the rotation matrix R and the translation vector t
    :param obj_points: 2-d list [[x1, y1, z1], [x2, y2. z2], [x3, y3, z3]]
    :param img_points: 2-d list [[x1, y1], [x2, y2], [x3, y3]]
    :return: R, t
    """
    objectPoints = np.array(obj_points, dtype="double")
    imagePoints = np.array(img_points, dtype="double")

    cameraMatrix = np.array([[intrinsic[0], 0, intrinsic[2]],
                             [0, intrinsic[1], intrinsic[3]],
                             [0, 0, 1]], dtype="double")
    distCoeffs = np.array(distortion, dtype="double")

    _, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)

    # Print the rotation and translation vectors
    return rvec, tvec


def compute_points_in_box(bin_path, fields_len, box):
    """
    Calculate the number of points in the 3D box
    :param fields_len:
    :param bin_path:
    :param box: [x, y, z, l, w, h, r, p, y]
    :return: int
    """
    point_cloud = np.fromfile(bin_path, dtype=np.float32)
    point_cloud = point_cloud.reshape(-1, fields_len)

    px, py, pz = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    cx, cy, cz, l, w, h, r, p, y = box

    inv_rotation = euler_to_rotation_matrix([float(r), float(p), float(y)]).T
    points_centered = np.array([px - cx, py - cy, pz - cz]).T
    points_rotated = np.dot(points_centered, inv_rotation)

    xmin, ymin, zmin = -l / 2, -w / 2, -h / 2
    xmax, ymax, zmax = l / 2, w / 2, h / 2
    inside_x = (points_rotated[:, 0] > xmin) & (points_rotated[:, 0] < xmax)
    inside_y = (points_rotated[:, 1] > ymin) & (points_rotated[:, 1] < ymax)
    inside_z = (points_rotated[:, 2] > zmin) & (points_rotated[:, 2] < zmax)
    inside_box = inside_x & inside_y & inside_z
    num_points = np.count_nonzero(inside_box)

    return num_points


def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion
    :param R: 3X3 rotation matrix
    :return:
    """
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return [qx, qy, qz, qw]


def rotation_matrix_to_euler(R):
    """
    Convert rotation matrix to euler angles
    :param R: 3X3 rotation matrix
    :return:
    """
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    r = math.atan2(R[2, 1], R[2, 2])
    p = math.atan2(-R[2, 0], sy)
    y = math.atan2(R[1, 0], R[0, 0])
    return [r, p, y]


def euler_to_rotation_matrix(euler):
    """
    Convert euler angles to rotation matrix
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    roll, pitch, yaw = euler
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix
    :param q:
    :return:
    """
    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < np.finfo(float).eps:
        return np.identity(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array([
        [1.0 - (yY + zZ), xY - wZ, xZ + wY],
        [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
        [xZ - wY, yZ + wX, 1.0 - (xX + yY)]
    ])


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert euler angles to quaternion
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    return [qx, qy, qz, qw]


def quaternion_to_euler(x, y, z, w):
    """
    Convert quaternion to euler angles
    :param x:
    :param y:
    :param z:
    :param w:
    :return:
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def box2corners(points):
    """
    Convert 3d box (7 numbers) to 8 corners (21 numbers)
    :param points: x, y, z, roll, pitch, yaw, length, width, height
    :return:
    """
    x, y, z, roll, pitch, yaw, length, width, height = points

    corners = np.array([
        [-length / 2, -width / 2, -height / 2],
        [length / 2, -width / 2, -height / 2],
        [length / 2, width / 2, -height / 2],
        [-length / 2, width / 2, -height / 2],
        [-length / 2, -width / 2, height / 2],
        [length / 2, -width / 2, height / 2],
        [length / 2, width / 2, height / 2],
        [-length / 2, width / 2, height / 2]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    R = np.dot(Rz, np.dot(Ry, Rx))

    corners = np.dot(corners, R.T)
    corners += np.array([x, y, z])

    return corners
