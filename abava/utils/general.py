# -*-coding:utf-8 -*-

import base64
import json
import math
import os
import cv2
from ..exception import *


def load_json(path):
    """
    Load json
    :param path: json path
    :return: dict
    """
    with open(path, 'r', encoding='utf-8') as f:
        load_dict = json.load(f)
    return load_dict


def str2dict(string):
    """
    String to dictionary
    :param string: String in dictionary form
    :return: dict
    """
    string = string.replace('true', 'True').replace('null', 'None').replace('false', 'False')
    data = string.encode("utf-8")
    str_url = base64.b64encode(data)
    out_dict = base64.b64decode(str_url).decode("utf-8")
    out_dict = eval(out_dict)
    return out_dict


def get_key(d, value):
    """
    Take the key according to the value
    :param d: dict
    :param value: value
    :return: key
    """
    k = [k for k, v in d.items() if v == value]
    return k


def chunks(obj, step):
    """
    Convert list shapes
    :param obj: list
    :param step: column
    :return:
    """
    cutListRes = [obj[i:i + step] for i in range(0, len(obj), step)]
    return cutListRes


def find_nth(string, substring, n):
    """
    Find the index of the nth occurrence of the specified character in the string
    :param string: String
    :param substring: Characters
    :param n: Number of times
    :return: index
    """
    if n == 1:
        return string.find(substring)
    else:
        return string.find(substring, find_nth(string, substring, n - 1) + 1)


def find_last(string, substring):
    """
    Index of the last occurrence of the specified character in the string
    :param string: String
    :param substring: Characters
    :return: index
    """
    for i in range(len(string)-1, -1, -1):
        if string[i] == substring:
            return i
        if i == 0:
            raise AbavaCommonException("Sorry! We haven't found the Search Character in this string ")


def s2bytes(data):
    if isinstance(data, str):
        return data.encode(encoding='utf-8')
    else:
        return data


def b2string(data):
    if isinstance(data, bytes):
        return data.decode('utf-8')
    else:
        return data


def cal_distance(point_a, point_b):
    """
    Calculating two-dimensional Euclidean distances
    :param point_a: [x, y]
    :param point_b: [x, y]
    :return: float distance
    """
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def cal_angle(point_a, point_b):
    """
    Calculate the angle of one side to the x-axis
    :param point_a: [x, y]
    :param point_b: [x, y]
    :return: angle [0, 360]
    """
    delta_x = point_b[0] - point_a[0]
    delta_y = point_b[1] - point_a[1]
    radian = math.atan2(delta_y, delta_x)
    degree = math.degrees(radian)
    if degree < 0:
        degree += 360
    return degree


def get_points(points):
    """
    This recursive method is used to process the deep points data returned by the mask tool.
    It is worth noting that in the case of inclusion, where there are two or more sets of points,
    the first set of points will be returned by default
    :param points:
    :return:
    """
    if isinstance(points, dict):
        if points.get('data'):
            if isinstance(points['data'], list) and isinstance(points['data'][0], int):
                return points['data']
            elif isinstance(points['data'], list) and isinstance(points['data'][0], dict):
                return get_points(points['data'])
            elif isinstance(points['data'], list) and isinstance(points['data'][0], list):
                return get_points(points['data'])
            elif isinstance(points['data'], dict):
                return get_points(points['data'])
        elif points.get('pathData'):
            if isinstance(points['pathData'], list) and isinstance(points['pathData'][0], int):
                return points['pathData']
            elif isinstance(points['pathData'], list) and isinstance(points['pathData'][0], dict):
                return get_points(points['pathData'])
            elif isinstance(points['pathData'], dict):
                return get_points(points['pathData'])
    elif isinstance(points, list):
        if isinstance(points[0], dict):
            return get_points(points[0])
        elif isinstance(points[0][0], int):
            return points


def get_coco_points(points):
    """
    Used with get_points in the mask tool export to get the coco bbox for xmin, ymin, xmax, ymax
    :param points:
    :return:
    """
    xl = sorted(points, key=lambda x: x[0])
    xmin, xmax = xl[0][0], xl[-1][0]
    yl = sorted(points, key=lambda x: x[1])
    ymin, ymax = yl[0][1], yl[-1][1]
    return [xmin, ymin, xmax, ymax]


def get_voc_points(points):
    """
    Used in conjunction with get_points in the mask tool export to generate multiple tag points
    :param points:
    :return:
    """
    from lxml import objectify
    E = objectify.ElementMaker(annotate=False)
    bnd_box = E.bndbox()
    for i, v in enumerate(points):
        bnd_box.append(E(f'x{i + 1}', v))
    return bnd_box


def cutFrames(infile_name, outfile_name, cut):
    """
    video frame extraction
    :param infile_name: video path
    :param outfile_name: output folder path
    :param cut: extracting a frame from several images
    :return:
    """
    videoCapture = cv2.VideoCapture(infile_name)
    i = 0
    j = 0
    while True:
        success, frame = videoCapture.read()
        i += 1
        if not success:
            print('video is all read')
            break
        if i == 1 or (i % cut == 0):
            print(i)
            j += 1
            saved_name = str(j).zfill(5) + '.jpg'
            cv2.imencode('.jpg', frame)[1].tofile(os.path.join(outfile_name, saved_name))
            print('image of %s is saved' % (os.path.join(outfile_name, saved_name)))
