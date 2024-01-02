# -*-coding:utf-8 -*-

import base64
import os
import random
import urllib.request
from collections import namedtuple
from pprint import pformat
import cv2
import numpy as np
from PIL import Image
from pathlib import Path


def image2base64(url):
    """
    image to base64
    :param url: image url
    :return: base64
    """
    img = read_url_image(url)
    binary_str = cv2.imencode('.jpg', img)[1].tobytes()
    base64_str = base64.b64encode(binary_str)
    base64_str = base64_str.decode('utf-8')

    return base64_str


def base642image(base64_str):
    """
    base64 to image
    :param base64_str: image base64
    :return: image
    """
    imgdata = base64.b64decode(base64_str)
    np_array = np.frombuffer(imgdata, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img


def read_url_image(url):
    """
    Read url images
    :param url: image url
    :return: image
    """
    res = urllib.request.urlopen(url)
    img = np.asarray(bytearray(res.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    return img


def get_urlimage_size(url):
    """
    Read url image to get width and height
    :param url: image url
    :return: size: height, width
    """
    img = read_url_image(url)
    height = img.shape[0]
    width = img.shape[1]

    return height, width


def hex_to_rgb(hex_color):
    """
    hex to rgb
    :param hex_color: #ffffff
    :return:
    """
    hex_color = hex_color.lstrip('#')  # 删除可能存在的开头的'#'
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    return red, green, blue


def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)


def draw_rectangle(xyxy, img, label=None, color=None, line_thickness=None):
    """
    Plots one bounding box on image img
    :param xyxy: [xmin, ymin, xmax, ymax]
    :param img: image
    :param color: (b, g, r)
    :param label: label name
    :param line_thickness: thickness, int
    :return: image
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_point(xy, img, category_name, colour, line_thickness=None):
    """
    Drawing key points
    :param img: image
    :param category_name: category name
    :param xy: Key point x and y [x, y]
    :param colour: Key point colour
    :param line_thickness: thickness, int
    :return: image
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    tf = max(tl - 1, 1)
    cv2.circle(img, (xy[0], xy[1]), 3, colour, -1)
    cv2.putText(img, category_name, (xy[0], xy[1]-3), 0, tl / (img.shape[0] % 10) / 4, colour, thickness=tf, lineType=cv2.LINE_AA)
    return img


def draw_polygon(points, img, colour):
    """
    Drawing key points
    :param img: image
    :param category_name: category name
    :param xy: Key point x and y [x, y]
    :param colour: Key point colour
    :param line_thickness: thickness, int
    :return: image
    """
    mask = np.zeros(img.shape, dtype=np.uint8)
    np_paths = []
    for polygon in points:
        np_paths.append(np.array(polygon, np.int32).reshape(-1, 2))
    cv2.fillPoly(mask, np_paths, color=colour)
    result = cv2.addWeighted(img, 1, mask, 0.5, 0)
    return result


def __create_psub_masks(mask_image):
    """
    Private, cut molecular mask according to mask colour
    :param mask_image: mask image
    :return: dict
    """
    width, height = mask_image.size
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    pixels = np.array(list(mask_image.getdata())).reshape(height, width)
    pixels_list = np.unique(pixels)
    for cat_id in pixels_list[1:]:
        mask = np.zeros(pixels.shape)
        mask[pixels==cat_id] = cat_id
        sub_masks[cat_id] = mask
    return sub_masks


def __create_sub_mask_annotation(sub_mask, file):
    """
    Private, get polygon of submask, generate annotation dictionary
    :param sub_mask: sub mask
    :param file: file name
    """
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    # contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')
    def recursive(sub_counters, ind):

        if sub_counters[ind][2] == -1:
            return counters_group
        else:
            counters_group.append(sub_counters[ind][2])
            recursive(sub_counters, sub_counters[ind][2])

    sub_mask = np.where(sub_mask != 0, 1, sub_mask)
    sub_mask = sub_mask.astype(np.uint8)

    contours, hierarchy = cv2.findContours(sub_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    data = []
    tag = []

    for idx, contour in enumerate(contours):
        if idx in tag:
            continue
        counters_group = [idx]
        recursive(hierarchy[0], idx)
        tag += counters_group
        for i, counter_idx in enumerate(counters_group):
            if i // 2 == 1:
                data.append({
                    'type': 'Path',
                    'data': contours[counter_idx][::-1].ravel().tolist()
                })
            else:
                data.append({
                    'type': 'Path',
                    'data': contours[counter_idx].ravel().tolist()
                })

    annotation = {
        "_id": file,
        "data": data
    }

    return annotation


# mask2polygon
def pmask2polygons(mask_path):
    """
    Converting p-mode masks(4-bit) to polygons
    :param mask_path: image folder path
    :return: dict
    """
    annotations = []
    mask_image = Image.open(mask_path)
    sub_masks = __create_psub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        annotation = __create_sub_mask_annotation(sub_mask, str(color))
        annotations.append(annotation)
    return annotations


def mask2rle(img):
    """
    mask to rle
    :param img: image
    :return: rle str
    """
    pixels = img.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape):
    """
    Decode data in RLE format into image data
    :mask_rle: RLE-encoded data
    :shape: (height, width)image height and width
    :return: Image data after decoding， type -> np.ndarray
    """
    rows, cols = shape[0], shape[1]
    rle_array = np.array([int(num_string) for num_string in mask_rle.split(' ')])
    rle_pairs = rle_array.reshape(-1, 2)
    img = np.zeros(rows * cols, dtype=np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index:index + length] = 255
    img = img.reshape(cols, rows)
    return img.T


def polygon2mask(polygon, width, height, color, out_path):
    """
    polygon to mask
    :param polygons: polygons one-dimensional arrays
    :param width: Image width
    :param height: Photo Height
    :param out_path: Output path
    :return:
    """
    mask = np.zeros((height, width), np.uint8)
    area = np.array(polygon, np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [area], color)

    cv2.imencode('.png', mask)[1].tofile(out_path)


def isin_external_rectangle(point, vertex_lst: list, contain_boundary=True):
    """
    Detects if the point is within the outer rectangle of the area
    :param point: detected point
    :param vertex_lst: polygon point list
    :param contain_boundary: Does it contain a boundary
    :return: boolean
    """
    lngaxis, lataxis = zip(*vertex_lst)
    minlng, maxlng = min(lngaxis), max(lngaxis)
    minlat, maxlat = min(lataxis), max(lataxis)
    lng, lat = point
    if contain_boundary:
        isin = (minlng <= lng <= maxlng) & (minlat <= lat <= maxlat)
    else:
        isin = (minlng < lng < maxlng) & (minlat < lat < maxlat)
    return isin


def __isintersect(poi, spoi, epoi):
    """
    The ray method to find out if a point is inside a polygon
    :param spoi: detected points
    :param epoi: All points of polygon
    :return: boolean
    """
    # 射线为向东的纬线
    # 可能存在的bug，当区域横跨本初子午线或180度经线的时候可能有问题
    lng, lat = poi
    slng, slat = spoi
    elng, elat = epoi
    if poi == spoi:
        # print("在顶点上")
        return None
    if slat == elat:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if slat > lat and elat > lat:  # 线段在射线上边
        return False
    if slat < lat and elat < lat:  # 线段在射线下边
        return False
    if slat == lat and elat > lat:  # 交点为下端点，对应spoint
        return False
    if elat == lat and slat > lat:  # 交点为下端点，对应epoint
        return False
    if slng < lng and elat < lat:  # 线段在射线左边
        return False
    # 求交点
    xseg = elng - (elng - slng) * (elat - lat) / (elat - slat)
    if xseg == lng:
        # print("点在多边形的边上")
        return None
    if xseg < lng:  # 交点在射线起点的左侧
        return False
    return True  # 排除上述情况之后


def isin_multipolygon(point, vertex_lst: list, contain_boundary=True):
    """
    Determine if a point is inside a polygon
    :param point: detected point
    :param vertex_lst: 2-d polygon point list
    :param contain_boundary: Does it contain a boundary
    :return: boolean
    """
    # 判断是否在外包矩形内，如果不在，直接返回false
    if not isin_external_rectangle(point, vertex_lst, contain_boundary):
        return False
    sinsc = 0
    for spoi, epoi in zip(vertex_lst[:-1], vertex_lst[1::]):
        intersect = __isintersect(point, spoi, epoi)
        if intersect is None:
            return (False, True)[contain_boundary]
        elif intersect:
            sinsc += 1
    return sinsc % 2 == 1


class Node(namedtuple('Node', 'location left_child right_child')):
    def __repr__(self):
        return pformat(tuple(self))


class KDTree():
    def __init__(self, points):
        self.points = points
        self.tree = self._make_kdtree(points)
        if len(points) > 0:
            self.k = len(points[0])
        else:
            self.k = None

    def _make_kdtree(self, points, depth=0):
        if not points:
            return None

        median = len(points) // 2

        return Node(
            location=points[median],
            left_child=self._make_kdtree(points[:median], depth + 1),
            right_child=self._make_kdtree(points[median + 1:], depth + 1))

    def find_nearest(self,
                     point,
                     root=None,
                     axis=0,
                     dist_func=lambda x, y: np.linalg.norm(x - y)):
        points_list = self.points
        if root is None:
            root = self.tree
            self._best = None

        # If not a leaf node, continue down
        if root.left_child or root.right_child:
            new_axis = (axis + 1) % self.k
            if point[axis] < root.location[axis] and root.left_child:
                self.find_nearest(point, root.left_child, new_axis)
            elif root.right_child:
                self.find_nearest(point, root.right_child, new_axis)

        # Trackback: trying to update best
        dist = dist_func(root.location, point)
        if self._best is None or dist < self._best[0]:
            self._best = (dist, root.location, points_list.index(root.location))

        # If the hypersphere intersects the other side of the hyperrectangle
        if abs(point[axis] - root.location[axis]) < self._best[0]:
            new_axis = (axis + 1) % self.k
            if root.left_child and point[axis] >= root.location[axis]:
                self.find_nearest(point, root.left_child, new_axis)
            elif root.right_child and point[axis] < root.location[axis]:
                self.find_nearest(point, root.right_child, new_axis)

        return self._best


def calculate_polygon_area(points_list):
    """
    calculate polygon area
    :param points_list: polygon point list
    :return:
    """
    n = len(points_list)
    if n < 3:
        return 0.0
    area = 0
    for i in range(n):
        x = points_list[i][0]
        y = points_list[i][1]
        area += x * points_list[(i + 1) % n][1] - y * points_list[(i + 1) % n][0]
    return area * 0.5


def judge_contain(data):
    """
    Determining polygon inclusion relationships, major label and minor label
    :param data: json data
    :return: dict
    """
    polygonDict = {}
    polygonList = []
    contain_tag = []

    for i in range(len(data)):
        polygonList.append(np.array(data[i]).reshape(-1, 2).tolist())
    # Get all polygons of a single colour
    count_annotation = 0
    for m in range(len(polygonList)):
        # Large labels [[],[],[],[],[],[]]
        tempa = polygonList[m]
        for n in range(len(polygonList)):
            # Small labels
            tempb = polygonList[n]
            if tempb == tempa:
                continue
            else:
                count = 0
                for idx in range(len(tempb)):
                    if isin_multipolygon(tempb[idx], tempa, contain_boundary=True):
                        count += 1
                if count == len(tempb):
                    polygonDict['annotations' + str(count_annotation) + ',outer' + str(m)] = tempa
                    polygonDict['annotations' + str(count_annotation) + ',inner' + str(n)] = tempb
                    if m not in contain_tag:
                        contain_tag.append(m)
                    if n not in contain_tag:
                        contain_tag.append(n)
        count_annotation += 1
    return polygonDict, contain_tag


def __get_rand_point(kPoints):
    """
    Take a random point in the polygon
    :param kPoints:list
    :return: point[list], index[int]
    """
    indx = random.randrange(0, len(kPoints), 2)
    kPoint = kPoints[indx]

    return kPoint, indx


def __findShortest(kpoint, pointlist):
    """
    Find the closest point
    :param kpoint: point[list]
    :param pointlist: points[[list]]
    :return: point[list], index[int]
    """
    minr = float("inf")
    spoint = [0, 0]
    markm = 0
    for m in range(len(pointlist)):
        r = pow(pow(pointlist[m][0] - kpoint[0], 2) + pow(pointlist[m][1] - kpoint[1], 2), 0.5)
        if r < minr:
            minr = r
            spoint[0] = pointlist[m][0]
            spoint[1] = pointlist[m][1]
            markm = m

    return spoint, markm


def __insertPoints(outerpoints, innerpoints, outerindex, innerindex):
    """
    Insert points to get a new polygon
    :param outerpoints: External polygons
    :param innerpoints: Internal polygons
    :param outerindex: Index of the start point of the external polygon
    :param innerindex: Index of the start point of the internal polygon
    :return: points[list]
    """
    new_points = []

    for i in range(outerindex + 1):
        new_points.append(outerpoints[i])
    for j in range(innerindex, len(innerpoints)):
        new_points.append(innerpoints[j])
    for p in range(innerindex + 1):
        new_points.append(innerpoints[p])
    for q in range(outerindex, len(outerpoints)):
        new_points.append(outerpoints[q])

    return new_points


def __multiInsertPoints(dict, outerpoints, keys_list, n):
    """
    Used when a polygon contains multiple polygons inside it
    :param dict: function <judge> output dict
    :param outerpoints: External polygon
    :param keys_list: dict key
    :param n: Number of internal polygons
    :return: points[list]
    """
    global input_inner
    if n == 0:
        return outerpoints
    else:
        innerpoints = []
        for i in range(len(keys_list)):
            innerpoints += dict[keys_list[i]]
        point, indx = __get_rand_point(outerpoints)
        kd_points = innerpoints
        kdtree = KDTree(kd_points)
        shortest = kdtree.find_nearest(np.array(point))
        dist, spoint, markm = shortest[0], shortest[1], shortest[2]
        for q in range(len(keys_list)):
            if spoint in dict[keys_list[q]]:
                input_inner = dict[keys_list[q]]
                keys_list.remove(keys_list[q])
                break
        start = innerpoints.index(input_inner[0])
        end = innerpoints.index(input_inner[-1], start + 1)
        del innerpoints[start:end + 1]
        mark = input_inner.index(spoint)
        output = __insertPoints(outerpoints, input_inner, indx, mark)
        return __multiInsertPoints(dict, output, keys_list, n - 1)


def polygonSkeleton(data):
    """
    Get the polygon after skeletonisation
    :param data: mask folder path
    :return: points[list]
    """
    key_tag = []
    keys_list = []
    total_list = []
    poly_dict, tag = judge_contain(data)
    if len(poly_dict) == 0:
        return 'Not Contain', []
    else:
        for k in poly_dict.keys():
            if k.split(',')[0] not in key_tag:
                key_tag.append(k.split(',')[0])
                keys_list.append([k])
            else:
                key_indx = key_tag.index(k.split(',')[0])
                keys_list[key_indx].append(k)
        for i in range(len(keys_list)):
            result = []
            outer = keys_list[i][0]
            point, indx = __get_rand_point(poly_dict[outer])

            keys_list[i].remove(outer)
            if len(keys_list[i]) == 1:
                inner = keys_list[i][0]
                kdtree = KDTree(poly_dict[inner])
                shortest = kdtree.find_nearest(np.array(point))
                dist, spoint, markm = shortest[0], shortest[1], shortest[2]
                points = __insertPoints(poly_dict[outer], poly_dict[inner], indx, markm)
                for i in range(len(points)):
                    for j in range(len(points[i])):
                        result.append(points[i][j])
                total_list.append(result)
            else:
                points = __multiInsertPoints(poly_dict, poly_dict[outer], keys_list[i], len(keys_list[i]))
                for i in range(len(points)):
                    for j in range(len(points[i])):
                        result.append(points[i][j])
                total_list.append(result)
    return total_list, tag


def check_clockwise(points):
    """
    Determine whether a polygon is sorted clockwise or counterclockwise.
    :param points: 2-d list [[x,y], [x,y], [x,y]]
    :return:
    """
    s = 0
    n = len(points)
    for i in range(n):
        point = points[i]
        point2 = points[(i+1)%n]
        s += (point2[0] - point[0]) * (point2[1] + point[1])
    return s > 0


def undistortSingleImage(image_path, camera_intrinstic, camera_dist):
    input_image = cv2.imread(image_path)
    img_size = (input_image.shape[1], input_image.shape[0])
    map1, map2 = cv2.initUndistortRectifyMap(camera_intrinstic, camera_dist, None, camera_intrinstic, img_size, cv2.CV_32FC1)
    undistort_image = cv2.remap(input_image, map1, map2, cv2.INTER_LINEAR)
    return undistort_image


def undistortSingleImagefisheye(image_path, camera_intrinstic, camera_dist):
    input_image = cv2.imread(image_path)
    img_size = (input_image.shape[1], input_image.shape[0])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_intrinstic, camera_dist, np.eye(3), camera_intrinstic,
                                                     img_size, cv2.CV_32FC1)
    undistort_image = cv2.remap(input_image, map1, map2, cv2.INTER_LINEAR)
    return undistort_image


def undistort(image_path, camera_intrinstic, camera_dist, output_path):
    """
    Image Removal Distortion
    :param image_path:
    :param camera_intrinstic: [fx,fy,cx,cy]
    :param camera_dist: distortion parameter
    :param output_path:
    :return:
    """
    fx, fy, cx, cy = camera_intrinstic

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]

    if len(camera_dist) == 8:
        undistort_image = undistortSingleImage(image_path, np.array(K), np.array(camera_dist))
    elif len(camera_dist) == 5:
        undistort_image = undistortSingleImagefisheye(image_path, np.array(K), np.array(camera_dist))
    else:
        raise "Camera distortion factor is wrong, please make sure that the camera distortion factor is entered " \
              "correctly"
    save_path = os.path.join(output_path, Path(image_path).parts[-2])
    save_name = Path(image_path).parts[-1]
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, save_name), undistort_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
