#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm
import urllib.parse

from ..export import ExportData
from .labelme import LABELME
from ...utils import judge_contain, calculate_polygon_area, image2base64, check_clockwise


class ExportLabelme(ExportData):
    def __init__(self, source_data, out_path=None, mapping=None):
        """
        Converting abava data format to labelme data format
        :param source_data: source abava data
        :param out_path:
        :param mapping: label-color mapping
        """
        super(ExportLabelme, self).__init__(source_data, out_path, mapping)
        label_obj_list, version, label_type, is_ori_tag = self.match_label_data()
        self.label_obj_list = label_obj_list
        self.version = version
        self.label_type = label_type
        self.is_ori_tag = is_ori_tag

    def abava_json2labelme(self):
        """
        Converting ABAVA raw data to labelme format
        :return: json file in labelme format
        """

        global annotation_tag

        for data in tqdm(self.label_obj_list):
            abava_url = data['info']
            image_name = Path(abava_url).parts[-1]
            LABELME.shapes = []
            labels = data['labels']
            if len(labels) == 0:
                continue
            labels = sorted(labels,
                            key=lambda x: x['data']['zIndex'] if 'zIndex' in self.abava2dict(x['data']) else x['data'][
                                'id'])
            for l in labels:
                drawType = l.data.drawType
                labelme_shape = self._query("labelme_shape", data)
                if drawType == 'RECTANGLE':
                    points = np.array(l.data.points)
                    label = l.data.label
                    if self.mapping:
                        label = self.mapping[label]
                    points_labelme = points.reshape((-1, 2))
                    labelme_shape.label = label
                    labelme_shape.points = points_labelme.tolist()
                    labelme_shape.group_id = None
                    labelme_shape.shape_type = 'rectangle'
                    labelme_shape.flags = {}

                    LABELME.shapes.append(labelme_shape)
                elif drawType == 'POLYGON':
                    points = np.array(l.data.points)
                    label = l.data.label
                    if self.mapping:
                        label = self.mapping[label]
                    points_labelme = points.reshape((-1, 2))
                    labelme_shape.label = label
                    labelme_shape.points = points_labelme.tolist()
                    labelme_shape.group_id = None
                    labelme_shape.shape_type = 'polygon'
                    labelme_shape.flags = {}

                    LABELME.shapes.append(labelme_shape)
                elif drawType == 'MASK' or drawType == 'MASK_BLOCK':
                    polygon_list = []
                    label = l.data.label
                    if self.mapping:
                        label = self.mapping[label]
                    polygonDict, tag = judge_contain(l.data.points)
                    if len(polygonDict) == 0:
                        for p in l.data.points:
                            polygon_list.append({
                                'label': label,
                                'points': np.array(p).reshape(-1, 2).tolist(),
                                'area': calculate_polygon_area(np.array(p).reshape(-1, 2).tolist())
                            })
                    else:
                        polygon_temp_list = []
                        for _, polygon in enumerate(polygonDict):
                            polygon_temp_list.append(polygonDict[polygon])
                            if 'outer' in polygon:
                                annotation_tag = polygon
                                polygon_list.append({
                                    'label': label,
                                    'points': polygonDict[polygon],
                                    'area': calculate_polygon_area(polygonDict[polygon])
                                })
                            elif 'inner' in polygon:
                                if check_clockwise(polygonDict[polygon]) == check_clockwise(polygonDict[annotation_tag]):
                                    polygon_list.append({
                                        'label': label,
                                        'points': polygonDict[polygon],
                                        'area': calculate_polygon_area(polygonDict[polygon])
                                    })
                        for points in l.data.points:
                            if points not in polygon_temp_list:
                                polygon_list.append({
                                    'label': label,
                                    'points': np.array(points).reshape(-1, 2).tolist(),
                                    'area': calculate_polygon_area(np.array(points).reshape(-1, 2).tolist())
                                })
                    sorted_polygon = sorted(polygon_list, key=lambda x: x['area'])
                    sorted_polygon.reverse()
                    for polygon in sorted_polygon:
                        labelme_shape = self._query("labelme_shape", data)
                        labelme_shape.label = polygon['label']
                        labelme_shape.points = polygon['points']
                        labelme_shape.group_id = None
                        labelme_shape.shape_type = 'polygon'
                        labelme_shape.flags = {}
                        LABELME.shapes.append(labelme_shape)
                else:
                    warnings.warn(
                        f"This method does not currently support exporting the {drawType} label type",
                        UserWarning)
            height = data['size']['height']
            width = data['size']['width']
            imageData = image2base64(abava_url)

            LABELME.version = '5.1.1'
            LABELME.flags = {}
            LABELME.imagePath = './' + image_name
            LABELME.imageData = imageData
            LABELME.imageHeight = height
            LABELME.imageWidth = width

            if '/'.join(Path(abava_url).parts[4:-1]) is not None:
                root = '/'.join(Path(abava_url).parts[4:-1])
            else:
                root = ''
            root = urllib.parse.unquote(root)

            file_name = image_name.replace(image_name.split('.')[-1], 'json')
            self.save_labels(self.abava2dict(LABELME), file_name, root)
