#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm
from datetime import datetime
from ..export import ExportData
from .coco import COCO
from ...utils import calculate_polygon_area


class ExportCoco(ExportData):
    def __init__(self, source_data, out_path=None, mapping=None):
        """
        Converting abava data format to coco data format
        :param source_data: source abava data
        :param out_path:
        :param mapping: label-color mapping
        """
        super(ExportCoco, self).__init__(source_data, out_path, mapping)
        label_obj_list, version, label_type, is_ori_tag = self.match_label_data()
        self.label_obj_list = label_obj_list
        self.version = version
        self.label_type = label_type
        self.is_ori_tag = is_ori_tag

    def abava_json2coco(self):
        """
        Converting ABAVA raw data to coco format
        :return: json file in coco format
        """
        labels = []
        image_id = 1
        id = 1

        # category
        for i in self.source_data.task.setting.labelConfig:
            coco_category = self._query("coco_category", i)
            label = i.label
            if self.mapping:
                label = self.mapping[label]
            labels.append(label)
            coco_category.supercategory = ""
            coco_category.id = labels.index(label)
            coco_category.name = label
            COCO.categories.append(coco_category)

        for data in tqdm(self.label_obj_list):
            abava_url = data['info']
            file_name = Path(abava_url).parts[-1]
            height = data['size']['height']
            width = data['size']['width']

            coco_image = self._query("coco_image", data)
            coco_image.file_name = file_name
            coco_image.abava_url = abava_url
            coco_image.height = height
            coco_image.width = width
            coco_image.date_captured = None
            coco_image.id = image_id

            COCO.images.append(coco_image)
            if len(data['labels']) == 0:
                continue
            for l in data['labels']:
                coco_annotation = self._query("coco_annotation", image_id)
                label = l.data.label
                if self.mapping:
                    label = self.mapping[label]
                segmentation = []
                drawType = l.data.drawType
                if drawType == 'RECTANGLE':
                    points = l.data.points
                    # left top
                    segmentation.append(round(points[0], 2))
                    segmentation.append(round(points[1], 2))
                    # right top
                    segmentation.append(round(points[2], 2))
                    segmentation.append(round(points[1], 2))
                    # right bottom
                    segmentation.append(round(points[2], 2))
                    segmentation.append(round(points[3], 2))
                    # left bottom
                    segmentation.append(round(points[0], 2))
                    segmentation.append(round(points[3], 2))

                    bbox = [round(points[0], 2),
                            round(points[1], 2),
                            round((points[2] - points[0]), 2),
                            round((points[3] - points[1]), 2)]
                    area = bbox[2] * bbox[3]

                    coco_annotation.segmentation = [segmentation]
                    coco_annotation.area = area
                    coco_annotation.iscrowd = 0
                    coco_annotation.bbox = bbox
                    coco_annotation.category_id = labels.index(label)
                    coco_annotation.id = id
                    coco_annotation.image_id = image_id

                    COCO.annotations.append(coco_annotation)
                    id += 1
                elif drawType == 'POLYGON':
                    points = np.array(l.data.points)
                    point_coco = points.reshape((-1, 2))
                    x = point_coco[:, 0]
                    y = point_coco[:, 1]
                    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

                    coco_annotation.segmentation = [np.round(points, 2).tolist()]
                    coco_annotation.area = area
                    coco_annotation.iscrowd = 0
                    coco_annotation.image_id = image_id
                    coco_annotation.bbox = [float(round(min(x), 2)),
                                            float(round(min(y), 2)),
                                            (float(round(max(x), 2)) - float(round(min(x), 2))) + 1,
                                            (float(round(max(y), 2)) - float(round(min(y), 2))) + 1]
                    coco_annotation.category_id = labels.index(label)
                    coco_annotation.id = id

                    COCO.annotations.append(coco_annotation)
                    id += 1
                elif drawType == 'MASK' or drawType == 'MASK_BLOCK':
                    points = l.data.points
                    area = 0
                    min_x = float('inf')
                    min_y = float('inf')
                    max_x = float('-inf')
                    max_y = float('-inf')
                    for idx, point_list in enumerate(points):
                        point_list = np.array(point_list).reshape(-1, 2)
                        if min(point_list[:, 0]) < min_x:
                            min_x = min(point_list[:, 0])
                        if min(point_list[:, 1]) < min_y:
                            min_y = min(point_list[:, 1])
                        if max(point_list[:, 0]) > max_x:
                            max_x = max(point_list[:, 0])
                        if max(point_list[:, 1]) > max_y:
                            max_y = max(point_list[:, 1])
                        area += calculate_polygon_area(point_list)

                    coco_annotation.segmentation = points
                    coco_annotation.area = area
                    coco_annotation.iscrowd = 0
                    coco_annotation.image_id = image_id
                    coco_annotation.bbox = [float(round(min_x, 2)),
                                            float(round(min_y, 2)),
                                            (float(round(max_x, 2)) - float(round(min_x, 2))) + 1,
                                            (float(round(max_y, 2)) - float(round(min_y, 2))) + 1]
                    coco_annotation.category_id = labels.index(label)
                    coco_annotation.id = id

                    COCO.annotations.append(coco_annotation)
                    id += 1
                else:
                    warnings.warn(
                        f"This method does not currently support exporting the {drawType} label type",
                        UserWarning)
            image_id += 1

        COCO.info = {
            'description': 'Convert from ABAVA dataset to COCO dataset',
            'url': 'https://github.com/ABAKA-AI/abavasdk',
            'version': 'ABAVA SDK V1.0',
            'year': f"{datetime.utcnow().year}",
            'contributor': '',
            'date_created': datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }

        file_name = self.source_data.task.name + '.json'
        self.save_labels(self.abava2dict(COCO), file_name)

