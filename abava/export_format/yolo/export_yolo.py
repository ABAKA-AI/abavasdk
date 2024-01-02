#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
from os.path import join
import urllib.parse
from pathlib import Path

from tqdm import tqdm

from ..export import ExportData


class ExportYolo(ExportData):
    def __init__(self, source_data, out_path=None, mapping=None):
        """
        Converting abava data format to yolo data format
        :param source_data: source abava data
        :param out_path:
        :param mapping: label-color mapping
        """
        super(ExportYolo, self).__init__(source_data, out_path, mapping)
        label_obj_list, version, label_type, is_ori_tag = self.match_label_data()
        self.label_obj_list = label_obj_list
        self.version = version
        self.label_type = label_type
        self.is_ori_tag = is_ori_tag

    def abava_json2yolo(self):
        """
        Converting ABAVA raw data to yolo format
        :return: txt file in yolo format
        """
        labels = []
        for i in self.source_data.task.setting.labelConfig:
            draw_type = i.drawType
            if draw_type == 'RECTANGLE':
                label = i.label
                if self.mapping:
                    label = self.mapping[label]
                labels.append(label)

        for data in tqdm(self.label_obj_list):
            abava_url = data['info']
            label_name = Path(abava_url).parts[-1].split('.')[0]
            label_name = label_name + '.txt'

            if self.out_path:
                file_path = 'labels'
                if '/'.join(Path(abava_url).parts[4:-1]) is not None:
                    root = '/'.join(Path(abava_url).parts[4:-1])
                    base_root = join(self.out_path, file_path)
                    out_root = join(base_root, root)
                else:
                    out_root = join(self.out_path, file_path)
                out_root = urllib.parse.unquote(out_root)
                if not os.path.exists(out_root):
                    os.makedirs(out_root)
            else:
                if '/'.join(Path(abava_url).parts[4:-1]) is not None:
                    root = '/'.join(Path(abava_url).parts[4:-1])
                    out_root = join('./labels', root)
                else:
                    out_root = './labels'
                out_root = urllib.parse.unquote(out_root)
                if not os.path.exists(out_root):
                    os.makedirs(out_root)
            yolo_anno = open(join(out_root, label_name), 'w', encoding='utf-8')

            height = data['size']['height']
            width = data['size']['width']

            for l in data.labels:
                drawType = l.data.drawType
                if drawType == 'RECTANGLE':
                    points = l.data.points
                    label = l.data.label
                    index = labels.index(label)
                    point_x = (points[2] + points[0]) / (2 * width)
                    point_y = (points[3] + points[1]) / (2 * height)
                    point_w = (points[2] - points[0]) / width
                    point_h = (points[3] - points[1]) / height
                    yolo_anno.write(
                        str(index) + ' ' + str(point_x) + ' ' + str(point_y) + ' ' + str(point_w) + ' ' + str(
                            point_h) + '\n')
                else:
                    warnings.warn(
                        f"This method does not currently support exporting the {drawType} label type",
                        UserWarning)
            yolo_anno.close()

        label_index = [i for i in range(0, len(labels) + 1)]
        label_dict = dict(zip(labels, label_index))
        label_file = 'labels_mapping.txt'
        if self.out_path:
            out_mapping = join(self.out_path, 'labels')
        else:
            out_mapping = './labels'
        labels_mapping = open(join(out_mapping, label_file), 'w')
        labels_mapping.write(str(label_dict))
        labels_mapping.close()
