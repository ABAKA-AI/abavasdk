#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import urllib.parse
import warnings
from os.path import join
from pathlib import Path

from tqdm import tqdm
import base64
import zlib
import json
import numpy as np
from ..export import ExportData
from ...utils import find_nth, find_last


class ExportKitti(ExportData):
    def __init__(self, source_data, out_path=None, mapping=None):
        """
        Converting abava data format to kitti data format
        :param source_data: source abava data
        :param out_path:
        :param mapping: label-color mapping
        """
        super(ExportKitti, self).__init__(source_data, out_path, mapping)

    def abava_json2odkitti(self):
        """
        Converting ABAVA raw data to kitti format
        :return: txt file in lidar-based kitti format
        """
        for i in tqdm(self.source_data.data):
            abava_url = i.info.info.url
            for j in range(len(abava_url)):
                image_name = Path(abava_url[j]).parts[-1].split('.')[0]
                image_name = image_name + '.txt'
                start = find_nth(abava_url[j], '/', 4)
                end = find_last(abava_url[j], '/')
                if self.out_path:
                    file_path = 'labels'
                    if start != end:
                        root = abava_url[j][start + 1: end]
                        base_root = join(self.out_path, file_path)
                        out_root = join(base_root, root)
                    else:
                        out_root = join(self.out_path, file_path)
                    out_root = urllib.parse.unquote(out_root)
                    if not os.path.exists(out_root):
                        os.makedirs(out_root)
                else:
                    if start != end:
                        root = abava_url[j][start + 1: end]
                        out_root = join('./labels', root)
                    else:
                        out_root = './labels'
                    out_root = urllib.parse.unquote(out_root)
                    if not os.path.exists(out_root):
                        os.makedirs(out_root)
                kitti_anno = open(join(out_root, image_name), 'w')
                k_content = ''

                for l in i.labels:
                    if len(i.labels) == 0:
                        continue
                    frames = l.data.frameIndex
                    if j == frames:
                        draw_type = l.data.drawType
                        if draw_type == 'box3d':
                            label = l.data.label
                            points = l.data.points
                            alpha = points[5]
                            alpha = alpha + math.atan2(points[1], points[0]) + 1.5 * math.pi
                            if alpha < -math.pi:
                                alpha = alpha + 2 * math.pi
                            elif alpha > math.pi:
                                alpha -= 2 * math.pi
                            k_content += label + ' ' + '0' + ' ' + '0' + str(
                                alpha) + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + '0' + ' ' + points[8] + ' ' + points[
                                             7] + ' ' + points[6] + ' ' + points[0] + ' ' + points[1] + ' ' + points[
                                             2] + ' ' + points[5] + ' ' + '1' + '\n'
                        else:
                            warnings.warn(
                                f"This method does not currently support exporting the {draw_type} label type",
                                UserWarning)

                kitti_anno.write(k_content)

    def abava_json2segkitti(self):
        """
        Converting ABAVA raw data to labelme format
        :return: json file in labelme format
        """
        label_dict = {}
        idMap = {}
        for id in range(len(self.source_data.labelConfig)):
            label_dict[id + 1] = self.source_data.labelConfig[id]["label"]
            idMap[label_dict[id + 1]] = id + 1
        for i in tqdm(self.source_data.data):
            abava_url = i.info.info.pcdUrl
            for j in range(len(abava_url)):
                image_name = Path(abava_url[j]).parts[-1].split('.')[0]
                image_name = image_name + '.label'
                start = find_nth(abava_url[j], '/', 4)
                end = find_last(abava_url[j], '/')
                if self.out_path:
                    file_path = 'labels'
                    if start != end:
                        root = abava_url[j][start + 1: end]
                        base_root = join(self.out_path, file_path)
                        out_root = join(base_root, root)
                    else:
                        out_root = join(self.out_path, file_path)
                    out_root = urllib.parse.unquote(out_root)
                    if not os.path.exists(out_root):
                        os.makedirs(out_root)
                else:
                    if start != end:
                        root = abava_url[j][start + 1: end]
                        out_root = join('./labels', root)
                    else:
                        out_root = './labels'
                    out_root = urllib.parse.unquote(out_root)
                    if not os.path.exists(out_root):
                        os.makedirs(out_root)
                itemLabelMap = {}
                labelIdMapStr = ''
                for label in i.labels:
                    frame = label.data.frameIndex
                    if j != frame:
                        continue
                    draw_type = label.data.drawType
                    if draw_type == "SEMANTIC_POINT":
                        itemLabelMap[label.data.id] = idMap[label.data.label]
                    elif draw_type == "SEMANTIC_BASE":
                        labelIdMapStr = label.data.pLabelIdMap
                    else:
                        warnings.warn(
                            f"This method does not currently support exporting the {draw_type} label type",
                            UserWarning)
                decompressed_data = zlib.decompress(base64.b64decode(labelIdMapStr), 16 + zlib.MAX_WBITS,
                                                    zlib.MAX_WBITS)
                labelIdMap = json.loads(decompressed_data)
                pcdLabels = []
                for labelId in labelIdMap:
                    if labelId in itemLabelMap:
                        pcdLabels.append(itemLabelMap[labelId])
                    else:
                        if labelId != -1:
                            print(f'警告，出现未定义标签：label.data.id={labelId}')
                        pcdLabels.append(-1)
                df = np.array(pcdLabels, dtype=np.uint32)
                df.tofile(join(out_root, image_name))
