#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings
from os.path import join
import urllib.parse
from pathlib import Path

from tqdm import tqdm
from lxml import etree, objectify

from .voc import VOC_MAKER
from ..export import ExportData
from ...utils import chunks


class ExportVoc(ExportData):
    def __init__(self, source_data, out_path=None, mapping=None):
        """
        Converting abava data format to voc data format
        :param source_data: source abava data
        :param out_path:
        :param mapping: label-color mapping
        """
        super(ExportVoc, self).__init__(source_data, out_path, mapping)
        label_obj_list, version, label_type, is_ori_tag = self.match_label_data()
        self.label_obj_list = label_obj_list
        self.version = version
        self.label_type = label_type
        self.is_ori_tag = is_ori_tag

    def abava_json2voc(self):
        """
        Converting ABAVA raw data to voc format
        :return: xml file in voc format
        """
        for data in tqdm(self.label_obj_list):
            abava_url = data['info']
            image_name = Path(abava_url).parts[-1]
            height = data['size']['height']
            width = data['size']['width']

            anno_tree = VOC_MAKER.annotation(
                VOC_MAKER.folder(),
                VOC_MAKER.filename(image_name),
                VOC_MAKER.path('/'.join(Path(abava_url).parts[4:])),
                VOC_MAKER.source(
                    VOC_MAKER.database('Unknown')
                ),
                VOC_MAKER.size(
                    VOC_MAKER.width(width),
                    VOC_MAKER.height(height),
                    VOC_MAKER.depth(3)
                ),
                VOC_MAKER.segmented(0),
            )
            for l in data['labels']:
                if len(data['labels']) == 0:
                    continue
                drawType = l.data.drawType
                name = l.data.label
                if self.mapping:
                    name = self.mapping[name]
                if drawType == 'RECTANGLE':
                    xmin = int(l.data.points[0])
                    ymin = int(l.data.points[1])
                    xmax = int(l.data.points[2])
                    ymax = int(l.data.points[3])
                    anno_tree.append(
                        VOC_MAKER.object(
                            VOC_MAKER.name(name),
                            VOC_MAKER.pose('Unspecified'),
                            VOC_MAKER.truncated('Unspecified'),
                            VOC_MAKER.difficult(0),
                            VOC_MAKER.bndbox(
                                VOC_MAKER.xmin(xmin),
                                VOC_MAKER.ymin(ymin),
                                VOC_MAKER.xmax(xmax),
                                VOC_MAKER.ymax(ymax)
                            )
                        )
                    )
                elif drawType == 'POLYGON':
                    E = objectify.ElementMaker(annotate=False)
                    bndbox = E.bndbox()
                    point_voc = chunks(l.data.points, 2)
                    for idx, v in enumerate(point_voc):
                        bndbox.append(E(f'x{idx}', v[0]))
                        bndbox.append(E(f'y{idx}', v[1]))
                    anno_tree.append(
                        VOC_MAKER.object(
                            VOC_MAKER.name(name),
                            VOC_MAKER.pose('Unspecified'),
                            VOC_MAKER.truncated('Unspecified'),
                            VOC_MAKER.difficult(0),
                            bndbox
                        )
                    )
                else:
                    warnings.warn(
                        f"This method does not currently support exporting the {drawType} label type",
                        UserWarning)

            if self.out_path:
                file_path = 'labels'
                if '/'.join(Path(abava_url).parts[4:-1]) is not None:
                    root = '/'.join(Path(abava_url).parts[4:-1])
                    base_root = join(self.out_path, file_path)
                    out_root = join(base_root, root)
                else:
                    out_root = join(self.out_path, file_path)
            else:
                if '/'.join(Path(abava_url).parts[4:-1]) is not None:
                    root = '/'.join(Path(abava_url).parts[4:-1])
                    out_root = join('./labels', root)
                else:
                    out_root = './labels'
            out_root = urllib.parse.unquote(out_root)
            if not os.path.exists(out_root):
                os.makedirs(out_root)
            etree.ElementTree(anno_tree).write(join(out_root, image_name[:-4] + ".xml"), pretty_print=True,
                                               encoding='utf-8')
