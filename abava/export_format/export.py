#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import urllib.parse
from os.path import join

import cv2

from ..utils import cv_tools
from ..abava_data import ABAVA


class ExportData():
    def __init__(self, source_data, out_path=None, mapping=None, **kwargs):
        self.source_data = ABAVA(source_data)
        self.mapping = mapping
        self.out_path = out_path
        self.personalization = kwargs.get("personalization", {})
        self.type = self.source_data.task.type

    def save_labels(self, final_data, filename, sub_folder=None):
        """
        :param final_data: Final converted data
        :param filename: Saved file name
        :param sub_folder: Subcategories
        :return: json file
        """
        filename = urllib.parse.unquote(filename)
        if self.out_path is None:
            if sub_folder:
                out_root = join('./labels', sub_folder)
            else:
                out_root = './labels/'
            if not os.path.exists(out_root):
                os.makedirs(out_root)
            with open(join(out_root, filename), 'w', encoding='utf-8') as write_f:
                json.dump(final_data, write_f, indent=2, ensure_ascii=False)
        else:
            file_path = 'labels'
            if sub_folder:
                base_root = join(self.out_path, file_path)
                out_root = join(base_root, sub_folder)
            else:
                out_root = join(self.out_path, file_path)
            if not os.path.exists(out_root):
                os.makedirs(out_root)
            with open(join(out_root, filename), 'w', encoding='utf-8') as write_f:
                json.dump(final_data, write_f, indent=2, ensure_ascii=False)

    def abava2dict(self, data):
        """
        Converting ABAVA types to dict
        :param data:
        :return:
        """
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    data[f'{k}'] = [self.abava2dict(d) for d in v]
                elif repr(v) == 'molar_ai_format':
                    data[f'{k}'] = self.abava2dict(v)
        elif repr(data) == 'molar_ai_format':
            data = self.abava2dict(vars(data))
        return data

    def gen_structure_json(self, data):
        """
        Writing dict to json files
        :param data:
        :return:
        """
        if isinstance(data, dict):
            with open('molar_ai_format.json', 'w') as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            raise Exception('formatting error')

    def match_label_data(self):
        # With or without the original picture
        is_ori_tag = self.source_data.config.resource

        # Generate objects from images
        label_obj_list = []
        label_type = self.source_data.task.type
        version = self.source_data.task.version
        for labeldata in self.source_data.data:
            label_data_infos_urls = labeldata.info.info.url
            label_data_infos_sizes = []
            if labeldata.info.info.size:
                label_data_infos_sizes = labeldata.info.info.size
            else:
                for url in labeldata.info.info.url:
                    height, width = cv_tools.get_urlimage_size(url)
                    size = {"width": width, "height": height}
                    label_data_infos_sizes.append(size)
            label_data_labels = labeldata.labels
            for mdiu_id, mdiu_va in enumerate(label_data_infos_urls):
                if version == 1:
                    labels = list(filter(lambda x: x['data']['frameIndex'] == mdiu_id,  label_data_labels))
                else:
                    labels = list(filter(lambda x: x['data']['count'] == mdiu_id+1,  label_data_labels))
                label_obj = {
                    'info': mdiu_va,
                    'size': label_data_infos_sizes[mdiu_id],
                    'labels': labels
                }
                if labels:
                    label_obj_list.append(label_obj)

        return label_obj_list, version, label_type, is_ori_tag

    def save_image(self, img, filename, out_path=None, folder=None):
        """
        Save image
        :param img: Image
        :param filename: Image filename
        :param out_path: Output location, or if None, output in the same directory as the script
        :param folder: Output folder
        :return: images
        """
        filename = urllib.parse.unquote(filename)
        if out_path is None:
            if folder:
                out_root = join('./images', folder)
            else:
                out_root = './images/'
            if not os.path.exists(out_root):
                os.makedirs(out_root)
            cv2.imwrite(join(out_root, filename), img)
        else:
            file_path = 'images'
            if folder:
                base_root = join(out_path, file_path)
                out_root = join(base_root, folder)
            else:
                out_root = join(out_path, file_path)
            if not os.path.exists(out_root):
                os.makedirs(out_root)
            img_out_path = join(out_root, filename)
            cv2.imencode('.png', img)[1].tofile(img_out_path)

    def _query(self, key, value):
        """
        Callback function for customising the addition or deletion of certain fields in the standard export format
        :param key: Property names in standard format
        :param value: Data available for manipulation and can be customised to add attributes
        :return:
        """
        if not self.personalization:
            return ABAVA({})
        for k, v in self.personalization.items():
            add_location = k
            for k1, v1 in v.items():
                add_key = k1
                add_value = v1
        if key == add_location:
            a = {f"{add_key}": add_value(value)}
            return ABAVA({f"{add_key}": add_value(value)})
        else:
            return ABAVA({})