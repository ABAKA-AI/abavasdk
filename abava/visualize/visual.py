# -*-coding:utf-8 -*-

import os
import random
import urllib.parse
from os.path import join
import cv2

from ..utils import general, cv_tools


class VisualData():
    def __init__(self):
        """
        :param source_data: Raw data from the abaka database
        """

    def get_label_colour(self, source_data):
        """
        Generate the colour corresponding to the label
        :return: dict
        """
        colour_dict = {}
        for i in source_data.task.setting.labelConfig:
            drawType = i.drawType
            if drawType == 'POINT':
                if 'valueOptions' in i.attributes[0]:
                    sub_label = i.attributes[0].valueOptions
                else:
                    sub_label = i.attributes[0].children
                for j in range(len(sub_label)):
                    label = sub_label[j]
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    colour_dict[label] = (b, g, r)
            else:
                label = i.label
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                colour_dict[label] = (b, g, r)
        return colour_dict

    def get_colour_yolo(self, label_path):
        """
        Generate the corresponding colours of the label in yolo format
        :param label_path: Location of label_mapping.txt
        :return: dict
        """
        colour_dict = {}
        with open(label_path, 'r') as f:
            labels = f.read()
            label_dict = general.str2dict(labels)
            for key in label_dict.keys():
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                colour_dict[key] = (r, g, b)
        return colour_dict

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

    def match_label_data(self, source_data):
        # 是否带原图
        is_ori_tag = source_data.config.resource

        # 根据图片生成对象
        label_obj_list = []
        label_type = source_data.task.type
        version = source_data.task.version
        for labeldata in source_data.data:
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
                    labels = list(filter(lambda x: x['data']['frameIndex'] == mdiu_id, label_data_labels))
                else:
                    labels = list(filter(lambda x: x['data']['count'] == mdiu_id + 1, label_data_labels))
                label_obj = {
                    'info': mdiu_va,
                    'size': label_data_infos_sizes[mdiu_id],
                    'labels': labels
                }
                if labels:
                    label_obj_list.append(label_obj)

        return label_obj_list, version, label_type, is_ori_tag
