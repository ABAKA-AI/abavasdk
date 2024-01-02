# -*-coding:utf-8 -*-

import json
import os
import urllib.parse
from os.path import join
from pathlib import Path

import PIL
import requests
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

from ..export import ExportData
from ...utils import cv_tools
from ...abava_data import abava2dict


class ExportMask(ExportData):
    def __init__(self, source_data, out_path=None, mapping=None):
        """
        Converting abava data format to mask
        :param source_data: Raw data from the abaka database
        :param out_path: output path
        :param mapping: colour mapping {'car': (23,123,43), 'person': (255, 12, 53)}
        """
        super().__init__(source_data)
        self.out_path = out_path
        self.mapping = mapping

    def p_mask(self, save_folder='images'):
        """
        Drawing color masks
        :return:
        """
        mask_data, version, mask_type, is_ori = self.match_label_data()
        unmarked_mapping = {'unmarked': (0, 0, 0)}
        if self.mapping:
            color_mapping = {**unmarked_mapping, **self.mapping}
        else:
            color_mapping = {'unmarked': (0, 0, 0)}
            for label_color in self.source_data['labelConfig']:
                color_mapping[label_color['label']] = cv_tools.hex_to_rgb(label_color['color'])
        grey_values = [(i, i, i) for i in range(len(color_mapping))]
        grey_mapping = dict(zip(list(color_mapping.keys()), grey_values))

        for i in tqdm(range(len(mask_data))):
            url = mask_data[i]['info']
            try:
                height = i.info.info.size.height
                width = i.info.info.size.width
            except:
                height, width = cv_tools.get_urlimage_size(url)
            img = np.zeros((height * 2, width * 2, 3), np.uint8)
            labels = mask_data[i]['labels']

            if mask_type == 'MASK':
                for l in labels:
                    label = l.data.label
                    color_grey = grey_mapping[label]
                    types = l.data.pathData.type
                    if types == 'CompoundPath':
                        path_list = []
                        for item in l.data.pathData.data:
                            path_list.append([[point[0] * 2, point[1] * 2] for point in item.data])
                        np_paths = []
                        for polygon in path_list:
                            points = [polygon]
                            path = np.array(points, np.int32)
                            np_paths.append(path)
                        cv2.fillPoly(img, np_paths, color=color_grey)
                    else:
                        points = [[point[0] * 2, point[1] * 2] for point in l.data.pathData.data]
                        path = np.array(points, np.int32)
                        cv2.fillPoly(img, [path], color=color_grey)
            else:
                labels = sorted(labels,
                                key=lambda x: x['data']['zIndex'] if 'zIndex' in abava2dict(x['data']) else x['data'][
                                    'id'])
                for l in labels:
                    if len(l.data.points) == 0:
                        continue
                    drawType = l.data.drawType
                    # OK
                    if drawType == 'POLYGON':
                        label = l.data.label
                        color_grey = grey_mapping[label]
                        item = l.data.points
                        area = np.array([[item[i] * 2, item[i + 1] * 2] for i in range(0, len(item), 2)], np.int32)
                        cv2.fillPoly(img, [area], color_grey)
                    # OK
                    elif drawType == 'RECTANGLE':
                        label = l.data.label
                        color_grey = grey_mapping[label]
                        top_left_x, top_left_y, bottom_right_x, bottom_right_y = l.data.points
                        top_right = [bottom_right_x * 2, top_left_y * 2]
                        bottom_left = [top_left_x * 2, bottom_right_y * 2]
                        area = np.array(
                            [[top_left_x * 2, top_left_y * 2], top_right, [bottom_right_x * 2, bottom_right_y * 2],
                             bottom_left],
                            np.int32)
                        cv2.fillPoly(img, [area], color_grey)
                    # OK
                    else:
                        path_list = []
                        for item in l.data.points:
                            path_list.append([[item[i] * 2, item[i + 1] * 2] for i in range(0, len(item), 2)])
                        np_paths = []
                        for polygon in path_list:
                            points = [polygon]
                            path = np.array(points, np.int32)
                            np_paths.append(path)
                        label = l.data.label
                        color_grey = grey_mapping[label]
                        cv2.fillPoly(img, np_paths, color=color_grey)
            new_img = img[1::2, 1::2, :]
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

            lbl_pil = PIL.Image.fromarray(new_img.astype(np.uint8), mode="P")
            lbl_pil.putpalette(np.asarray(list(color_mapping.values())).astype(np.uint8).flatten())

            if 'http' in url and '/'.join(Path(url).parts[4:-1]) is not None:
                folder = '/'.join(Path(url).parts[4:-1])
                folder = urllib.parse.unquote(folder)
            elif 'http' not in url and '/'.join(Path(url).parts[3:-1]) is not None:
                folder = '/'.join(Path(url).parts[3:-1])
                folder = urllib.parse.unquote(folder)
            else:
                folder = None

            mask_name = url.split('/')[-1]
            if mask_name.split('.')[-1] != 'png':
                mask_name = mask_name.replace(mask_name.split('.')[-1], 'png')

            if self.out_path is None:
                if folder:
                    out_root = join(f'./{save_folder}/data/', folder)
                    ori_root = os.path.join(f'./{save_folder}/resource/', folder)
                else:
                    out_root = f'./{save_folder}/data/'
                    ori_root = f'./{save_folder}/resource/'
                if not os.path.exists(out_root):
                    os.makedirs(out_root)
                if not os.path.exists(ori_root):
                    os.makedirs(ori_root)
                with open('./color_mapping.json', 'w', encoding='utf-8') as write_f:
                    json.dump(color_mapping, write_f, indent=2, ensure_ascii=False)
            else:
                file_path = f'{save_folder}/data/'
                ori_path = f'{save_folder}/resource/'
                if folder:
                    base_out_root = join(self.out_path, file_path)
                    out_root = join(base_out_root, folder)
                    base_ori_root = join(self.out_path, ori_path)
                    ori_root = join(base_ori_root, folder)
                else:
                    out_root = join(self.out_path, file_path)
                    ori_root = join(self.out_path, ori_path)
                if not os.path.exists(out_root):
                    os.makedirs(out_root)
                if not os.path.exists(ori_root):
                    os.makedirs(ori_root)
                with open(self.out_path + 'colour_mapping.json', 'w', encoding='utf-8') as write_f:
                    json.dump(color_mapping, write_f, indent=2, ensure_ascii=False)
            lbl_pil.save(join(out_root, mask_name))

            if is_ori:
                resp = requests.get(url)
                ori_save_path = os.path.join(ori_root, mask_name)
                with open(ori_save_path, 'wb') as f:
                    f.write(resp.content)
        return out_root
