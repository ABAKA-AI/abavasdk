# -*-coding:utf-8 -*-

import urllib.parse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..visual import VisualData
from ...abava_data import ABAVA
from ...utils import cv_tools


class VisualSource(VisualData):
    def __init__(self, source_data, out_path=None, colour_dict=None):
        """
        :param source_data: Raw data from the abaka database
        :param out_path: output path
        :param colour_dict: label-color mapping
        """
        super().__init__()
        if colour_dict is None:
            colour_dict = {}
        self.source_data = ABAVA(source_data)
        self.out_path = out_path
        self.colour_dict = colour_dict
        label_obj_list, version, label_type, is_ori_tag = self.match_label_data(self.source_data)
        self.label_obj_list = label_obj_list
        self.version = version
        self.label_type = label_type
        self.is_ori_tag = is_ori_tag

    def visual_source(self):
        """
        Visualisation of raw data, RECTANGLE, POINT
        :return:
        """
        if self.colour_dict == {}:
            self.colour_dict = self.get_label_colour(self.source_data)

        for data in tqdm(self.label_obj_list):
            abava_url = data['info']
            file_name = Path(abava_url).parts[-1]
            img = cv_tools.read_url_image(abava_url)

            for l in data['labels']:
                label = l.data.label
                drawtype = l.data.drawType
                if drawtype == 'RECTANGLE':
                    points = l.data.points
                    img = cv_tools.draw_rectangle((points[0], points[1], points[2], points[3]),
                                                  img, label, self.colour_dict[label])
                elif drawtype == 'POINT':
                    points = np.array(l.data.points)
                    point_source = points.reshape((-1, 2))
                    for k in range(len(point_source)):
                        x = point_source[k][0]
                        y = point_source[k][1]
                        cv_tools.draw_point([x,y], img, label, self.colour_dict[label])
                elif drawtype == 'MASK' or drawtype == 'MASK_BLOCK':
                    points = l.data.points
                    img = cv_tools.draw_polygon(points, img, self.colour_dict[label])

            if '/'.join(Path(abava_url).parts[4:-1]) is not None:
                folder = '/'.join(Path(abava_url).parts[4:-1])
            else:
                folder = None
            folder = urllib.parse.unquote(folder)
            self.save_image(img, file_name, self.out_path, folder)
