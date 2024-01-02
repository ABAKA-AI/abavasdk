# -*-coding:utf-8 -*-
import glob
from os.path import join
from pathlib import Path
import cv2

from ..visual import VisualData
from ...utils import general, cv_tools


class VisualYolo(VisualData):
    def __init__(self, data_path, label_path, image_path, out_path=None, colour_dict=None):
        """
        :param data_path: yolo file path
        :param label_path: yolo format label mapping file location
        :param image_path: Image path
        :param out_path: Output location, or if empty, output in the same directory as the script
        :param colour_dict: label-color mapping
        """
        super().__init__()
        if colour_dict is None:
            colour_dict = {}
        self.data_path = data_path
        self.label_path = label_path
        self.image_path = image_path
        self.out_path = out_path
        self.colour_dict = colour_dict

    def visual_yolo(self):
        """
        Visualisation of yolo data
        :return:
        """
        colour_dict = self.get_colour_yolo(self.label_path)
        with open(self.label_path, 'r') as label_txt:
            labels = label_txt.read()
            label_dict = general.str2dict(labels)

        label_paths = glob.glob(self.data_path)
        for label_path in label_paths:
            img_name = Path(label_path).parts[-1].replace('txt', 'jpg')
            img = cv2.imread(join(self.image_path, img_name))
            width = img.shape[1]
            height = img.shape[0]
            label_datas = open(label_path, 'r', encoding='utf-8')
            datas = label_datas.readlines()
            for object in datas:
                label, x, y, w, h = object.strip('\n').split(' ')
                label_name = general.get_key(label_dict, int(label))[0]
                xmin = (float(x) - float(w) / 2.) * width
                ymin = (float(y) - float(h) / 2.) * height
                xmax = (float(x) + float(w) / 2.) * width
                ymax = (float(y) + float(h) / 2.) * height
                img = cv_tools.draw_rectangle((xmin, ymin, xmax, ymax), img, label_name, colour_dict[label_name])
            self.save_image(img, img_name, self.out_path)
