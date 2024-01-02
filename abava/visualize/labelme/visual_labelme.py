# -*-coding:utf-8 -*-
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm

from ..visual import VisualData
from ...utils import general, cv_tools
from ...abava_data import ABAVA


class VisualLabelme(VisualData):
    def __init__(self, data_path, out_path=None, colour_dict=None):
        """
        :param data_path: Labelme folder path
        :param out_path: Output path
        :param colour_dict: label-color mapping
        """
        super().__init__()
        if colour_dict is None:
            colour_dict = {}
        self.data_path = data_path
        self.out_path = out_path
        self.colour_dict = colour_dict

    def visual_labelme(self):
        """
        Visualize labelme data, RECTANGLE, POINT
        :return:
        """
        jsons = glob.glob(self.data_path + '/*')
        for json in tqdm(jsons):
            load_dict = ABAVA(general.load_json(json))

            for ctg in load_dict.shapes:
                if ctg['label'] not in self.colour_dict:
                    self.colour_dict[ctg['label']] = cv_tools.generate_random_color()
            url = load_dict.imagePath
            file_name = Path(url).parts[-1]
            base64 = load_dict.imageData
            img = cv_tools.base642image(base64)
            for i in load_dict.shapes:
                drawtype = i.shape_type
                if drawtype == 'rectangle':
                    label = i.label
                    xmin = int(i.points[0][0])
                    ymin = int(i.points[0][1])
                    xmax = int(i.points[1][0])
                    ymax = int(i.points[1][1])
                    img = cv_tools.draw_rectangle((xmin, ymin, xmax, ymax), img, label, self.colour_dict[label])
                elif drawtype == 'point':
                    label = i.label
                    x = int(i.points[0][0])
                    y = int(i.points[0][1])
                    img = cv_tools.draw_point([x,y], img, label, self.colour_dict[label])
                elif drawtype == 'polygon':
                    label = i.label
                    points = [np.array(i.points).flatten().tolist()]
                    img = cv_tools.draw_polygon(points, img, self.colour_dict[label])

            self.save_image(img, file_name, self.out_path)
