# -*-coding:utf-8 -*-
import glob
from os.path import join
import cv2

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
from ..visual import VisualData
from ...utils import cv_tools


class VisualVoc(VisualData):
    def __init__(self, data_path, image_path, out_path=None, colour_dict=None):
        """
        :param data_path: xml file path
        :param image_path: image path
        :param out_path: output path
        :param colour_dict: label-color mapping
        """
        super().__init__()
        if colour_dict is None:
            colour_dict = {}
        self.data_path = data_path
        self.image_path = image_path
        self.out_path = out_path
        self.colour_dict = colour_dict

    def visual_voc(self):
        """
        Visualisation of voc data
        :return:
        """
        # colour_dict = self.get_label_colour()
        # label_dict = self.get_label_dict()

        vocs = glob.glob(self.data_path)
        for voc in vocs:
            img_name = voc.replace('xml', 'jpg')

            img = cv2.imread(join(self.image_path, img_name))
            tree = ET.parse(voc)
            root_xml = tree.getroot()
            for object in root_xml.findall('object'):
                label_name = object.find('name').text
                if label_name not in self.colour_dict:
                    self.colour_dict[label_name] = cv_tools.generate_random_color()

                bndbox = object.find('bndbox')
                children = list(bndbox)
                drawtype = None
                if len(children) == 4:
                    drawtype == 'RECTANGLE'
                elif len(children) == 2:
                    drawtype == 'POINT'
                elif len(children) > 4:
                    drawtype == 'POLYGON'

                if drawtype == 'RECTANGLE':
                    xmin = bndbox.find('xmin').text
                    ymin = bndbox.find('ymin').text
                    xmax = bndbox.find('xmax').text
                    ymax = bndbox.find('ymax').text
                    img = cv_tools.draw_rectangle((xmin, ymin, xmax, ymax),
                                                  img, label_name, self.colour_dict[label_name])
                elif drawtype == 'POINT':
                    x = bndbox.find('x').text
                    y = bndbox.find('y').text
                    img = cv_tools.draw_point((x, y), img, label_name, self.colour_dict[label_name])
                elif drawtype == 'POLYGON':
                    points = []
                    for i in range(len(children)):
                        x = bndbox.find(f'x{i}').text
                        y = bndbox.find(f'y{i}').text
                        points.append(x)
                        points.append(y)
                    img = cv_tools.draw_polygon([points], img, self.colour_dict[label_name])

            self.save_image(img, img_name, self.out_path)
