# -*-coding:utf-8 -*-
from tqdm import tqdm

from ..visual import VisualData
from ...utils import general, cv_tools
from ...abava_data import ABAVA


class VisualCoco(VisualData):
    def __init__(self, data_path, out_path=None, colour_dict=None):
        """
        :param data_path: coco data path
        :param out_path: output path
        :param colour_dict: label-color mapping
        """
        super().__init__()
        if colour_dict is None:
            colour_dict = {}
        self.data_path = data_path
        self.out_path = out_path
        self.colour_dict = colour_dict

    def visual_coco(self):
        """
        Visualize coco data, RECTANGLE, POINT
        :return: image
        """
        load_dict = ABAVA(general.load_json(self.data_path))

        images = [None] * len(load_dict.images)
        for label in load_dict.annotations:
            image_id = label.image_id
            if images[image_id - 1] is None:
                images[image_id-1] = {
                    'file_name': load_dict.images[image_id-1].file_name,
                    'abava_url': load_dict.images[image_id-1].abava_url,
                    'label': [label]
                }
            else:
                images[image_id - 1]['label'].append(label)
        if self.colour_dict == {}:
            for ctg in load_dict.categories:
                self.colour_dict[ctg['id']] = cv_tools.generate_random_color()

        for image in tqdm(images):
            file_name = image['file_name']
            img = cv_tools.read_url_image(image['abava_url'][0])
            for label in image['label']:
                points = label.segmentation
                img = cv_tools.draw_polygon(points, img, self.colour_dict[label['category_id']])

            self.save_image(img, file_name, self.out_path)