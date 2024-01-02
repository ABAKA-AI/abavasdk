# -*-coding:utf-8 -*-
from abava import Visual


data_path = "./test.json"
out_path = "./output"
colour_dict = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
}
Visual.visual_coco(data_path, out_path)