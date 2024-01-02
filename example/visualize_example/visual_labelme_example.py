# -*-coding:utf-8 -*-
from abava import Visual


data_path = "./test"
out_path = "./output"
# colour_dict = {
#     '车': (0, 0, 255),
#     '人': (0, 255, 0),
#     '建筑': (255, 0, 0),
# }
Visual.visual_labelme(data_path, out_path)