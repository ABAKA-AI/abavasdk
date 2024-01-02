# -*-coding:utf-8 -*-
from abava import Visual


data_path = "./datapath"
out_path = './outpath'
image_path = './imagepath'
# colour_dict = {
#     '车': (0, 0, 255),
#     '人': (0, 255, 0),
#     '建筑': (255, 0, 0),
# }
Visual.visual_voc(data_path, image_path, out_path)
