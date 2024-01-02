# -*-coding:utf-8 -*-
from abava import Visual


data_path = "./datapath"
label_path = "./label.txt"
image_path = './imagepath'
out_path = './outpath'
# colour_dict = {
#     '车': (0, 0, 255),
#     '人': (0, 255, 0),
#     '建筑': (255, 0, 0),
# }
Visual.visual_yolo(data_path, label_path, image_path, out_path)
