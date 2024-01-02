# -*-coding:utf-8 -*-
import json

import abava
from abava import Check

"""
You can get your ak/sk in the platform's "Team Space" -> "Setting"
你可以在平台的"空间管理"->"空间设置"中拿到你的ak/sk
"""
ak = "Access Key"
sk = "Secret Key"
client = abava.Client(ak, sk)

"""
After creating an export task, you can see the export task id of the corresponding task 
in "Import/Export"->"Data Export".
创建导出任务后可以在"导入导出"->"数据导出"中看到对应任务的导出编号
"""
source_data = client.get_data('export_task_id')


def test_count_labels():
    count = Check.count_labels(source_data)
    print(count)


def test_count_aim_labels():
    aim_label = 'athlete'
    count = Check.count_aim_labels(source_data, aim_label)
    print(count)


def test_count_drawtype():
    drawtype = 'RECTANGLE'
    count = Check.count_drawtype(source_data, drawtype)
    print(count)


def test_count_files():
    files = Check.count_files(source_data)
    print(files)


def test_count_images():
    count = Check.count_images(source_data)
    print(count)


def test_unlabeld_images():
    count = Check.unlabeld_images(source_data)
    print(count)


def test_labeled_images():
    count = Check.labeled_images(source_data)
    print(count)


if __name__ == '__main__':
    test_count_labels()
    test_count_aim_labels()
    test_count_drawtype()
    test_count_files()
    test_count_images()
    test_unlabeld_images()
    test_labeled_images()