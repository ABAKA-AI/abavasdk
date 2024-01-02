# -*-coding:utf-8 -*-
import abava
from abava import Export

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
mapping = {"背景": (100, 100, 100), "草坪": (200, 200, 200),"路": (300, 300, 300),"地形": (41, 43, 41),"固定障碍物": (5, 5, 5), "静态障碍物": (6, 6, 6), "动态障碍物": (7 ,7, 7),
           "灌木": (8, 8, 8), "粪便": (9, 9, 9), "充电桩": (10, 10, 10), "脏污": (11, 11, 11), "阳光光线": (12,12,12), "玻璃": (13,13,13), '未标注': (0, 0, 0)}

# plate-mode mask
out_path = "./output"
Export.p_mask_plus(source_data, out_path)

