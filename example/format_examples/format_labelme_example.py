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
mapping = {"背景": 'Background', "草坪": 'Lawn',"路": 'Road',
           "地形": 'Terrain',"固定障碍物": 'Fixed Obstacle', '静态障碍物': 'Static Obstacle', '动态障碍物': 'Dynamic Obstacle',
           '灌木': 'Bush', '粪便': 'Faeces', '充电桩': 'Charging Station', '脏污': 'Dirt', '未标注': 'Unlabeled', '阳光光线': 'Sunlight', '玻璃': 'Glass'}


# labelme
out_path = "./output"
Export.abava_json2labelme(source_data, out_path, mapping)
