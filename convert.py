import json
with open('transforms-org.json') as file:
    org = json.load(file)
# org = {
#   "fx": 600.0,
#   "fy": 600.0,
#   "cx": 256.0,
#   "cy": 256.0,
#   "h": 512,
#   "w": 512,
#   "frames": [],
#   "shape_params": []
# }

# 首先，加载flame_params.json文件
with open('id1-flame.json', 'r') as file:
    flame_data = json.load(file)

# 遍历frames列表
img_id = 0
for frame in flame_data['frames']:
    # 提取需要的字段，并为transform_matrix增加一行
    new_frame = {
        "img_id": img_id,
        "pose": frame["pose"],
        "expression": frame["expression"]
    }
    # 将新的frame字典添加到org的frames列表中
    org['frames'][img_id]["img_id"] += 1
    org['frames'][img_id]["pose"] = frame["pose"]
    org['frames'][img_id]["expression"] = frame["expression"]
    # 用推理的相机参数
    org["frames"][img_id]["transform_matrix"] = frame["world_mat"] + [[0.0, 0.0, 0.0, 1.0]]
    del org['frames'][img_id]["near"]
    del org['frames'][img_id]["far"]
    del org['frames'][img_id]["aud_id"]
    del org['frames'][img_id]["exp"]
    del org['frames'][img_id]["exp_ori"]
    del org['frames'][img_id]["face_rect"]


    img_id += 1

# 添加shape_params字段的值
org['shape_params'] = flame_data['shape_params']

# 将org字典输出为json格式，并保存到当前目录下的transforms.json文件中
with open('new.json', 'w') as outfile:
    json.dump(org, outfile, indent=2)
'''
请你补全这段python代码：
读取当前目录下的flame_params.json，这个文件中有一个frames字段是一个列表，列表的每一个元素是一个字典，
遍历这个列表，将元素的pose，expression，world_mat字段都提取出来，并包装在一个字典中，并附上一个img_id值从1开始，
对于world_mat，他的值应该放在字典的"transform_matrix"字段中，这是一个3x4的列表，请你为他增加一行[0.0,0.0,0.0,1.0]变为4x4,
最后，元素应该像这样：
{
  "img_id": 1,
  "transform_matrix":[...],
  "pose": [...],
  "expression": [...],
}
然后把每一个这样的元素依次放入org的frames列表中，
  
然后，flame_params.json中还有一个字段是"shape_params"，需要把它的值放到org这个字典里面。
最后，将org这个字典以json格式输出为当前目录下的文件"transforms.json"。
'''