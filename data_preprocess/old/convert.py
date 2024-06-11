import json
# 用于Gaussian-head给定相机参数 + DECA推理FLAME参数合并
with open('transforms-org.json') as file:
    org = json.load(file)

# 加载flame_params.json文件
with open('id1-flame.json', 'r') as file:
    flame_data = json.load(file)

# 遍历frames列表
img_id = 0
for frame in flame_data['frames']:
    # 将新的frame字典添加到org的frames列表中
    org['frames'][img_id]["img_id"] += 1
    org['frames'][img_id]["pose"] = frame["pose"]
    org['frames'][img_id]["expression"] = frame["expression"]
    # 用推理的相机参数
    # org["frames"][img_id]["transform_matrix"] = frame["world_mat"] + [[0.0, 0.0, 0.0, 1.0]]
    del org['frames'][img_id]["near"]
    del org['frames'][img_id]["far"]
    del org['frames'][img_id]["aud_id"]
    del org['frames'][img_id]["exp"]
    del org['frames'][img_id]["exp_ori"]
    del org['frames'][img_id]["face_rect"]

    img_id += 1

# 添加shape_params字段的值
org['shape_params'] = flame_data['shape_params']

# 将org字典输出为json格式，并保存到当前目录下的new.json文件中
with open('new.json', 'w') as outfile:
    json.dump(org, outfile, indent=2)
