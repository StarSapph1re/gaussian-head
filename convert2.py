import json
# 用于没有原来的相机外参，纯用视频+IMAvatar输出的情况
org = {
  "fx": 600.0,
  "fy": 600.0,
  "cx": 256.0,
  "cy": 256.0,
  "h": 512,
  "w": 512,
  "frames": [],
  "shape_params": []
}

# 加载flame_params.json文件
with open('flame_params.json', 'r') as file:
    flame_data = json.load(file)

# 遍历frames列表
img_id = 1
for frame in flame_data['frames']:
    # 提取需要的字段，并为transform_matrix增加一行
    new_frame = {
        "img_id": img_id,
        "transform_matrix": frame["world_mat"] + [[0.0, 0.0, 0.0, 1.0]],
        "pose": frame["pose"],
        "expression": frame["expression"]
    }
    org["frames"].append(new_frame)
    img_id += 1

# 添加shape_params字段的值
org['shape_params'] = flame_data['shape_params']

# 将org字典输出为json格式，并保存到当前目录下的transforms.json文件中
with open('new.json', 'w') as outfile:
    json.dump(org, outfile, indent=2)

print("Done!")