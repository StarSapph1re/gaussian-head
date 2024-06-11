import json

# 读取w2c.json文件
with open('w2c.json', 'r') as w2c_file:
    w2c_data = json.load(w2c_file)

# 读取flame_params.json文件
with open('flame_params.json', 'r') as flame_file:
    flame_data = json.load(flame_file)

# 删除flame_params.json中frames列表的最后一个元素（不知道为什么会多一帧）
del flame_data['frames'][-1]

# 遍历flame_params.json文件中的frames列表，将expression和pose字段的值赋值到w2c.json的frames列表中
for idx, frame in enumerate(flame_data['frames']):
    w2c_data['frames'][idx]['expression'] = frame['expression']
    w2c_data['frames'][idx]['pose'] = frame['pose']

w2c_data["shape_params"] = flame_data["shape_params"]

# 将合并后的结果输出为transforms.json
with open('transforms.json', 'w') as output_file:
    json.dump(w2c_data, output_file, indent=4)