import os
import json
import shutil
import re

def merge_directories(input_dirs, output_dir):
    image_output_dir = os.path.join(output_dir, 'ori_imgs')
    mask_output_dir = os.path.join(output_dir, 'mask')
    
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)
    
    image_counter = 1
    mask_counter = 1
    
    for input_dir in input_dirs:
        image_dir = os.path.join(input_dir, 'image')
        mask_dir = os.path.join(input_dir, 'mask')
        
        # 获取文件名列表并按数字顺序排序
        image_files = sorted([filename for filename in os.listdir(image_dir) if filename.endswith('.png')],
                             key=lambda x: int(re.findall(r'\d+', x)[0]))
        mask_files = sorted([filename for filename in os.listdir(mask_dir) if filename.endswith('.png')],
                            key=lambda x: int(re.findall(r'\d+', x)[0]))
        
        # 复制图像文件
        for filename in image_files:
            shutil.copy(os.path.join(image_dir, filename), os.path.join(image_output_dir, f"{image_counter}.png"))
            # print("image", filename, "copied as", f"{image_counter}.png")
            image_counter += 1
        
        # 复制掩码文件
        for filename in mask_files:
            shutil.copy(os.path.join(mask_dir, filename), os.path.join(mask_output_dir, f"{mask_counter}.png"))
            # print("mask", filename, "copied as", f"{mask_counter}.png")
            mask_counter += 1

def merge_flame_params(input_dirs, output_file):
    transformed_data = {
        "fx": 600.0,
        "fy": 600.0,
        "cx": 256.0,
        "cy": 256.0,
        "h": 512,
        "w": 512,
        "frames": [],
        "shape_params": []
    }

    image_id_counter = 1

    for input_dir in input_dirs:
        flame_params_file = os.path.join(input_dir, 'flame_params.json')

        with open(flame_params_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for frame in data["frames"]:
            img_id = image_id_counter

            transform_matrix = frame["world_mat"]
            transform_matrix.append([0.0, 0.0, 0.0, 1.0])
            transform_matrix[2][3] = 0.35

            transformed_frame = {
                "img_id": img_id,
                "aud_id": img_id,
                "transform_matrix": transform_matrix,
                "expression": frame["expression"],
                "pose": frame["pose"]
            }

            transformed_data["frames"].append(transformed_frame)

            image_id_counter += 1

    transformed_data["shape_params"] = data["shape_params"]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_dirs = ["MVI_1810", "MVI_1811", "MVI_1812", "MVI_1814"]
    output_dir = "merged_data"
    output_file = "transforms.json"
    
    merge_directories(input_dirs, output_dir)
    merge_flame_params(input_dirs, os.path.join(output_dir, output_file))
    
    print("Data merging completed successfully!")
