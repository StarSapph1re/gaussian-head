import numpy as np
import torch
import os
import json
# 把metrical tracker的transform matrix提取出来写入c2w.json
org = {
    "fx": 2401,
    "fy": 2401,
    "cx": 256.00,
    "cy": 256.00,
    "h": 512,
    "w": 512,
    "frames": []
}

def rev(R, T):
    # 创建 w2c 矩阵，首先用 R 的转置和 T 初始化
    w2c = np.eye(4)  # 创建一个单位矩阵
    w2c[:3, :3] = R.T  # 设置旋转部分为 R 的转置
    w2c[:3, 3] = T  # 设置平移部分

    # 计算 c2w 矩阵，即 w2c 的逆
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1
    return c2w

if __name__ == "__main__":
    ckpt_dir = "checkpoint"
    N_frames = len(os.listdir(ckpt_dir))
    for frame_id in range(N_frames):
        image_name_mica = str(frame_id).zfill(5)  # obey mica tracking

        ckpt_path = os.path.join(ckpt_dir, image_name_mica + '.frame')
        payload = torch.load(ckpt_path)

        opencv = payload['opencv']
        w2cR = opencv['R'][0]
        w2cT = opencv['t'][0]
        R = np.transpose(w2cR)  # R is stored transposed due to 'glm' in CUDA code
        T = w2cT
        f = {
            "img_id": frame_id + 1,
            "transform_matrix": rev(R, T).tolist()
        }
        org["frames"].append(f)


    with open("w2c.json", "w") as json_file:
        json.dump(org, json_file, indent=4)

    print("Data has been written to c2w.json!")