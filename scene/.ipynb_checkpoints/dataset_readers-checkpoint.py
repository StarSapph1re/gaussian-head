import os
import torch
from tqdm import tqdm
from PIL import Image
from typing import NamedTuple, Optional
from utils.graphics_utils import getWorld2View2, focal2fov
import numpy as np
import json
import cv2 as cv
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

from scipy.spatial.transform import Slerp, Rotation


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    exp: np.array
    pose: np.array
    depth: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    mean_exp: torch.tensor
    var_exp: torch.tensor
    shape_params: torch.tensor


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, pose


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                        vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readNerfBlendShapeCameras(path, is_eval, is_debug, novel_view, only_head, reenact_path=None):
    '''
    需要用到transform.json中的以下信息:
      1. h w 图像的高度和宽度 512 512
      2. fx fy 焦距
      3. cx cy 图像的中心点 256 256
      4. mask
      5. transform matrix!
      6. exp_ori
      near far face_rect exp都是多余的
    '''
    with open(os.path.join(path, "transforms.json"), 'r') as f:
        meta_json = json.load(f)
        
    print("reenact_path in read =", reenact_path)


    # --------------------for reenact task--------------------
    if reenact_path is not None:
        with open(os.path.join(reenact_path, "transforms.json"), 'r') as f:
            re_json = json.load(f)
            print("Load expression parameter from ", reenact_path)
            re_frames = re_json['frames']
            # re_frames = re_frames[-50:]  # 取最后50帧做reenact
    # --------------------for reenact task--------------------
    
    # 新数据集 -50
    test_frames = -50
    frames = meta_json['frames']
    total_frames = len(frames)
    
    if not is_eval:
        print(f'Loading train dataset from {path}...')
        frames = frames[0 : (total_frames + test_frames)]
        if is_debug:
            frames = frames[0: 50]
    else:
        print(f'Loading test dataset from {path}...')
        frames = frames[-50:]
        if is_debug:
            frames = frames[-50:]

    cam_infos = []
    h, w = meta_json['h'], meta_json['w']
    fx, fy, cx, cy = meta_json['fx'], meta_json['fy'], meta_json['cx'], meta_json['cy']
    fovx = focal2fov(fx, pixels=w)
    fovy = focal2fov(fy, h)

    # --------------------for reenact task--------------------
    if reenact_path is not None:
        for idx, frame in enumerate(tqdm(re_frames, desc="Loading reenactment camera into memory in advance")):
            image_id = frame['img_id']
            image_path = os.path.join(reenact_path, "ori_imgs", str(image_id+1) + '.png')
            image = np.array(Image.open(image_path))
            if not only_head:
                mask_path = os.path.join(reenact_path, "mask", str(image_id+1) + '.png')
                seg = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                mask = np.repeat(np.asarray(seg)[:, :, None], 3, axis=2) / 255
            else:
                mask_path = os.path.join(reenact_path, "parsing", str(image_id) + '.png')
                seg = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
                if seg.shape[-1] == 3:
                    seg = cv.cvtColor(seg, cv.COLOR_BGR2RGB)
                else:
                    seg = cv.cvtColor(seg, cv.COLOR_BGRA2RGBA)
                mask = (seg[:, :, 0] == 0) * (seg[:, :, 1] == 0) * (seg[:, :, 2] == 255)
                mask = np.repeat(np.asarray(mask)[:, :, None], 3, axis=2)

            white_background = np.ones_like(image) * 255
            image = Image.fromarray(np.uint8(image * mask + white_background * (1 - mask)))

            expression = np.array(frame['expression'])
            pose = np.array(frame['pose'])

            # MICA推理的pose长度为6，使用默认eyeball和neck pose
            default_eyeball_pose = np.zeros(6)
            default_neck_pose = np.zeros(3)
            if pose.shape[0] == 6:
                pose = np.concatenate([pose[:3], default_neck_pose, pose[3:], default_eyeball_pose])
                if idx == 1:
                    print("pose.shape=", pose.shape)

            # transform_matrix就选训练集中第一个作为参考
            c2w = np.array(frames[0]['transform_matrix'])
            c2w[:3, 1:3] *= -1
            # 世界坐标系到相机坐标系的变换
            w2c = np.linalg.inv(c2w)
            # 提取旋转部分
            R = np.transpose(w2c[:3, :3])
            # 提取平移部分
            T = w2c[:3, 3]

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                        image_name=image_id, width=image.size[0], height=image.size[1], exp=expression,
                                        fid=image_id, pose=pose))

    # --------------------for reenact task--------------------

    else:
        for idx, frame in enumerate(tqdm(frames, desc="Loading data into memory in advance")):
            image_id = frame['img_id']
            # 新数据集 .jpg改为.png
            image_path = os.path.join(path, "ori_imgs", str(image_id)+'.png')
            image = np.array(Image.open(image_path))
            if not only_head:
                # 新数据集 image_id+1改为image_id
                mask_path = os.path.join(path, "mask", str(image_id)+'.png')
                seg = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                # Reference MODNet colab implementation
                mask = np.repeat(np.asarray(seg)[:,:,None], 3, axis=2) / 255
            else:
                mask_path = os.path.join(path, "parsing", str(image_id)+'.png')
                seg = cv.imread(mask_path, cv.IMREAD_UNCHANGED)
                if seg.shape[-1] == 3:
                    seg = cv.cvtColor(seg, cv.COLOR_BGR2RGB)
                else:
                    seg = cv.cvtColor(seg, cv.COLOR_BGRA2RGBA)
                mask=(seg[:,:,0]==0)*(seg[:,:,1]==0)*(seg[:,:,2]==255)
                mask = np.repeat(np.asarray(mask)[:,:,None], 3, axis=2)

            white_background = np.ones_like(image)* 255
            image = Image.fromarray(np.uint8(image * mask + white_background * (1 - mask)))

            expression = np.array(frame['expression'])
            pose = np.array(frame['pose'])

            if novel_view:
                vec=np.array([0,0,0.3493212163448334])
                rot_cycle=100
                tmp_pose=np.identity(4,dtype=np.float32)
                r1 = Rotation.from_euler('y', 15+(-30)*((idx % rot_cycle)/rot_cycle), degrees=True)
                tmp_pose[:3,:3]=r1.as_matrix()
                trans=tmp_pose[:3,:3]@vec
                tmp_pose[0:3,3]=trans
                c2w = tmp_pose
            else:
                c2w = np.array(frame['transform_matrix'])
            c2w[:3, 1:3] *= -1
            # 世界坐标系到相机坐标系的变换
            w2c = np.linalg.inv(c2w)
            # 提取旋转部分
            R = np.transpose(w2c[:3,:3])
            # 提取平移部分
            T = w2c[:3, 3]



            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovX=fovx, FovY=fovy, image=image, image_path=image_path,
                                        image_name=image_id, width=image.size[0], height=image.size[1], exp=expression,
                                        fid=image_id, pose=pose))



    '''finish load all data'''
    exp_list = [torch.tensor(info.exp) for info in cam_infos]
    # 将exp列表转换为一个大张量，形状为(num_frames, 50)
    exp_tensor = torch.stack(exp_list)

    # 计算表情参数的均值和方差
    exp_mean = torch.mean(exp_tensor, 0, keepdim=True)
    exp_var = torch.var(exp_tensor, 0, keepdim=True)
    # 读取shape参数
    shape = torch.tensor(meta_json['shape_params']).float().unsqueeze(0)
    return cam_infos, exp_mean, exp_var, shape

def readNeRFBlendShapeDataset(path, eval, is_debug, novel_view, only_head, reenact_path=None):
    print("Load NeRFBlendShape Train Dataset")
    train_cam_infos, mean_exp_train, var_exp_train, shape_params = readNerfBlendShapeCameras(path=path, is_eval=False, is_debug=is_debug, 
                                                                                             novel_view=novel_view, only_head=only_head, reenact_path=reenact_path)
    print("Load NeRFBlendShape Test Dataset")
    test_cam_infos, mean_exp_test, var_exp_test, shape_params = readNerfBlendShapeCameras(path=path, is_eval=eval, is_debug=is_debug, 
                                                                                          novel_view=novel_view, only_head=only_head, reenact_path=reenact_path)


    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []


    if len(train_cam_infos) != 0:
        nerf_normalization = getNerfppNorm(train_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(test_cam_infos)


    ply_path = os.path.join(path, "points3d.ply")
    '''Init point cloud'''
    if not os.path.exists(ply_path):
        # Since mono dataset has no colmap, we start with random points
        num_pts = 10_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd, 
                           train_cameras=train_cam_infos, 
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization, 
                           mean_exp=mean_exp_train,
                           var_exp=var_exp_train,
                           shape_params=shape_params,
                           ply_path=ply_path)

    return scene_info


sceneLoadTypeCallbacks = {"nerfblendshape":readNeRFBlendShapeDataset,}
