import torch
from functorch import jacfwd, vmap

from utils.embedder_utils import *
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from pytorch3d.ops import knn_points
from flame.FLAME import FLAME
import torch.nn.functional as F

class ForwardDeformer(nn.Module):
    def __init__(self,
                FLAMEServer,
                d_in,
                dims,
                multires,
                num_exp=50,
                weight_norm=True,
                ghostbone=False,
                ):
        super().__init__()
        self.FLAMEServer = FLAMEServer
        print("creating deform network")
        # pose correctives, expression blendshapes and linear blend skinning weights
        d_out = 36 * 3 + num_exp * 3 + 3
        self.num_exp = num_exp
        dims = [d_in] + dims + [d_out]
        self.embed_fn = None

        # conf中multires=0
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch
            
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 网络前面的线性层部分
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            lin = lin.to(device)
            
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        
        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out).to(device)
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2]).to(device)
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5).to(device)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear).to(device)
        # initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)
        
        self.ghostbone = ghostbone
        
        self.leaky_relu = nn.LeakyReLU(0.1)  # 设置负斜率为0.1

        # 自己的MLP，推理d_scaling, d_rotation
        W2 = 128
        D2 = 3
        DS = 3
        DR = 3
        self.linear_front = nn.ModuleList(
            [nn.Linear(9, W2)] +
            [nn.Linear(W2, W2) for i in range(D2 - 1)]
        ).to(device)

        self.scale_head = nn.ModuleList(
            [nn.Linear(W2, W2) for i in range(DS - 1)] +
            [nn.Linear(W2, 3)]
        ).to(device)

        self.rotation_head = nn.ModuleList(
            [nn.Linear(W2, W2) for i in range(DR - 1)] +
            [nn.Linear(W2, 4)]
        ).to(device)
        
        print("deform network init success")

    def query_weights(self, pnts_c, mask=None):
        if mask is not None:
            pnts_c = pnts_c[mask]
        if self.embed_fn is not None:
            x = self.embed_fn(pnts_c)
        else:
            x = pnts_c
        
        # 线性层，激活函数用softplus
        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        # 推理得到blendshape posedirs shapedirs lbsweight
        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :36 * 3]
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
        lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
        lbs_weights = F.softmax(lbs_weights * 20, dim=-1)

        pnts_c_flame = pnts_c + blendshapes[:, -3:]

        return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5), pnts_c_flame, blendshapes[:, -3:]

    # 没有调用过
    def forward_lbs(self, pnts_c, pose_feature, betas, transformations, mask=None):
        shapedirs, posedirs, lbs_weights, pnts_c_flame, delta_xyz_init_to_canonical = self.query_weights(pnts_c, mask)
        pts_p = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32)
        return pts_p, pnts_c_flame, delta_xyz_init_to_canonical

    def forward(self, pnts_c, expression, flame_pose):
        expression = expression.unsqueeze(0)
        flame_pose = flame_pose.unsqueeze(0)

        batch_size = 1
        Num = pnts_c.shape[0]
        betas = expression.unsqueeze(1).expand(-1, Num, -1).reshape(Num, -1)  # Num x 50

        verts, pose_feature, transformations = self.FLAMEServer(expression_params=expression, full_pose=flame_pose)

        transformations = torch.cat(
            [torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)

        # >>>>>>> 注意 <<<<<<<
        pnts_c.requires_grad_(True)

        shapedirs, posedirs, lbs_weights, pnts_c_flame, delta_xyz_init_to_canonical = self.query_weights(pnts_c)

        shapedirs = shapedirs.expand(Num, -1, -1)
        posedirs = posedirs.expand(Num, -1, -1)
        lbs_weights = lbs_weights.expand(Num, -1)
        pnts_c_flame = pnts_c_flame.expand(Num, -1)

        knn_v = self.FLAMEServer.canonical_verts.unsqueeze(0).clone()
        flame_distance, index_batch, _ = knn_points(pnts_c_flame.unsqueeze(0), knn_v, K=1, return_nn=True)
        index_batch = index_batch.reshape(-1)

        pnts_d = self.FLAMEServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs,
                                              lbs_weights)

        pnts_d = pnts_d.reshape(-1)
        pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)

        front_x = torch.cat((pnts_c, pnts_d, delta_xyz_init_to_canonical), dim=1)

        for i, l in enumerate(self.linear_front):
            front_x = self.linear_front[i](front_x)
            front_x = self.leaky_relu(front_x)
            # front_x = self.softplus(front_x)

        d_scaling = front_x.clone()
        
        for i, l in enumerate(self.scale_head):
            d_scaling = self.scale_head[i](d_scaling)
            d_scaling = self.leaky_relu(d_scaling)
            # d_scaling = self.softplus(d_scaling)
        d_scaling = F.sigmoid(d_scaling)  # [Num, 3]

        d_rotation = front_x.clone()
        
        for i, l in enumerate(self.rotation_head):
            d_rotation = self.rotation_head[i](d_rotation)
            d_rotation = self.leaky_relu(d_rotation)
            # d_rotation = self.softplus(d_rotation)
        d_rotation = F.normalize(d_rotation, p=2, dim=1)  # [Num, 4]

        flame_loss_pack = {
            'index_batch': index_batch,
            'posedirs': posedirs,
            'shapedirs': shapedirs,
            'lbs_weights': lbs_weights,
            'flame_posedirs': self.FLAMEServer.posedirs,
            'flame_shapedirs': self.FLAMEServer.shapedirs,
            'flame_lbs_weights': self.FLAMEServer.lbs_weights
        }
        
        

        return pnts_d, d_scaling / 100, d_rotation, flame_loss_pack




if __name__ == "__main__":
    shape_params = torch.randn(1, 100).float().cuda()
    canonical_pose = 0.4
    mean_expression = torch.randn(1, 50).float().cuda()
    flameServer= FLAME('../flame/FLAME2020/generic_model.pkl', '../flame/FLAME2020/landmark_embedding.npy',
                             n_shape=100,
                             n_exp=50,
                             shape_params=shape_params,
                             canonical_expression=mean_expression,
                             canonical_pose=canonical_pose)

    flameServer.canonical_verts, flameServer.canonical_pose_feature, flameServer.canonical_transformations = \
        flameServer(expression_params=flameServer.canonical_exp, full_pose=flameServer.canonical_pose)
    flameServer.canonical_verts =flameServer.canonical_verts.squeeze(0)
    flameServer.canonical_transformations = torch.cat(
        [torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), flameServer.canonical_transformations], 1)

    d_in = 3
    dims = [128, 128, 128, 128]
    multires = 0
    num_exp = 50
    weight_norm = True
    ghostbone = True
    deformer_network = ForwardDeformer(flameServer, d_in, dims, multires, num_exp, weight_norm, ghostbone)

    batch_size = 1
    expression = torch.randn(1, 50).float()  # batch-size x 50
    flame_pose = torch.randn(1, 15).float()  # batch-size x 15


    pnts_c = torch.randn(10000, 3).float()  # Num x 3
    Num = pnts_c.shape[0]

    betas = expression.unsqueeze(1).expand(-1, Num, -1).reshape(Num, -1)  # Num x 50

    verts, pose_feature, transformations = flameServer(expression_params=expression, full_pose=flame_pose)

    transformations = torch.cat(
        [torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1).float().cuda(), transformations], 1)



    shapedirs, posedirs, lbs_weights, pnts_c_flame, delta_xyz_init_to_canonical = deformer_network.query_weights(pnts_c)

    shapedirs = shapedirs.expand(Num, -1, -1)
    posedirs = posedirs.expand(Num, -1, -1)
    lbs_weights = lbs_weights.expand(Num, -1)
    pnts_c_flame = pnts_c_flame.expand(Num, -1)

    pnts_d = flameServer.forward_pts(pnts_c_flame, betas, transformations, pose_feature, shapedirs, posedirs,
                                          lbs_weights)
    pnts_d = pnts_d.reshape(-1)


    pnts_d = pnts_d.reshape(-1, batch_size, 3).transpose(0, 1).reshape(-1, 3)
    print(pnts_d.shape)


'''
    shapedirs torch.Size([1, 3, 50])
    posedirs torch.Size([1, 36, 3])
    lbs_weights torch.Size([1, 6])
    pnts_c_flame torch.Size([1, 3])
    -----foward pts-----
    pnts_c_flame torch.Size([batch-size, 3])
    betas torch.Size([batch-size, 50])
    transformations torch.Size([batch-size, 6, 4, 4])
    pose_feature torch.Size([batch-size, 36])
    shapedirs torch.Size([batch-size, 3, 50])
    posedirs torch.Size([batch-size, 36, 3])
    lbs_weights torch.Size([batch-size, 6])
    # print("pnts_c_flame", pnts_c_flame.shape)
    # print("betas", betas.shape)
    # print("transformations", transformations.shape)
    # print("pose_feature", pose_feature.shape)
    # print("shapedirs", shapedirs.shape)
    # print("posedirs", posedirs.shape)
    # print("lbs_weights", lbs_weights.shape)
'''




