import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.lbs import lbs

class MyDeformNetwork(nn.Module):
    def __init__(self, W1=256, D1=3, expBasisW=150, poseBasisW=108, W2=64, D2=3, DS=3, DR=3):
        super(MyDeformNetwork, self).__init__()

        self.linear_back = nn.ModuleList(
            [nn.Linear(3, W1)] +
            [nn.Linear(W1, W1) for i in range(D1 - 1)])

        self.lbsW_net = nn.ModuleList(
            [nn.Linear(W1, W1)] + [nn.Linear(W1, 5)])

        self.jr_net = nn.ModuleList(
            [nn.Linear(W1, W1)] + [nn.Linear(W1, 5)])

        self.exp_basis_net = nn.Linear(W1, expBasisW)

        self.pose_basis_net = nn.Linear(W1, poseBasisW)

        self.delta_xyz_net = nn.Linear(W1, 3)

        self.linear_front = nn.ModuleList(
            [nn.Linear(9, W2)] +
            [nn.Linear(W2, W2) for i in range(D2 - 1)])

        self.scale_head = nn.ModuleList(
            [nn.Linear(W2, W2) for i in range(DS - 1)] +
            [nn.Linear(W2, 3)]
        )

        self.rotation_head = nn.ModuleList(
            [nn.Linear(W2, W2) for i in range(DR - 1)] +
            [nn.Linear(W2, 4)]
        )

    def forward(self, x, expression_params, pose_params):
        '''
          input:
            初始空间中的点位置x [Num, 3]
            表情参数expression_params [50]
            姿势参数pose_params [6]
          output:
            形变空间中点位置x_deformed [Num, 3]
            缩放偏移d_scaling [Num, 3]
            旋转偏移d_opacity [Num, 4]
        '''
        Num = x.shape[0]

        # 1 for batch size
        pose_input = pose_params.reshape(1, 15)  # [1 x 6]
        exp_input = expression_params.reshape(1, 50)  # [1 x 50]

        xyz_init = x  # [Num x 3]

        for i, l in enumerate(self.linear_back):
            x = self.linear_back[i](x)
            x = F.relu(x)

        joint_regressor = x
        for i, l in enumerate(self.jr_net):
            joint_regressor = self.jr_net[i](joint_regressor)
            joint_regressor = F.relu(joint_regressor)
        joint_regressor = joint_regressor.reshape(5, Num)  # [Num x 5] --> [5 x Num]

        lbs_weight = x
        for i, l in enumerate(self.lbsW_net):
            lbs_weight = self.lbsW_net[i](lbs_weight)
            lbs_weight = F.relu(lbs_weight)
        lbs_weight = F.softmax(lbs_weight, dim=1)  # [Num x 5]

        exp_basis = self.exp_basis_net(x).reshape(Num, 3, 50)  # [Num x 150]  -->  [Num x 3 x 50]

        # 36, 3*Num -> 63, 3*Num
        pose_basis = self.pose_basis_net(x).reshape(36, 3 * Num)  # [Num x 108] -->  [36 x 3Num]

        delta_xyz = self.delta_xyz_net(x)  # [Num x 3]

        xyz_canonical = xyz_init + delta_xyz  # [Num x 3]

        xyz_deformed = LBS(exp_input, pose_input, xyz_canonical,
                                         exp_basis, pose_basis, lbs_weight, joint_regressor).reshape(Num, 3)  # [Num x 3]

        front_x = torch.cat((xyz_deformed, xyz_init, delta_xyz), dim=1)
        for i, l in enumerate(self.linear_front):
            front_x = self.linear_front[i](front_x)
            front_x = F.relu(front_x)

        d_scaling = front_x
        for i, l in enumerate(self.scale_head):
            d_scaling = self.scale_head[i](d_scaling)
            d_scaling = F.relu(d_scaling)
        d_scaling = F.sigmoid(d_scaling)  # [Num, 3]
        
        d_rotation = front_x
        for i, l in enumerate(self.rotation_head):
            d_rotation = self.rotation_head[i](d_rotation)
            d_rotation = F.relu(d_rotation)
        d_rotation = F.normalize(d_rotation, p=2, dim=1)  # [Num, 4]

        return xyz_deformed / 100, d_scaling / 100, d_rotation / 100

def LBS(expression_params, pose_params, xyz_canonical, exp_basis, pose_basis, lbs_weight, joint_regressor):

    batch_size = 1

    # default_eyeball_pose = torch.zeros([1, 6], dtype=torch.float32, requires_grad=False, device="cuda")
    # eye_pose_params = default_eyeball_pose.expand(batch_size, -1)  # [1 x 6]
    # default_neck_pose = torch.zeros([1, 3], dtype=torch.float32, requires_grad=False, device="cuda")
    # neck_pose_params = default_neck_pose.expand(batch_size, -1)

    betas = expression_params

    full_pose = pose_params

    template_vertices = xyz_canonical.unsqueeze(0).expand(batch_size, -1, -1)  # [1 x Num x 3]

    parents = torch.tensor([-1, 0, 1, 1, 1], requires_grad=False)
    vertices, _ = lbs(
        betas, full_pose, template_vertices,
        exp_basis, pose_basis, joint_regressor, parents,
        lbs_weight, dtype=torch.float32, detach_pose_correctives=False
    )

    return vertices



if __name__ == "__main__":
    myNet = MyDeformNetwork()
    input = torch.randn(10000, 3).float()
    pose_params = torch.randn(15).float()
    expression_params = torch.randn(50).float()

    xyz_d, d_s, d_r = myNet(input, expression_params, pose_params)
    print("xyz", xyz_d.shape)
    print("d_s", d_s.shape)
    print("d_r", d_r.shape)





