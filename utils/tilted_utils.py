import torch

# randm to quaternion
# 从单位球面S^3上均匀随机采样，以生成四元数的组成部分。
def sample_uniform(key):
    # Uniformly sample over S^3.
    # Reference: http://planning.cs.uiuc.edu/node198.html
    key = key.detach().cpu().numpy()
    # 生成三个随机数
    u1, u2, u3 = torch.rand(3, generator=torch.Generator().manual_seed(int(key)))

    a = torch.sqrt(1.0 - u1)
    b = torch.sqrt(u1)

    # 构造出四元数
    return [a * torch.sin(u2),a * torch.cos(u2),b * torch.sin(u3),b * torch.cos(u3),]


# 将四元数转换为对应的3x3旋转矩阵
def as_matrix(x) -> torch.Tensor:
    # 计算四元数的范数
    norm = torch.norm(x, dim=-1)
    q = x * torch.sqrt(2.0 / norm.unsqueeze(-1))
    q = torch.bmm(q.unsqueeze(-1), q.unsqueeze(1))
    # 通过四元数与自身的外积构造旋转矩阵
    matrix = torch.cat(
        [
            1.0 - q[:, 2, 2] - q[:, 3, 3], q[:, 1, 2] - q[:, 3, 0], q[:, 1, 3] + q[:, 2, 0],
            q[:, 1, 2] + q[:, 3, 0], 1.0 - q[:, 1, 1] - q[:, 3, 3], q[:, 2, 3] - q[:, 1, 0],
            q[:, 1, 3] - q[:, 2, 0], q[:, 2, 3] + q[:, 1, 0], 1.0 - q[:, 1, 1] - q[:, 2, 2],
        ], dim=-1
        ).view(-1,3,3)
    return matrix.to(dtype=torch.float32, device='cuda')

# 特殊正交群SO(3)，即三维空间中的旋转群
class So3(torch.nn.Module):
    def __init__(self, num_points):
        super(So3, self).__init__()
        samples = []
        self.trans_num = 4
        # 4个随机采样的四元数样本 作为旋转参数
        for _ in range(self.trans_num):
             key = torch.randint(0, 2**32, (1,))
             sample = sample_uniform(key=key)
             samples.append(sample)

        samples = torch.tensor(samples)
        samples = samples.unsqueeze(0).expand(num_points, -1, -1)
        # 将参数存储在tau中，这些样本被用来扩展到num_points个点，以便对多个点分别进行旋转
        self.tau = torch.nn.Parameter(samples, requires_grad=True)  
        
    def forward(self, coords, iters): 
        R =[]
        # 对每一个旋转四元数，使用as_matrix函数将其转换为旋转矩阵Ri，并将这些矩阵堆叠起来形成R
        for i in range(self.trans_num):
                Ri = as_matrix(x=self.tau[:, i, :])
                R.append(Ri)
        R = torch.stack(R, dim=1)
        # "n transform i j, n j -> transform n j"
        # 将旋转矩阵R作用于坐标，返回旋转后的坐标
        transformed_coords = torch.einsum("nTij, nj -> Tnj", R, coords)  
        return transformed_coords 