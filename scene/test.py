import torch

exp_list = [torch.tensor([1, 2, 3]).float(),
            torch.tensor([4, 5, 6]).float()]

exp_tensor = torch.stack(exp_list)

# 计算exp张量沿着第一个维度（num_frames）的平均值
exp_mean = torch.mean(exp_tensor, 0, keepdim=True)
exp_var = torch.var(exp_tensor, 0, keepdim=True)
print(exp_mean.shape)
print(exp_var.shape)