import torch

from flame.FLAME import FLAME
from scene.deformer_network import ForwardDeformer
from utils.deform_utils import DeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class MyDeformModel:
    def __init__(self, shape_params, mean_expression, canonical_pose=0.4):
        FLAMEServer = FLAME('./flame/FLAME2020/generic_model.pkl', './flame/FLAME2020/landmark_embedding.npy',
                            n_shape=100,
                            n_exp=50,
                            shape_params=shape_params,
                            canonical_expression=mean_expression,
                            canonical_pose=canonical_pose)
        FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
            FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
        FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.squeeze(0)
        FLAMEServer.canonical_transformations = torch.cat(
            [torch.eye(4).unsqueeze(0).unsqueeze(0).float().cuda(), FLAMEServer.canonical_transformations], 1)

        d_in = 3
        dims = [128, 128, 128, 128]
        multires = 0
        num_exp = 50
        weight_norm = True
        ghostbone = True
        self.deform = ForwardDeformer(FLAMEServer, d_in, dims, multires, num_exp, weight_norm, ghostbone)
        self.optimizer = None
        # < 5
        self.spatial_lr_scale = 5

    def step(self, pnts_c, expression, flame_pose):
        return self.deform(pnts_c, expression, flame_pose)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final * 0.1,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)
        
    def printGrad(self, iteration):
        grad_dir = 'logs/grad'
        os.makedirs(grad_dir, exist_ok=True)
        grad_file = os.path.join(grad_dir, f'grad{iteration}.txt')
        with open(grad_file, 'w') as f:
            for name, param in self.deform.named_parameters():
                if param.grad is not None:
                    grad_mean = torch.mean(param.grad).item()
                    grad_var = torch.var(param.grad).item()
                    f.write(f'Layer: {name}, Grad_mean: {grad_mean}, Grad_var: {grad_var}\n')

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def print_model_parameters(self):
        Model_parameters = 0
        for name, module in self.deform.named_children():
            total_params = sum(p.numel() for p in module.parameters())
            print(f"{name} parameters: {total_params}")
            Model_parameters += total_params
        print(f'Total parameters: {Model_parameters}')

if __name__ == "__main__":
    shape_params = torch.randn(1, 100, device='cuda').float()
    canonical_pose = 0.4
    mean_expression = torch.randn(1, 50, device='cuda').float()
    my = MyDeformModel(shape_params, mean_expression, canonical_pose)
    op = OptimizationParams(parser)
    my.train_setting(op)
    for i in range(3000, 80001, 5000):
        print("iteration", i, ":", my.update_learning_rate(i))
    
    
#     pnts_c = torch.randn(10000, 3).float()
#     pose = torch.randn(15).float()
#     expression = torch.randn(50).float()

#     pnts_d, d_scaling, d_rotation = my.step(pnts_c, expression, pose)
#     print(pnts_d.shape)
#     print(d_scaling.shape)
#     print(d_rotation.shape)
#     my.print_model_parameters()
