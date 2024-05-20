import torch

from utils.my_deform_utils import MyDeformNetwork
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func

'''
学习率：
  position_lr_init
  position_lr_final
  position_lr_delay_mult
  deform_lr_max_steps
'''
class MyDeformModel:
    def __init__(self):
        self.deform = MyDeformNetwork().cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, exp_params, pose_params):
        return self.deform(xyz, exp_params, pose_params)

    def train_setting(self, training_args):
        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.position_lr_init * self.spatial_lr_scale,
             "name": "deform"}]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=training_args.position_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=training_args.deform_lr_max_steps)

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
    myModel = MyDeformModel()
    '''
    xyz = torch.randn(10000, 3).float().cuda()
    exp_param = torch.randn(50).float().cuda()
    pose_param = torch.randn(6).float().cuda()
    xyz_deformed, d_scaling, d_rotation = myModel.step(xyz, exp_param, pose_param)
    print("xyz", xyz_deformed.shape)
    print("d_s", d_scaling.shape)
    print("d_rotation", d_rotation.shape)
    '''
    myModel.print_model_parameters()


