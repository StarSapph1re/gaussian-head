7  #
import os
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim, VGGPerceptualLoss, FLAMELoss
from gaussian_renderer import render
import sys
from scene import Scene
from scene.my_deform_model import MyDeformModel
from scene.tri_plane import TriPlaneModel
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import cv2

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, is_debug, novel_view, only_head):
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, is_debug, novel_view, only_head)
    exp_dims = scene.train_cameras[1.0][0].exp.size()

    deform = MyDeformModel(scene.shape_params, scene.mean_exp)
    deform.train_setting(opt)
    tilted = gaussians.tilted
    tilted.train_setting(opt)
    triplane = TriPlaneModel(opt.warm_up, dataset.sh_degree, tilted=tilted)
    triplane.train_setting(opt)
    # init vgg loss
    perceptual_vgg = VGGPerceptualLoss().to(device="cuda")

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0

    # FLAME loss
    flameLoss = FLAMELoss(scene.var_exp)

    # 可视化相关
    name = "train_render"
    video_name = os.path.join(dataset.model_path, name, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 512, 512
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    render_path = os.path.join(dataset.model_path, name)
    os.makedirs(render_path, exist_ok=True)

    print("warmup =", opt.warm_up)

    # --------------------

    progress_bar = tqdm(range(opt.iterations), desc="Training progress")


    # Alpha = np.linspace(0.000001, 0.1, num=deform_warm)

    for iteration in range(1, opt.iterations + 1):
        iter_start.record()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))  # random choose one to train
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        exp = viewpoint_cam.exp
        pose = viewpoint_cam.pose

        if iteration < opt.warm_up:
            xyz_deformed, d_scaling, d_rotation = gaussians.get_xyz, 0.0, 0.0
            flame_loss_pack = None
        else:
            xyz_deformed, d_scaling, d_rotation, flame_loss_pack = deform.step(gaussians.get_xyz, exp, pose)
            flame_loss_pack = None
        
            

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, triplane, pipe, background, xyz_deformed, d_rotation, d_scaling,
                               iteration)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re[
            "visibility_filter"], render_pkg_re["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        # loss funcs
        Ll1 = l1_loss(image, gt_image)
        Lvgg = perceptual_vgg(gt_image.unsqueeze(0), image.unsqueeze(0))
        Lssim = 1.0 - ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim + opt.lambda_vgg * Lvgg

        if flame_loss_pack:
            flame_loss = flameLoss.get_flame_loss(flame_loss_pack) * opt.lambda_flame
            loss += flame_loss
        else:
            flame_loss = 0.0

        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{6}f}",
                                          "L1loss": f"{Ll1:.{6}f}",
                                          # "ssimloss": f"{Lssim:.{6}f}",
                                          "flame": f"{flame_loss:.{6}f}",
                                          "VGGloss": f"{Lvgg:.{3}f}",
                                          "Num": f"{gaussians.get_xyz.shape[0]}"
                                          })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            if iteration >= 3000 and iteration % 5000 == 0:
                deform.printGrad(iteration)

            # -------------Data Visualization----------------

            if iteration > 0 and iteration < 5050:
                img_np = image.cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                img_umat = cv2.UMat(img_np)
                text = f"Iter: {iteration}"
                cv2.putText(img_umat, text, (width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                img_bgr = cv2.cvtColor(img_umat, cv2.COLOR_RGB2BGR)
                video_writer.write(img_bgr)
            elif iteration == 5050:
                video_writer.release()

            if (iteration >= 79950) or (iteration < 80000 and iteration % 1000 == 0):
                torchvision.utils.save_image(torch.cat((image, gt_image), dim=2), os.path.join(render_path, '{0:05d}'.format(iteration) + ".png"))

            # -------------Data Visualization----------------

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, triplane, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)
                triplane.save_weights(args.model_path, iteration)
                triplane.save_weights(args.model_path, iteration)
                tilted.save_weights(args.model_path, iteration)
                
            # if iteration % 2000 == 0:
            #     gaussians.printOp(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                   
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # org opacity threshold = 0.005
                    if iteration <= 3000:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold) 
                    else:
                        # 控制高斯数量
                        grad_threshold = opt.densify_grad_threshold - opt.densify_grad_threshold * 0.15 * iteration / opt.densify_until_iter
                        gaussians.densify_and_prune(grad_threshold, 0.005, scene.cameras_extent, size_threshold) 
                    print("gaussian num ---->", gaussians.get_xyz.shape[0])

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                triplane.optimizer.step()
                tilted.optimizer.step()

                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform_lr = deform.update_learning_rate(iteration)
                if iteration % 5000 == 0:
                    print("deform_lr=", deform_lr)

                triplane.optimizer.zero_grad()
                triplane.update_learning_rate(iteration)

                tilted.optimizer.zero_grad()
                tilted.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(lpargs, opargs, ppargs):
    if not lpargs.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        lpargs.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(lpargs.model_path))
    os.makedirs(lpargs.model_path, exist_ok=True)
    with open(os.path.join(lpargs.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(lpargs))) + '\n \n')

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(lpargs.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene,
                    triplane: TriPlaneModel, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        config = ({'name': 'test', 'cameras': scene.getTestCameras()})
        if config['cameras'] and len(config['cameras']) > 0:
            images = torch.tensor([], device="cuda")
            gts = torch.tensor([], device="cuda")
            for idx, viewpoint in enumerate(config['cameras']):
                if load2gpu_on_the_fly:
                    viewpoint.load2device()
                exp = viewpoint.exp
                pose = viewpoint.pose
                xyz = scene.gaussians.get_xyz
                
                xyz_deformed, d_scaling, d_rotation, _ = deform.step(xyz, exp, pose)
                # eval的时候把opacity设置成0.8看看是否能解决模糊问题
                scene.gaussians.setOp()
                
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, triplane, *renderArgs, xyz_deformed,
                                               d_rotation, d_scaling, iteration)["render"], 0.0, 1.0)

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                images = torch.cat((images, image.unsqueeze(0)), dim=0)
                gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                if idx % 10 == 0:
                    save_image = torch.cat((gt_image, image), dim=2)
                    save_path = os.path.join(scene.model_path, 'eval')
                    os.makedirs(save_path, exist_ok=True)
                    torchvision.utils.save_image(save_image,
                                                 os.path.join(save_path, '{}_{}'.format(iteration, idx) + ".png"))

                if load2gpu_on_the_fly:
                    viewpoint.load2device('cpu')
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                         image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                             gt_image[None], global_step=iteration)

            l1_test = l1_loss(images, gts)
            psnr_test = psnr(images, gts).mean()
            if config['name'] == 'test' or len(config[0]['cameras']) == 0:
                test_psnr = psnr_test
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int,
                        default=[30000, 50000] + list(range(80_000, op.iterations + 1, 10_000)))
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 10_000, 20_000, 30_000, 40_000] + list(
                            range(50_000, op.iterations + 1, 10_000)))
    # default=[4000, 5000, 40_000] + list(range(50_000, op.iterations + 1, 10_000)))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--is_debug", type=bool, default=False)
    parser.add_argument("--novel_view", type=bool, default=False)
    parser.add_argument("--only_head", type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.is_debug, args.novel_view, args.only_head)