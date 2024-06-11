import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Assuming utils.loss_utils and lpipsPyTorch are in the PYTHONPATH
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image)


def compute_metrics(render_dir, gt_dir):
    render_files = os.listdir(render_dir)
    gt_files = os.listdir(gt_dir)

    # Ensuring filenames in renders match those in gt
    assert set(render_files) == set(gt_files), "Mismatch in filenames between folders"

    l1_losses, ssims, psnrs, lpipss = [], [], [], []

    for filename in render_files:
        img_path = os.path.join(render_dir, filename)
        gt_path = os.path.join(gt_dir, filename)

        img = load_image(img_path)
        gt = load_image(gt_path)

        # Calculate metrics
        l1_losses.append(l1_loss(img, gt).item())
        ssims.append(ssim(img, gt).item())
        psnrs.append(psnr(img, gt).item())
        lpipss.append(lpips(img, gt).item())

    # Compute average for each metric
    avg_l1_loss = sum(l1_losses) / len(l1_losses)
    avg_ssim = sum(ssims) / len(ssims)
    avg_psnr = sum(psnrs) / len(psnrs)
    avg_lpips = sum(lpipss) / len(lpipss)

    print(f"Average L1 Loss: {avg_l1_loss}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average LPIPS: {avg_lpips}")


# Set the directories here
render_dir = 'path/to/renders'
gt_dir = 'path/to/gt'

compute_metrics(render_dir, gt_dir)
