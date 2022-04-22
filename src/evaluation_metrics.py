import os
import lpips
import torch
import pytorch_fid
import math
import numpy as np
import cv2
from utils import get_checkpoints_dir


def get_fid(cfg):
    if cfg["fid_device"] is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(cfg["fid_device"])

    if cfg["fid_num_workers"] is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = cfg["fid_num_workers"]

    # Ground truth test dataset
    gt_dataset_path = os.path.join(
        cfg["datasets_dir"], cfg["dataset_type"], "test", "output"
    )

    # generated dataset
    gen_dataset_path = os.path.join(get_checkpoints_dir(cfg), "final_images")

    paths = [gt_dataset_path, gen_dataset_path]

    fid_value = pytorch_fid.calculate_fid_given_paths(
        paths, cfg["fid_batch_size"], device, cfg["fid_dims"], num_workers
    )
    return fid_value


def ssim_one_channel(img1, img2):
    """
    Taken from https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def ssim_multiple_channels(img1, img2):
    """calculate SSIM
    Taken from https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_one_channel(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim_one_channel(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


def get_ssim(cfg):
    gt_dataset_path = os.path.join(
        cfg["datasets_dir"], cfg["dataset_type"], "test", "output"
    )
    gt_image_paths = [
        os.path.join(gen_dataset_path, gt_image_name)
        for gt_image_name in os.listdir(gt_dataset_path)
        if gt_image_name[-4:] == ".png"
    ]

    gen_dataset_path = os.path.join(get_checkpoints_dir(cfg), "final_images")
    gen_image_paths = [
        os.path.join(gen_dataset_path, gt_image_name)
        for gt_image_name in os.listdir(gt_dataset_path)
        if gt_image_name[-4:] == ".png"
    ]

    assert len(gt_image_paths) == len(
        gen_image_paths
    ), "Number of images in generated dataset should be equal to number of images in ground truth dataset."

    ssim_sum = 0
    ssim_count = 0

    for gen_image_path, gt_image_path in zip(gen_image_paths, gt_image_paths):
        gen_image = cv2.imread(gen_image_path)
        gt_image = cv2.imread(gt_image_path)
        ssim_sum += ssim_multiple_channels(gen_image, gt_image)
        ssim_count += 1

    return ssim_sum / ssim_count


# import lpips
# loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

# import torch
# img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
# img1 = torch.zeros(1,3,64,64)
# d = loss_fn_alex(img0, img1)