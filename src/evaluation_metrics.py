import os
import lpips
import torch
import lpips
import pytorch_fid
import math
import numpy as np
import cv2
from utils import get_checkpoints_dir


def get_fid(cfg):
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

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


def get_dataset_paths(cfg):
    gt_dataset_path = os.path.join(
        cfg["datasets_dir"], cfg["dataset_type"], "test", "output"
    )
    gt_image_paths = [
        os.path.join(gt_dataset_path, gt_image_name)
        for gt_image_name in sorted(os.listdir(gt_dataset_path))
        if gt_image_name[-4:] == ".png"
    ]

    gen_dataset_path = os.path.join(get_checkpoints_dir(cfg), "final_images")
    gen_image_paths = [
        os.path.join(gen_dataset_path, gen_image_name)
        for gen_image_name in sorted(os.listdir(gen_dataset_path))
        if gen_image_name[-4:] == ".png"
    ]

    assert len(gt_image_paths) == len(
        gen_image_paths
    ), "Number of images in generated dataset should be equal to number of images in ground truth dataset."

    return gen_image_paths, gt_image_paths


def get_ssim(cfg):
    gen_image_paths, gt_image_paths = get_dataset_paths(cfg)

    ssim_sum = 0
    ssim_count = 0

    for gen_image_path, gt_image_path in zip(gen_image_paths, gt_image_paths):
        gen_image = cv2.imread(gen_image_path)
        gt_image = cv2.imread(gt_image_path)
        ssim_sum += ssim_multiple_channels(gen_image, gt_image)
        ssim_count += 1

    return ssim_sum / ssim_count


def get_lpips(cfg, lpips_type):
    # device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    gen_image_paths, gt_image_paths = get_dataset_paths(cfg)

    lpips_sum = 0
    lpips_count = 0

    if lpips_type == "alex":
        lpips_function = lpips.LPIPS(net="alex").float()
    elif lpips_type == "vgg":
        lpips_function = lpips.LPIPS(net="vgg").float()
    else:
        raise Exception(f"Not a valid lpips_function {lpips_function}.")

    def read_and_process_image(image_path):
        image = cv2.imread(image_path)[:, :, [2, 1, 0]]  # (H, W, RGB)
        image = np.transpose(image, (2, 0, 1))  # (RGB, H, W)
        image = (image / 255.0) * 2.0 - 1.0  # (RGB, H, W), between [-1, 1]
        image = (
            torch.tensor(image).unsqueeze(0).float()
        )  # (1, RGB, H, W), between [-1, 1]
        return image

    for gen_image_path, gt_image_path in zip(gen_image_paths, gt_image_paths):
        gen_image = read_and_process_image(gen_image_path)
        gt_image = read_and_process_image(gt_image_path)

        lpips_sum += float(lpips_function.forward(gen_image, gt_image))
        lpips_count += 1

    return lpips_sum / lpips_count


def save_evaluation_scores_of_final_images(cfg):
    if cfg["dataset_type"] == "face":
        scores = {"fid": get_fid(cfg)}
    elif cfg["dataset_type"] == "body_smplpix":
        scores = {
            "ssim": get_ssim(cfg),
            "lpips_alex": get_lpips(cfg, "alex"),
            "lpips_vgg": get_lpips(cfg, "vgg"),
        }
    else:
        raise Exception(f"Not a valid dataset_type {dataset_type}.")

    evaluation_scores_path = os.path.join(
        get_checkpoints_dir(cfg), "evaluation_scores.txt"
    )
    with open(evaluation_scores_path, "w") as writer:
        for key, value in scores.items():
            writer.write(f"{key}: {value}\n")
