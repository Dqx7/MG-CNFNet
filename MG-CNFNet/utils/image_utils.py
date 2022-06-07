import torch
import numpy as np
import cv2
# from skimage.measure import compare_ssim
import math

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

def torchRMSE(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    return rmse

# def torchSSIM(tar_img, prd_img, cr1):
#     # ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
#     ssim_pre, ssim_map = compare_ssim(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
#     ssim_map = ssim_map * cr1
#     r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
#     win_size = 2 * r + 1
#     pad = (win_size - 1) // 2
#     ssim = ssim_map[pad:-pad,pad:-pad,:]
#     crop_cr1 = cr1[pad:-pad,pad:-pad,:]
#     ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
#     ssim = np.mean(ssim)
#     return ssim