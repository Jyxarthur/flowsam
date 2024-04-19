import os
import cv2
import json
import torch
import numpy as np
from torch.nn import functional as F


def readRGB(sample_dir, gt_resolution = None):
    rgb = cv2.imread(sample_dir)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if gt_resolution is not None:
        rgb = cv2.resize(rgb, (gt_resolution[1], gt_resolution[0]), interpolation=cv2.INTER_LINEAR)
    return rgb



def processMultiSeg(sample_dir, gt_resolution = None, out_channels = 10, dataset = 'dvs17m', with_bg = False):
    rgb_anno = cv2.imread(sample_dir)
    rgb_anno = cv2.cvtColor(rgb_anno, cv2.COLOR_BGR2RGB)
    img = rgb_anno
    colors = []

    with open('data/color_palette.json') as f:
        color_dict = json.load(f)
    if dataset in color_dict.keys():    
        colors = color_dict[dataset]
    else:
        colors = [[0,0,0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [191, 0, 0], [64, 128, 0]]
    colors = colors[0 : min(len(colors), out_channels)]
    
    masks = []
    for color in colors:
        offset = np.broadcast_to(np.array(color), (img.shape[0], img.shape[1], 3))
        mask = (np.mean(offset == img, 2) == 1).astype(np.float32)
        mask =  np.repeat(mask[:, :, np.newaxis], 3, 2)
        masks.append(mask)
    for j in range(out_channels):
        masks.append(np.zeros((img.shape[0], img.shape[1], 3)))
    masks_raw = masks[0 : out_channels]
    masks_float = []
    for i, mask in enumerate(masks_raw):
        if gt_resolution is not None:
            mask_float = (cv2.resize(mask, (gt_resolution[1], gt_resolution[0]), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)
        else:
            mask_float = mask
        masks_float.append(mask_float)
    if with_bg:
        masks_float = np.stack(masks_float, 0)[:, :, :, 0]
    else:
        masks_float = np.stack(masks_float, 0)[1:, :, :, 0]
    return masks_float


def preprocess(x: torch.Tensor) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean =  torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x