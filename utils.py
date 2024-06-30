import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def is_bg_mask(
    masks_fil, thres = 0.5
):  
    masks_edge = (masks_fil[:, 0].mean(-1) + masks_fil[:, -1].mean(-1) + masks_fil[:, :, 0].mean(-1) + masks_fil[:, :, -1].mean(-1))/4 
    return masks_edge > thres


def iou(masks, gt, thres=0.5, emp=True):
    """ IoU predictions """
    if isinstance(masks, torch.Tensor): # for tensor inputs
        masks = (masks>thres).float()
        gt = (gt>thres).float()
        intersect = (masks * gt).sum(dim=[-2, -1])
        union = masks.sum(dim=[-2, -1]) + gt.sum(dim=[-2, -1]) - intersect
        empty = (union < 1e-6).float()
        iou = torch.clip(intersect/(union + 1e-12) + empty, 0., 1.)
        return iou
    else: # for numpy inputs
        masks = (masks>thres)
        gt = (gt>thres)
        intersect = (masks * gt).sum((-1,-2))
        union = masks.sum((-1,-2)) + gt.sum((-1,-2)) - intersect
        empty = (union < 1e-6) if emp else 0
        iou = np.clip(intersect/(union + 1e-12) + empty, 0., 1.)
        return iou



def imwrite_indexed(filename, array, color_palette):
    """ Save indexed png """
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')
    

def save_indexed(filename, img):
    """ Save image with given colour palette """
    color_palette = np.array([[0,0,0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [191, 0, 0], [64, 128, 0]]).astype(np.uint8)
    imwrite_indexed(filename, img, color_palette)


def is_box_near_image_edge(
    boxes, orig_box, atol: float = 20.0
):
    """Filter masks at the edge of a crop, but not at the edge of the original image."""
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    return torch.any(near_image_edge, dim=1)


def filter_data(data_list, condition, is_idx = False):
    """Filter data according to condition provided"""
    """ is_idx = True represents that the conditions are given as the index in the tensor"""
    """ is_idx = False represents that the condtions are binary masks"""
    data_fil_list = []
    for i, data in enumerate(data_list):
        if is_idx:
            if condition is None:
                data_fil = torch.zeros_like(data[0:1], device=data.device)
            else:
                data_fil = data[torch.as_tensor(condition, device=data.device)]
        else:
            data_fil_tmp = [a for i, a in enumerate(data) if condition[i]]
            if len(data_fil_tmp) == 0:
                data_fil = torch.zeros_like(data[0:1], device=data.device)
            else:
                data_fil = torch.stack(data_fil_tmp, 0)
        data_fil_list.append(data_fil)
    return data_fil_list

def hard_thres(masks, ious, output_savemask = False):
    """ Hard thresholding (overlaying) the masks according to IoUs (Scores)"""
    masks_np = masks.detach().cpu().numpy()
    ious_np = ious.detach().cpu().numpy()
    saveidxs_np = np.arange(masks.shape[0])
    ious_rank = np.argsort(ious_np)

    output_mask = np.copy(masks_np[0]) * 0.
    for score_idx in ious_rank:
        output_mask = output_mask * (1 - masks_np[score_idx]) + masks_np[score_idx] * (saveidxs_np[score_idx] + 1)
    mask_out = np.clip(output_mask, 0, masks.shape[0])

    masks_out_torch = []
    for obj_idx in range(1, masks.shape[0] + 1):
        mask_torch = torch.from_numpy(mask_out == obj_idx).float().cuda()
        masks_out_torch.append(mask_torch)
    masks_out_torch = torch.stack(masks_out_torch, 0)
    if output_savemask: # Optionally output the mask for saving
        return masks_out_torch, mask_out
    else:
        return masks_out_torch
       
def update_iousummary(masks_hung, masks_nonhung, anno, num_obj, path, iou_summary, save_path = None):
    # Updating the performance
    for obj_idx in range(1, num_obj + 1):
        iou_summary[os.path.dirname(path[0])][obj_idx - 1].append(iou(masks_hung[obj_idx - 1], anno[0, obj_idx - 1]).item())
    if save_path is not None:
        save_path_hung = os.path.join(save_path, "hung")
        os.makedirs(os.path.dirname(os.path.join(save_path_hung, path[0])), exist_ok = True)
        save_path_nonhung = os.path.join(save_path, "nonhung")
        os.makedirs(os.path.dirname(os.path.join(save_path_nonhung, path[0])), exist_ok = True)
        # Saving Hungarian matched masks
        masks_hung_np = masks_hung.detach().cpu().numpy()
        saved_mask_hung = np.copy(masks_hung_np[0]) * 0.
        for save_idx in range(masks_hung_np.shape[0]):
            saved_mask_hung = saved_mask_hung * (1 - masks_hung_np[save_idx]) + masks_hung_np[save_idx] * (save_idx + 1)
        saved_mask_hung = np.clip(saved_mask_hung, 0, masks_hung_np.shape[0])
        save_indexed(os.path.join(save_path_hung, path[0]), saved_mask_hung.astype(np.uint8))
        # Saving Non-Hungarian masks
        masks_nonhung_np = masks_nonhung.detach().cpu().numpy()
        saved_mask_nonhung = np.copy(masks_nonhung_np[0]) * 0.
        for save_idx in range(masks_nonhung_np.shape[0]):
            saved_mask_nonhung = saved_mask_nonhung * (1 - masks_nonhung_np[save_idx]) + masks_nonhung_np[save_idx] * (save_idx + 1)
        saved_mask_nonhung = np.clip(saved_mask_nonhung, 0, masks_nonhung_np.shape[0])
        save_indexed(os.path.join(save_path_nonhung, path[0]), saved_mask_nonhung.astype(np.uint8))
    return iou_summary

def remove_overlapping_masks(masks, filter=False):
    result_mask = masks.copy()
    for i in range(result_mask.shape[0] - 1):
        result_mask[i+1:] = np.logical_and(result_mask[i+1:], np.logical_not(masks[i]))
    result_mask = np.clip(result_mask, 0 ,1)
    return result_mask

def warp_flow(curImg, flow):
    H, W = np.shape(curImg)
    flow = cv2.resize(flow, (W, H))
    h, w = flow.shape[:2]
    flow = flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    prevImg = cv2.remap(curImg, flow, None, cv2.INTER_LINEAR)
    return prevImg

def hungarian_iou(masks, gt, thres = 0.5, emp=False):
    masks = (masks>thres)
    gt = (gt>thres)
    ious = iou(gt[:, None], masks[None], emp=False)
    g, p = np.shape(ious)

    orig_idx, hung_idx = linear_sum_assignment(-ious)
    out = ious[orig_idx, hung_idx]
    return out, masks[hung_idx] * 1.

def seq_hungarian_iou(masks, gts, thres = 0.5):
    g, h, w = np.shape(gts[0])
    p = np.shape(masks[0])[0]
    ious = np.zeros([g, 20])
    num_seq = len(gts)
    for i in range(num_seq):
        ious = ious + iou(gts[i][:, None], np.concatenate([masks[i], np.zeros([20-p,h,w])], 0)[None])
    orig_idx, hung_idx = linear_sum_assignment(-ious)
    out = ious[orig_idx, hung_idx] / num_seq
    return out, 0
