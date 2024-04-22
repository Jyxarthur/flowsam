import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from argparse import ArgumentParser
from torchvision.ops.boxes import batched_nms
from scipy.optimize import linear_sum_assignment
from segment_anything.utils.amg import batched_mask_to_box, calculate_stability_score

from model.model_config import config_model
from data.dataset_config import config_eval_dataloader
from utils import iou, is_bg_mask, save_indexed, update_iousummary, filter_data, hard_thres


def run_flowpsam(args, flowsam, info):
    with torch.no_grad():
        # Inputs
        original_size = (info["size"][0][0].item(), info["size"][1][0].item())
        input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
        flow_image = info["flow_image"].cuda()   # 1 4 3 1024 1024
        rgb_image = info["rgb_image"].cuda()  # 1 3 1024 1024
        grid_coords_set = info["grid"].cuda().squeeze(0) # 100 1 2
        # Inference with iterative point prompt inputs
        masks_set = []
        scores_set = []
        flowsam.rgb_feature = None
        flowsam.flow_feature = None
        for coords_idx in range(grid_coords_set.shape[0] // 10):
            grid_coords = grid_coords_set[coords_idx * 10 : coords_idx * 10 + 10]
            point_labels = torch.ones(grid_coords.size()[:2], dtype=torch.int, device=grid_coords.device)
            point_prompts = (grid_coords, point_labels)
            masks_logit, fiou, mos = flowsam(rgb_image, flow_image, point_prompts, use_cache = True)  
            fiou = fiou[:, args.sam_channel]
            mos = mos[:, 0]
            score = fiou + mos
            masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
            masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)
            masks = (masks_logit > args.mod_thres).float()
            masks = masks[:, args.sam_channel]
            masks_set.append(masks)
            scores_set.append(score)
    masks_set = torch.cat(masks_set, 0)
    scores_set = torch.cat(scores_set, 0)
    boxes_set = batched_mask_to_box(masks_set.long()).float()
    return masks_set, scores_set, boxes_set


def run_flowisam(args, flowsam, info):
    with torch.no_grad():
        # Inputs
        original_size = (info["size"][0][0].item(), info["size"][1][0].item())
        input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
        flow_image = info["flow_image"].cuda()   # 1 4 3 1024 1024
        grid_coords_set = info["grid"].cuda().squeeze(0) # 100 1 2
        # Inference with iterative point prompt inputs
        masks_set = []
        scores_set = []
        flowsam.flow_feature = None
        for coords_idx in range(grid_coords_set.shape[0] // 10):
            grid_coords = grid_coords_set[coords_idx * 10 : coords_idx * 10 + 10]
            point_labels = torch.ones(grid_coords.size()[:2], dtype=torch.int, device=grid_coords.device)
            point_prompts = (grid_coords, point_labels)
            masks_logit, fiou = flowsam(flow_image, point_prompts, use_cache = True)  
            fiou = fiou[:, args.sam_channel]
            score = fiou 
            masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
            masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)
            masks = (masks_logit > args.mod_thres).float()
            masks = masks[:, args.sam_channel]
            masks_set.append(masks)
            scores_set.append(score)
    masks_set = torch.cat(masks_set, 0)
    scores_set = torch.cat(scores_set, 0)
    boxes_set = batched_mask_to_box(masks_set.long()).float()
    return masks_set, scores_set, boxes_set


def eval(args, val_loader, flowsam):
    print("")
    print("---Evaluation steps {}".format(args.model))
    flowsam.eval()
    iou_summary = {}
    for idx, info in enumerate(val_loader):
        if idx % 100 == 0:
            print("---Inference step: {}".format(idx))

        # Set up performance logger
        if os.path.dirname(info["path"][0]) not in iou_summary.keys() and ("num_obj" in info.keys()):
            iou_summary[os.path.dirname(info["path"][0])] = {}
            for obj_idx in range(info["num_obj"].item()):
                iou_summary[os.path.dirname(info["path"][0])][obj_idx] = []
        
        # Running model
        if args.model == "flowpsam":
            masks_set, scores_set, boxes_set = run_flowpsam(args, flowsam, info)
        else: #flowisam
            masks_set, scores_set, boxes_set = run_flowisam(args, flowsam, info)

        """
        Post-processing
        """
        if "anno" in info.keys():  
            anno = info["anno"].cuda()  # 1 C H W
        else: # No GT
            anno = torch.zeros(1, 1) # empty array with anno.shape[1]=1
        
                
        # NMS
        keep_idx = batched_nms(boxes_set, scores_set, torch.zeros_like(boxes_set[:, 0]), iou_threshold=0.9)
        masks_fil, scores_fil, boxes_fil = filter_data([masks_set, scores_set, boxes_set], keep_idx, is_idx = True)
        
        # Removing bg masks
        keep_maskidx = ~is_bg_mask(masks_fil)
        masks_fil, scores_fil, boxes_fil = filter_data([masks_fil, scores_fil, boxes_fil], keep_maskidx)
 
        # Ordering masks according to the scores
        sel_idxs = torch.argsort(scores_fil, descending = True)
        scores = (scores_fil[sel_idxs])[0:max(args.max_obj, anno.shape[1])]
        masks_nonhung = (masks_fil[sel_idxs])[0:max(args.max_obj, anno.shape[1])]
        # Overlaying masks
        masks_nonhung, saved_mask_nonhung = hard_thres(masks_nonhung, scores, output_savemask = True)
        # Padding masks to match with num_obj
        if masks_nonhung.shape[0] < max(args.max_obj, anno.shape[1]):
            masks_nonhung_pad = torch.repeat_interleave(torch.zeros_like(masks_nonhung[0:1], device = masks_nonhung.device), max(args.max_obj, anno.shape[1]) - masks_nonhung.shape[0], 0)
            masks_nonhung = torch.cat([masks_nonhung, masks_nonhung_pad], 0)
            scores_pad = torch.zeros(max(args.max_obj, anno.shape[1]) - masks_nonhung.shape[0]).cuda()
            scores = torch.cat([scores, scores_pad], 0)
        
        if "anno" in info.keys():    
            # Hungarian matching and result summary
            result_iou = iou(anno[0, :, None], masks_nonhung[None])
            orig_idx, hung_idx = linear_sum_assignment(-result_iou.cpu().detach().numpy())
            masks_hung = masks_nonhung[hung_idx]  # Hungarian matched masks
            iou_summary = update_iousummary(masks_hung, masks_nonhung, anno, info["num_obj"].item(), info["path"], iou_summary, save_path = args.save_path)
        else:  # No GT
            if args.save_path:
                save_path_nonhung = os.path.join(args.save_path, "nonhung")
                os.makedirs(os.path.dirname(os.path.join(save_path_nonhung, info["path"][0])), exist_ok = True)
                save_indexed(os.path.join(save_path_nonhung, info["path"][0]), saved_mask_nonhung.astype(np.uint8))
        
    if len(iou_summary.keys()) != 0:  
        # IoU result output
        iou_list = []     
        for cat in iou_summary.keys():
            for obj in iou_summary[cat].keys():
                iou_list.append(np.mean(np.array(iou_summary[cat][obj])))
        print("---Mean IoU is: {} ".format(np.mean(np.array(iou_list))))
        print("")
        return np.mean(np.array(iou_list))



if __name__ == '__main__':
    parser = ArgumentParser()
    #optimization
    parser.add_argument('--batch_size', type=int, default=8)
    # Model and ckpt information
    parser.add_argument(
        '--model', 
        type=str,
        default="flowpsam",
        choices = ["flowpsam", "flowisam"],
    )
    parser.add_argument(
        '--ckpt_path', 
        type=str,
        default=None,
        help="ckpt path of flowi-sam / flowp-sam",
    )   
    parser.add_argument(
        '--rgb_encoder', 
        type=str,
        default="vit_h",
        help="size of SAM image encoder to take in rgb",
    )
    parser.add_argument(
        '--rgb_encoder_ckpt_path', 
        type=str,
        default="/path/to/sam_vit_h_4b8939.pth",
        help="ckpt path of SAM image encoder to take in rgb, the ckpt can be downloaded from the official SAM repo (https://github.com/facebookresearch/segment-anything/)",
    )
    parser.add_argument(
        '--flow_encoder', 
        type=str,
        default="vit_b",
        help="size of SAM image encoder to take in flow",
    )
    parser.add_argument(
        '--flow_encoder_ckpt_path', 
        type=str,
        default="/path/to/sam_vit_b_01ec64.pth",
        help="ckpt path of SAM image encoder to take in flow, the ckpt can be downloaded from the official SAM repo (https://github.com/facebookresearch/segment-anything/)",
    )

    # Input configuration
    parser.add_argument(
        '--flow_gaps', 
        type=str,
        default="1,-1,2,-2",
        help="flow frame gaps, a string without spacing",
    ) 
    parser.add_argument(
        '--num_gridside', 
        type=int, 
        default=10,
        help="total number of uniform grid point prompts = num_gridside ** 2",
    )
    parser.add_argument(
        '--dataset', 
        default=None, 
        choices=['dvs17', 'dvs17m', 'dvs16', 'ytvos', 'example'],
        help="evaluation datasets",
    )  

    # Output configuration
    parser.add_argument(
        '--max_obj', 
        type=int, 
        default=5,
        help="max number of objects output",
    )
    parser.add_argument(
        '--sam_channel', 
        type=int, 
        default=0,
        help="the default channel is 0 (in total four channels: 0 1 2 3)",
    )
    parser.add_argument(
        '--mod_thres', 
        type=float, 
        default=-0.,
    )
    parser.add_argument(
        '--save_path', 
        default=None,
        help="path to save masks",
    )   

    args = parser.parse_args()

    # Initialising model
    flowsam = config_model(args)
    for param in flowsam.parameters():
        param.requires_grad=False

    # Initialising dataloader
    val_loader = config_eval_dataloader(args)

    # evaluation 
    eval(args, val_loader, flowsam) 
    
   