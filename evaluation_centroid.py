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
from data.dataset_config import config_eval_centroid_dataloader
from utils import iou, save_indexed, update_iousummary, hard_thres
    

def run_flowpsam(args, flowsam, info):
    original_size = (info["size"][0][0].item(), info["size"][1][0].item())
    input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
    with torch.no_grad():
        # Inputs
        original_size = (info["size"][0][0].item(), info["size"][1][0].item())
        input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
        flow_image = info["flow_image"].cuda()   # 1 4 3 1024 1024
        rgb_image = info["rgb_image"].cuda()  # 1 3 1024 1024
        centroid_coords = info["centroid"].cuda().squeeze(0) # N 1 2 
        point_labels = torch.ones(centroid_coords.size()[:2], dtype=torch.int, device=centroid_coords.device)
        point_prompts = (centroid_coords, point_labels)
        # Inference
        masks_logit, fiou, mos = flowsam(rgb_image, flow_image, point_prompts, use_cache = False)  
        fiou = fiou[:, args.sam_channel]
        mos = mos[:, 0]
        scores = fiou + mos
        masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
        masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)
        masks = (masks_logit > args.mod_thres).float()
        masks = masks[:, args.sam_channel]
        # If gt anno is empty (i.e., centroid coords are loaded as [0,0]), we force the the predicted to be empty as well
        masks = masks * (centroid_coords.mean(dim = [-2, -1]) > 0).float()[:, None, None]
    masks, output_mask = hard_thres(masks, scores, output_savemask=True)
    if args.save_path is not None:
        save_path = os.path.join(args.save_path, info["path"][0])
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        save_indexed(save_path, output_mask.astype(np.uint8))
    return masks


def run_flowisam(args, flowsam, info):
    original_size = (info["size"][0][0].item(), info["size"][1][0].item())
    input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
    with torch.no_grad():
        # Inputs
        original_size = (info["size"][0][0].item(), info["size"][1][0].item())
        input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
        flow_image = info["flow_image"].cuda()   # 1 4 3 1024 1024
        centroid_coords = info["centroid"].cuda().squeeze(0) # N 1 2 
        point_labels = torch.ones(centroid_coords.size()[:2], dtype=torch.int, device=centroid_coords.device)
        point_prompts = (centroid_coords, point_labels)
        # Inference
        masks_logit, fiou = flowsam(flow_image, point_prompts, use_cache = False)  
        fiou = fiou[:, args.sam_channel]
        scores = fiou
        masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
        masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)
        masks = (masks_logit > args.mod_thres).float()
        masks = masks[:, args.sam_channel]
        # If gt anno is empty (i.e., centroid coords are loaded as [0,0]), we force the the predicted to be empty as well
        masks = masks * (centroid_coords.mean(dim = [-2, -1]) > 0).float()[:, None, None]    
    masks, output_mask = hard_thres(masks, scores, output_savemask=True)
    if args.save_path is not None:
        save_path = os.path.join(args.save_path, info["path"][0])
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        save_indexed(save_path, output_mask.astype(np.uint8))
    return masks


def eval_centroid(args, val_loader, flowsam):
    print("")
    print("---Evaluation centroid steps {}".format(args.model))
    flowsam.eval()
    iou_summary = {}
    for idx, info in enumerate(val_loader):
        if idx % 100 == 0:
            print("---Inference step: {}".format(idx))

        # Set up performance logger
        if os.path.dirname(info["path"][0]) not in iou_summary.keys():
            iou_summary[os.path.dirname(info["path"][0])] = {}
            for obj_idx in range(info["num_obj"].item()):
                iou_summary[os.path.dirname(info["path"][0])][obj_idx] = []
        
        # Running model
        if args.model == "flowpsam":
            masks = run_flowpsam(args, flowsam, info)
        else: #flowisam
            masks = run_flowisam(args, flowsam, info)
        
        # Evaluating IoUs and updating 
        anno = info["anno"].cuda()     # 1 C H W
        for obj_idx in range(info["num_obj"].item()):
            iou_summary[os.path.dirname(info["path"][0])][obj_idx].append(iou(masks[obj_idx], anno[0, obj_idx]).item())

    obj_avg_list = []
    for cat in iou_summary.keys():
        for obj in iou_summary[cat].keys():
            obj_avg_list.append(np.mean(np.array(iou_summary[cat][obj])))
    print("---Mean centroid IoU is: {} ".format(np.mean(np.array(obj_avg_list))))
    print("")
    return np.mean(np.array(obj_avg_list))


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
        '--dataset', 
        default=None, 
        choices=['dvs17', 'dvs17m', 'dvs16', 'ytvos'],
        help="evaluation datasets",
    )  
    # Output configuration
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
    val_loader = config_eval_centroid_dataloader(args)

    # evaluation 
    eval_centroid(args, val_loader, flowsam) 
        

