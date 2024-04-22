import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from argparse import ArgumentParser

from evaluation import eval
from evaluation_centroid import eval_centroid
from model.model_config import config_model
from data.dataset_config import config_eval_dataloader, config_eval_centroid_dataloader, config_train_dataloader
from utils import iou, save_indexed, update_iousummary 


def train_flowisam(args, train_loader, flowsam, val_loader=None, val_loader_centroid=None):
    print("Training steps {}".format(args.model))
    iters = 0
    epochs = 50 * args.accum_step
    total_iters = 20000 * args.accum_step
    eval_freq = 500 * args.accum_step
    save_freq = 500 * args.accum_step
    log_freq = 20 * args.accum_step
        
    optimizer = optim.Adam(flowsam.mask_decoder.parameters(), lr=args.lr)
    writer = SummaryWriter(logdir=args.model_save_path + "/logs_flowisam/")
    flowsam.train()
    for epoch in range(epochs):
        if iters > total_iters:
            break
        for idx, info_train in enumerate(train_loader):
            print("Starting iteration {}".format(iters))
            if iters % eval_freq == 0:
                result_list = []
                if val_loader_centroid is not None:
                    result = eval_centroid(args, val_loader_centroid, flowsam)
                    writer.add_scalar('IoU/val_centroid', result, iters)
                    result_list.append(result)
                if val_loader is not None:
                    result = eval(args, val_loader, flowsam)
                    writer.add_scalar('IoU/val', result, iters)
                    result_list.append(result)
                optimizer.zero_grad()
                flowsam.train()
                if iters % save_freq == 0:
                    while len(result_list) < 2:
                        result_list.append(0.)
                    filename = os.path.join(args.model_save_path + "/models_flowisam/", 'checkpoint_{}-{}_{}.pth'.format(iters, result_list[0], result_list[1]))
                    os.makedirs(os.path.dirname(filename), exist_ok = True)
                    torch.save({
                        'iteration': iters,
                        'model_state_dict': flowsam.mask_decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, filename)

            original_size = (info_train["size"][0][0].item(), info_train["size"][1][0].item())
            input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
            flow_image = info_train["flow_image"].cuda()   # B 4 3 1024 1024
            anno = info_train["anno"].cuda()     # B 270 480
            random_coords = info_train["random"].cuda() # B N 2
            point_labels = torch.ones(random_coords.size()[:2], dtype=torch.int, device=random_coords.device)
            point_prompts = (random_coords, point_labels)

            masks_logit, fiou = flowsam(flow_image, point_prompts, use_cache = False)  
            masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
            masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)[:, args.sam_channel]
            masks = masks_logit.sigmoid()
            fiou = fiou[:, args.sam_channel]
                
            bg_channel = (info_train["anno_random_idx"].cuda() == 0).float() # B
            gt_fiou = iou(masks, anno).detach() * (1 - bg_channel)
            loss_fiou = ((fiou - gt_fiou) ** 2 * (1 - (1 - args.bg_fiou_scale) * bg_channel)).mean()
            loss_mask = (F.binary_cross_entropy_with_logits(masks_logit, anno, reduction = "none") * (1 - bg_channel[:, None, None])).mean()
            loss = loss_mask + args.loss_scale_fiou * loss_fiou
            loss.backward()

            if iters % args.accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(" --- Training loss: {:.4f}; {:.4f}, {:.4f}".format(loss.item(), loss_mask.item(), loss_fiou.item()))
            
            if iters % log_freq == 0:
                writer.add_scalar('Loss/total', loss.item(), iters)
                writer.add_scalar('Loss/mask', loss_mask.item(), iters)
                writer.add_scalar('Loss/fiou', loss_fiou.item(), iters)
    
            iters += 1
                
                

def train_flowpsam(args, train_loader, flowsam, val_loader=None, val_loader_centroid=None):
    print("Training steps {}".format(args.model))
    iters = 0
    epochs = 50 * args.accum_step
    total_iters = 20000 * args.accum_step
    eval_freq = 500 * args.accum_step
    save_freq = 500 * args.accum_step
    log_freq = 20 * args.accum_step
        
    optimizer = optim.Adam(flowsam.mask_decoder.parameters(), lr=args.lr)
    writer = SummaryWriter(logdir=args.model_save_path + "/logs_flowpsam/")
    flowsam.train()
    for epoch in range(epochs):
        if iters > total_iters:
            break
        for idx, info_train in enumerate(train_loader):
            print("Starting iteration {}".format(iters))
            if iters % eval_freq == 0:
                result_list = []
                if val_loader_centroid is not None:
                    result = eval_centroid(args, val_loader_centroid, flowsam)
                    writer.add_scalar('IoU/val_centroid', result, iters)
                    result_list.append(result)
                if val_loader is not None:
                    result = eval(args, val_loader, flowsam)
                    writer.add_scalar('IoU/val', result, iters)
                    result_list.append(result)
                optimizer.zero_grad()
                flowsam.train()
                if iters % save_freq == 0:
                    while len(result_list) < 2:
                        result_list.append(0.)
                    filename = os.path.join(args.model_save_path + "/models_flowpsam/", 'checkpoint_{}-{}_{}.pth'.format(iters, result_list[0], result_list[1]))
                    os.makedirs(os.path.dirname(filename), exist_ok = True)
                    torch.save({
                        'iteration': iters,
                        'model_state_dict': flowsam.mask_decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, filename)
            original_size = (info_train["size"][0][0].item(), info_train["size"][1][0].item())
            input_size = (int(original_size[0] * 1024 / max(*original_size)), int(original_size[1] * 1024 / max(*original_size)))
            flow_image = info_train["flow_image"].cuda()   # B 4 3 1024 1024
            rgb_image = info_train["rgb_image"].cuda()  # B 3 1024 1024
            anno = info_train["anno"].cuda()     # B 270 480
            random_coords = info_train["random"].cuda() # B N 2
            point_labels = torch.ones(random_coords.size()[:2], dtype=torch.int, device=random_coords.device)
            point_prompts = (random_coords, point_labels)

            masks_logit, fiou, mos = flowsam(rgb_image, flow_image, point_prompts, use_cache = False)  
            masks_logit = masks_logit[..., : input_size[0], : input_size[1]]
            masks_logit = F.interpolate(masks_logit, original_size, mode="bilinear", align_corners=False)[:, args.sam_channel]
            masks = masks_logit.sigmoid()
            fiou = fiou[:, args.sam_channel]
            mos = mos[:, 0]
            
            bg_channel = (info_train["anno_random_idx"].cuda() == 0).float() # B
            gt_fiou = iou(masks, anno).detach() * (1 - bg_channel)
            loss_fiou = ((fiou - gt_fiou) ** 2 * (1 - (1 - args.bg_fiou_scale) * bg_channel)).mean()
            loss_mos = (F.binary_cross_entropy(mos, 1 - bg_channel)).mean()
            loss_mask = (F.binary_cross_entropy_with_logits(masks_logit, anno, reduction = "none") * (1 - bg_channel[:, None, None])).mean()
            loss = loss_mask + args.loss_scale_fiou * loss_fiou + args.loss_scale_mos * loss_mos
            loss.backward()

            if iters % args.accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            print(" --- Training loss: {:.4f}; {:.4f}, {:.4f}, {:.4f}".format(loss.item(), loss_mask.item(), loss_fiou.item(), loss_mos.item()))
            
            if iters % log_freq == 0:
                writer.add_scalar('Loss/total', loss.item(), iters)
                writer.add_scalar('Loss/mask', loss_mask.item(), iters)
                writer.add_scalar('Loss/fiou', loss_fiou.item(), iters)
                writer.add_scalar('Loss/mos', loss_mos.item(), iters)
    
            iters += 1
     

if __name__ == '__main__':
    parser = ArgumentParser()
    # Training information
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8,
    )
    parser.add_argument('--accum_step', 
        type=int, 
        default=1,
        help="gradient accummulation"
    ) 
    parser.add_argument('--lr', 
        type=float, 
        default=1e-5,
    )
    parser.add_argument(
        '--dataset', 
        default=None, 
        choices=['dvs17', 'dvs17m', 'dvs16', 'oclrsyn'],
        help="train datasets",
    )  
    parser.add_argument(
        '--loss_scale_fiou', 
        type=float, 
        default=0.01,
        help="the loss scale for fiou"
    )  
    parser.add_argument(
        '--loss_scale_mos', 
        type=float, 
        default=0.01,
        help="the loss scale for mos"
    )    
    parser.add_argument(
        '--bg_fiou_scale', 
        type=float, 
        default=0.2,
        help="the loss scale when the gt_fiou=0 (i.e., the point prompt is within the background)"
    )   
    
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
        help="resume ckpt path of flowi-sam / flowp-sam",
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
        help="flow frame gaps, a string without spacing. This is for evaluation",
    ) 
    parser.add_argument(
        '--num_gridside', 
        type=int, 
        default=10,
        help="total number of uniform grid point prompts = num_gridside ** 2",
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
        '--model_save_path', 
        default=None,
        help="path to save log and ckpt",
    )   


    args = parser.parse_args()
    args.save_path = None

    # Initialising model
    flowsam = config_model(args)
    for param in flowsam.parameters():
        param.requires_grad=False
    for param in flowsam.mask_decoder.parameters():
        param.requires_grad=True

    # Initialising dataloader
    train_loader = config_train_dataloader(args)
    val_loader = config_eval_dataloader(args)
    val_loader_centroid = config_eval_centroid_dataloader(args)

    if args.model == "flowpsam":
        train_flowpsam(args, train_loader, flowsam, val_loader=val_loader, val_loader_centroid=val_loader_centroid)
    else:
        train_flowisam(args, train_loader, flowsam, val_loader=val_loader, val_loader_centroid=val_loader_centroid)
