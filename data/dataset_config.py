import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from data.dataloaders.dvs_loader import DAVIS_eval_dataset, DAVIS_dataset
from data.dataloaders.oclrsyn_loader import OCLRsyn_dataset
from data.dataloaders.ytvos_loader import YTVOS18m_eval_dataset
from data.dataloaders.example_loader import Example_eval_dataset
from segment_anything.build_sam import sam_model_registry

def config_train_dataloader(args):
    """
    val_data_dirs: paths of:
        [flowimages (0th element), 
        gt_annotations (1st element), 
        rgbimages (2nd element)]
    """
    flow_gaps = [int(i) for i in args.flow_gaps.split(",")]
    ref_sam = sam_model_registry[args.rgb_encoder](checkpoint=args.rgb_encoder_ckpt_path)

    if args.dataset == "dvs17m":   
        train_data_dirs = ["/path/to/DAVIS2017m/FlowImages_gap1/", 
                        "/path/to/DAVIS2017m/Annotations/480p/",
                        "/path/to/DAVIS2017m/JPEGImages/480p/"]
        train_seq = ['bear', 'bmx-bumps', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',  
                    'classic-car', 'crossing', 'dance-jump', 'dog-agility', 'dog-gooses', 'drift-turn',  
                    'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk',  
                    'koala', 'longboard', 'lucia', 'mallard-fly', 'motocross-bumps', 'motorbike',  
                    'paragliding', 'rallye', 'rhino', 'rollerblade', 'scooter-gray', 'skate-park',  
                    'soccerball', 'stunt', 'surf', 'swing', 'tennis', 'tractor-sand', 'walking', 
                    'drone', 'lindy-hop', 'miami-surf', 'night-race', 'planes-water', 'schoolgirls', 'sheep',
                    'scooter-board', 'snowboard', 'varanus-cage', 'cat-girl', 'stroller', 'train',
                    'boat', 'color-run', 'dancing', 'disc-jockey', 'dogs-scale', 'lady-running', 'mallard-water', 'tuk-tuk', 'upside-down']
        train_dataset = DAVIS_dataset(data_dirs=train_data_dirs, seqs=train_seq, ref_sam=ref_sam, train = True,
                                            dataset=args.dataset, flow_gaps=flow_gaps)
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    

    elif args.dataset == "dvs17":   
        train_data_dirs = ["/path/to/DAVIS2017/FlowImages_gap1/", 
                        "/path/to/DAVIS2017/Annotations/480p/",
                        "/path/to/DAVIS2017/JPEGImages/480p/"]
        train_seq = ['bear', 'bmx-bumps', 'boxing-fisheye', 'breakdance-flare', 'bus', 'car-turn',  
                    'classic-car', 'crossing', 'dance-jump', 'dog-agility', 'dog-gooses', 'drift-turn',  
                    'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low', 'kid-football', 'kite-walk',  
                    'koala', 'longboard', 'lucia', 'mallard-fly', 'motocross-bumps', 'motorbike',  
                    'paragliding', 'rallye', 'rhino', 'rollerblade', 'scooter-gray', 'skate-park',  
                    'soccerball', 'stunt', 'surf', 'swing', 'tennis', 'tractor-sand', 'walking', 
                    'drone', 'lindy-hop', 'miami-surf', 'night-race', 'planes-water', 'schoolgirls', 'sheep',
                    'scooter-board', 'snowboard', 'varanus-cage', 'cat-girl', 'stroller', 'train',
                    'boat', 'color-run', 'dancing', 'disc-jockey', 'dogs-scale', 'lady-running', 'mallard-water', 'tuk-tuk', 'upside-down']
        train_dataset = DAVIS_dataset(data_dirs=train_data_dirs, seqs=train_seq, ref_sam=ref_sam, train = True,
                                            dataset=args.dataset, flow_gaps=flow_gaps)
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)
    
    elif args.dataset == "dvs16":   
        train_data_dirs = ["/path/to/DAVIS2016/FlowImages_gap1/", 
                        "/path/to/DAVIS2016/Annotations/480p/",
                        "/path/to/DAVIS2016/JPEGImages/480p/"]
        train_seq =  ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 
                'car-turn', 'dance-jump', 'dog-agility', 'drift-turn', 'elephant', 
                'flamingo', 'hike', 'hockey', 'horsejump-low', 'kite-walk', 
                'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike',
                'paragliding', 'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 
                'stroller', 'surf', 'swing', 'tennis', 'train']
        train_dataset = DAVIS_dataset(data_dirs=train_data_dirs, seqs=train_seq, ref_sam=ref_sam, train = True,
                                            dataset=args.dataset, flow_gaps=flow_gaps)
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    elif args.dataset == "oclrsyn":   
        train_data_dirs = ["/path/to/OCLRsyn/FlowImages_gap1/", 
                        "/path/to/OCLRsyn/Annotations/",
                        "/path/to/OCLRsyn/JPEGImages/"]
        train_seq = [x for x in sorted(os.listdir(train_data_dirs[1])) if x[6] in ['a', 'b', 'c', 'd', 'e', 'f']]
        train_dataset = OCLRsyn_dataset(data_dirs=train_data_dirs, seqs=train_seq, ref_sam=ref_sam, train = True,
                                            dataset=args.dataset, flow_gaps=flow_gaps)
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False)

    return train_loader

def config_eval_centroid_dataloader(args):
    """
    val_data_dirs: paths of:
        [flowimages (0th element), 
        gt_annotations (1st element), 
        rgbimages (2nd element)]
    val_seq: specify the seqeunce name (default is None, which will automatically load the sequence name inside dataset files)
    """
    flow_gaps = [int(i) for i in args.flow_gaps.split(",")]
    ref_sam = sam_model_registry[args.rgb_encoder](checkpoint=args.rgb_encoder_ckpt_path)

    if args.dataset == "dvs17m" or args.dataset == "oclrsyn":  
        val_data_dirs = ["/path/to/DAVIS2017m/FlowImages_gap1/", 
                        "/path/to/DAVIS2017m/Annotations/480p/",
                        "/path/to/DAVIS2017m/JPEGImages/480p/"] 
        val_seq = None
        val_dataset = DAVIS_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, train = False,
                                            dataset=args.dataset, flow_gaps=flow_gaps)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    
    elif args.dataset == "dvs17":   
        val_data_dirs = ["/path/to/DAVIS2017/FlowImages_gap1/", 
                        "/path/to/DAVIS2017/Annotations/480p/",
                        "/path/to/DAVIS2017/JPEGImages/480p/"] 
        val_seq = None
        val_dataset = DAVIS_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, train = False,
                                            dataset=args.dataset, flow_gaps=flow_gaps)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    
    elif args.dataset == "dvs16":   
        val_data_dirs = ["/path/to/DAVIS2016/FlowImages_gap1/", 
                        "/path/to/DAVIS2016/Annotations/480p/",
                        "/path/to/DAVIS2016/JPEGImages/480p/"] 
        val_seq = None
        val_dataset = DAVIS_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, train = False,
                                            dataset=args.dataset, flow_gaps=flow_gaps)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
        
    return val_loader
    

def config_eval_dataloader(args):
    """
    val_data_dirs: paths of:
        [flowimages (0th element), 
        gt_annotations (1st element), 
        rgbimages (2nd element)]
    """
    flow_gaps = [int(i) for i in args.flow_gaps.split(",")]
    ref_sam = sam_model_registry[args.rgb_encoder](checkpoint=args.rgb_encoder_ckpt_path)

    if args.dataset == "dvs17m" or args.dataset == "oclrsyn": 
        val_data_dirs = ["/path/to/DAVIS2017m/FlowImages_gap1/", 
                        "/path/to/DAVIS2017m/Annotations/480p/",
                        "/path/to/DAVIS2017m/JPEGImages/480p/"]
        val_seq = None
        val_dataset = DAVIS_eval_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, 
                                            dataset=args.dataset, flow_gaps=flow_gaps, num_gridside=args.num_gridside)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    
    elif args.dataset == "dvs17":   
        val_data_dirs = ["/path/to/DAVIS2017/FlowImages_gap1/", 
                        "/path/to/DAVIS2017/Annotations/480p/",
                        "/path/to/DAVIS2017/JPEGImages/480p/"]
        val_seq = None
        val_dataset = DAVIS_eval_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, 
                                            dataset=args.dataset, flow_gaps=flow_gaps, num_gridside=args.num_gridside)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    
    elif args.dataset == "dvs16":   
        val_data_dirs = ["/path/to/DAVIS2016/FlowImages_gap1/", 
                        "/path/to/DAVIS2016/Annotations/480p/",
                        "/path/to/DAVIS2016/JPEGImages/480p/"]
        val_seq = None
        val_dataset = DAVIS_eval_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, 
                                            dataset=args.dataset, flow_gaps=flow_gaps, num_gridside=args.num_gridside)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    elif args.dataset == "ytvos":
        val_data_dirs = ["/path/to/YTVOS2018m/FlowImages_gap1/", 
                        "/path/to/YTVOS2018m/Annotations/",
                        "/path/to/YTVOS2018m/JPEGImages/"]
        val_seq = None   
        val_dataset = YTVOS18m_eval_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, 
                                            dataset=args.dataset, flow_gaps=flow_gaps, num_gridside=args.num_gridside)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    elif args.dataset == "example":  # No annotations (annotation path) given
        val_data_dirs = ["/path/to/example_dataset/FlowImages_gap1/", 
                        "/path/to/example_dataset/JPEGImages/"]
        val_seq = None 
        val_dataset = Example_eval_dataset(data_dirs=val_data_dirs, seqs=val_seq, ref_sam=ref_sam, 
                                            dataset=args.dataset, flow_gaps=flow_gaps, num_gridside=args.num_gridside)
        val_loader = DataLoader(val_dataset, num_workers=8, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    
    return val_loader