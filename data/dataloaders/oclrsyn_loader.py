import os
import cv2
import torch
import random
import einops
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.utils.amg import build_all_layer_point_grids

from ..dataset_utils import readRGB, processMultiSeg, processMultiAmodalSeg, preprocess


class OCLRsyn_dataset(Dataset):
    """
    - The default dataset, loading amodal masks and convert into modal masks
      Annotation directory: 'AmodalAnnotations'
    - If the amodal masks in OCLR-syn are converted to modal masks as preprocessing, please consider set 'amodal_mask_inputs' to False
      Annotation directory: 'Annotations'
    """
    def __init__(
        self, 
        data_dirs, 
        ref_sam, 
        dataset, 
        seqs = None, 
        flow_gaps = [1, -1, 2, -2], 
        num_gridside = 10,
        train = True,
        amodal_mask_inputs = True
    ):
        self.flow_dir = data_dirs[0]
        self.amodal_mask_inputs = amodal_mask_inputs
        # Check if the annotation directory is for amodal segmentation
        if self.amodal_mask_inputs and "AmodalAnnotations" not in data_dirs[1]:
            self.anno_dir = data_dirs[1].replace("Annotations", "AmodalAnnotations")
        else:
            self.anno_dir = data_dirs[1]              
        self.rgb_dir = data_dirs[2]
        self.ref_sam = ref_sam
        self.train = train
        if train:
            self.flow_gaps = [[1, -1, 2, -2]]
        else:
            self.flow_gaps = [flow_gaps]
        self.preprocess = preprocess
        self.transform = ResizeLongestSide(1024)
        self.seqs = seqs

        self.rgb_paths = []
        self.flow_paths = []
        self.anno_paths = []
        for idx, seq in enumerate(sorted(self.seqs)):
            if idx % 100 == 0:  print("Loading training sequence: {}".format(idx))
            seq_dir = os.path.join(self.rgb_dir, seq)
            for filename in sorted(os.listdir(seq_dir)):       
                anno_path = os.path.join(seq_dir.replace(self.rgb_dir, self.anno_dir), filename.replace(".jpg", ".png")) # e.g., "00000.png"
                flow_path = os.path.join(seq_dir.replace(self.rgb_dir, self.flow_dir), filename.replace(".jpg", ".png"))
                rgb_path = os.path.join(seq_dir, filename)
                self.anno_paths.append(anno_path)
                self.flow_paths.append(flow_path)
                self.rgb_paths.append(rgb_path)
        print("Dataset consists of {} images".format(len(self.flow_paths)))

    def __len__(self):
        return len(self.flow_paths) * len(self.flow_gaps)
        
    def train_loading(self, idx_all):
        info = {}
        original_size = (480, 854)
        idx = idx_all % len(self.flow_paths) 
        flow_idx = idx_all // len(self.flow_paths) # the idx of selected flow gap set for training

        # Read optical flow
        flow_images = []
        selected_flow_gaps = self.flow_gaps[flow_idx]
        for flow_gap in selected_flow_gaps:
            flow_path = self.flow_paths[idx].replace("FlowImages_gap1", "FlowImages_gap{}".format(flow_gap))
            if os.path.exists(flow_path):
                flow_image = readRGB(flow_path, gt_resolution = original_size)
            elif os.path.exists(self.flow_paths[idx].replace("FlowImages_gap1", "FlowImages_gap{}".format(-1 * flow_gap))):
                flow_path = self.flow_paths[idx].replace("FlowImages_gap1", "FlowImages_gap{}".format(-1 * flow_gap))
                flow_image = readRGB(flow_path, gt_resolution = original_size)
            else:
                print(flow_path) # For debugging
                assert 0 == 1
            flow_image = self.transform.apply_image(flow_image)
            flow_image_torch = torch.as_tensor(flow_image)
            flow_image_torch = flow_image_torch.permute(2, 0, 1).contiguous()
            flow_image = self.preprocess(flow_image_torch)  # 3 1024 1024
            flow_images.append(flow_image)
        flow_images = torch.stack(flow_images, 0)

        # Read RGB 
        rgb_image = readRGB(self.rgb_paths[idx], gt_resolution = original_size)
        rgb_image = self.transform.apply_image(rgb_image)
        rgb_image_torch = torch.as_tensor(rgb_image)
        rgb_image_torch = rgb_image_torch.permute(2, 0, 1).contiguous()
        rgb_image = self.preprocess(rgb_image_torch)  #3 1024 1024
        
        if self.amodal_mask_inputs:
            annos = [anno_ for anno_ in list(processMultiAmodalSeg(self.anno_paths[idx], gt_resolution = original_size, with_bg = True)) if np.sum(anno_) > 0]
        else:
            annos = [anno_ for anno_ in list(processMultiSeg(self.anno_paths[idx], gt_resolution = original_size, with_bg = True)) if np.sum(anno_) > 0]

        # Select a random object
        anno_random_idx = random.choice(list(np.arange(len(annos))))
        anno = annos[anno_random_idx]  

        # Select a random point from the object
        ones_indices = np.where(anno == 1)
        point_random_idx = random.choice(list(np.arange(len(ones_indices[0])))) 
        random_points = np.array([ones_indices[1][point_random_idx], ones_indices[0][point_random_idx]])
        random_coords = self.transform.apply_coords(random_points, original_size)
        random_coords = torch.from_numpy(random_coords).unsqueeze(0)

        info["rgb_image"] = rgb_image.float()    # RGB image
        info["flow_image"] = flow_images.float() # Flow images with different frame gaps
        info["anno"] = torch.from_numpy(anno).float() # GT annotation for single object
        info["anno_random_idx"] = anno_random_idx
        info["random"] = random_coords.float()   # random point with gt anno, size 1, 1, 2
        info["size"] = original_size             # Original image size
        info["path"] = os.path.join(*self.anno_paths[idx].split("/")[-2:])  # Filename reference
        return info
  
    def __getitem__(self, idx):
        return self.train_loading(idx)

