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

from ..dataset_utils import readRGB, processMultiSeg, preprocess



class DAVIS_eval_dataset(Dataset):
    def __init__(
        self, 
        data_dirs, 
        ref_sam, 
        dataset, 
        seqs = None, 
        flow_gaps = [1, -1, 2, -2], 
        num_gridside = 10,
    ):
        self.flow_dir = data_dirs[0]
        self.anno_dir = data_dirs[1]
        self.rgb_dir = data_dirs[2]
        self.ref_sam = ref_sam
        self.flow_gaps = flow_gaps
        self.preprocess = preprocess
        self.transform = ResizeLongestSide(1024)
        self.point_grids = build_all_layer_point_grids(n_per_side = num_gridside, n_layers = 0, scale_per_layer = 1)[0]
        if dataset == 'dvs17m':
            self.val_objnum = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 1, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 ,
                    'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 5,
                    'horsejump-high': 1, 'india': 3, 'judo': 2, 'kite-surf': 1, 'lab-coat': 1, 'libby': 1, 'loading': 3, 'mbike-trick': 1,
                    'motocross-jump': 1, 'paragliding-launch': 1, 'parkour': 1, 'pigs': 3, 'scooter-black': 1, 'shooting': 1, 'soapbox': 1}
        elif dataset == 'dvs16':
            self.val_objnum = {'blackswan': 1, 'bmx-trees': 1, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 , 'cows': 1, 
                    'dance-twirl': 1, 'dog': 1, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'horsejump-high': 1, 'kite-surf': 1, 
                    'libby': 1, 'motocross-jump': 1, 'paragliding-launch': 1, 'parkour': 1, 'scooter-black': 1,  'soapbox': 1}
        else: # dvs17
            self.val_objnum = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 2, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 ,
               'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 5,
                'horsejump-high': 2, 'india': 8, 'judo': 2, 'kite-surf': 3, 'lab-coat': 5, 'libby': 1, 'loading': 3, 'mbike-trick': 2,
                 'motocross-jump': 2, 'paragliding-launch': 3, 'parkour': 1, 'pigs': 3, 'scooter-black': 2, 'shooting': 3, 'soapbox': 3}
        if seqs is None:
            self.seqs = self.val_objnum.keys()
        else: 
            self.seqs = seqs

        self.rgb_paths = []
        self.flow_paths = []
        self.anno_paths = []
        for idx, seq in enumerate(sorted(self.seqs)):
            if idx % 100 == 0:  print("Loading validation sequence: {}".format(idx))
            seq_dir = os.path.join(self.anno_dir, seq)
            for filename in sorted(os.listdir(seq_dir)):              
                flow_path = os.path.join(seq_dir.replace(self.anno_dir, self.flow_dir), filename.replace(".jpg", ".png"))
                rgb_path = os.path.join(seq_dir.replace(self.anno_dir, self.rgb_dir), filename.replace(".png", ".jpg"))
                self.anno_paths.append(os.path.join(seq_dir, filename.replace(".jpg", ".png")))
                self.flow_paths.append(flow_path)
                self.rgb_paths.append(rgb_path)
        print("Dataset consists of {} images".format(len(self.flow_paths)))

    def __len__(self):
        return len(self.flow_paths)

    def val_loading(self, idx):
        info = {}

        # Get reference resolution 
        ref = readRGB(self.anno_paths[idx])
        original_size = ref.shape[0:2]

        # Get number of objects
        cat_name = os.path.basename(os.path.dirname(self.anno_paths[idx]))
        info["num_obj"] = self.val_objnum[cat_name]

        # Read optical flow
        flow_images = []
        for flow_gap in self.flow_gaps:
            flow_path = self.flow_paths[idx].replace("FlowImages_gap1", "FlowImages_gap{}".format(flow_gap))
            if os.path.exists(flow_path):
                flow_image = readRGB(flow_path, gt_resolution = original_size)
            else:
                flow_path = self.flow_paths[idx].replace("FlowImages_gap1", "FlowImages_gap{}".format(-1 * flow_gap))
                flow_image = readRGB(flow_path, gt_resolution = original_size)
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

        # Read gt annotation
        annos = list(processMultiSeg(self.anno_paths[idx], gt_resolution = None))[:info["num_obj"]]
        annos = torch.from_numpy(np.stack(annos, 0))

        # Setup grid points
        grid_points = np.array(self.point_grids) * np.array(original_size)[None, ::-1]        
        grid_coords = self.transform.apply_coords(grid_points, original_size)  # 100, 2
        
        info["rgb_image"] = rgb_image.float()    # RGB image
        info["flow_image"] = flow_images.float() # Flow images with different frame gaps
        info["anno"] = annos.float()             # GT annotation
        info["grid"] = torch.from_numpy(grid_coords).unsqueeze(1).float()   # Uniform grid points, size 100, 1, 2
        info["size"] = original_size             # Original image size
        info["path"] = os.path.join(*self.anno_paths[idx].split("/")[-2:])  # Filename reference
        return info
        
    def __getitem__(self, idx):
        return self.val_loading(idx)
      
 

class DAVIS_dataset(Dataset):
    def __init__(
        self, 
        data_dirs, 
        ref_sam, 
        dataset, 
        seqs = None, 
        flow_gaps = [1, -1, 2, -2], 
        num_gridside = 10,
        train = True,
    ):
        self.flow_dir = data_dirs[0]
        self.anno_dir = data_dirs[1]
        self.rgb_dir = data_dirs[2]
        self.ref_sam = ref_sam
        self.train = train
        if train:
            self.flow_gaps = [[1, -1, 2, -2], [1, -1, 2, -2], [1, -1, 1, -1], [2, -2, 2, -2]]
        else:
            self.flow_gaps = [flow_gaps]
        self.preprocess = preprocess
        self.transform = ResizeLongestSide(1024)
        if dataset == 'dvs17m':
            self.val_objnum = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 1, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 ,
                    'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 5,
                    'horsejump-high': 1, 'india': 3, 'judo': 2, 'kite-surf': 1, 'lab-coat': 1, 'libby': 1, 'loading': 3, 'mbike-trick': 1,
                    'motocross-jump': 1, 'paragliding-launch': 1, 'parkour': 1, 'pigs': 3, 'scooter-black': 1, 'shooting': 1, 'soapbox': 1}
        elif dataset == 'dvs16':
            self.val_objnum = {'blackswan': 1, 'bmx-trees': 1, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 , 'cows': 1, 
                    'dance-twirl': 1, 'dog': 1, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'horsejump-high': 1, 'kite-surf': 1, 
                    'libby': 1, 'motocross-jump': 1, 'paragliding-launch': 1, 'parkour': 1, 'scooter-black': 1,  'soapbox': 1}
        else: # dvs17
            self.val_objnum = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 2, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 ,
               'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 5,
                'horsejump-high': 2, 'india': 8, 'judo': 2, 'kite-surf': 3, 'lab-coat': 5, 'libby': 1, 'loading': 3, 'mbike-trick': 2,
                 'motocross-jump': 2, 'paragliding-launch': 3, 'parkour': 1, 'pigs': 3, 'scooter-black': 2, 'shooting': 3, 'soapbox': 3}
        if seqs is None and not train:
            self.seqs = self.val_objnum.keys()
        else: 
            self.seqs = seqs

        self.rgb_paths = []
        self.flow_paths = []
        self.anno_paths = []
        for idx, seq in enumerate(sorted(self.seqs)):
            if train:
                if idx % 100 == 0:  print("Loading training sequence: {}".format(idx))
            else:
                if idx % 100 == 0:  print("Loading validation centroid sequence: {}".format(idx))
            seq_dir = os.path.join(self.anno_dir, seq)
            for filename in sorted(os.listdir(seq_dir)): 
                if train:
                    ref_anno = cv2.imread(os.path.join(seq_dir, filename))
                    ref_anno = cv2.cvtColor(ref_anno, cv2.COLOR_BGR2RGB)
                    if np.sum(ref_anno) == 0:
                        continue             
                flow_path = os.path.join(seq_dir.replace(self.anno_dir, self.flow_dir), filename.replace(".jpg", ".png"))
                rgb_path = os.path.join(seq_dir.replace(self.anno_dir, self.rgb_dir), filename.replace(".png", ".jpg"))
                self.anno_paths.append(os.path.join(seq_dir, filename.replace(".jpg", ".png")))
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
        

    def val_loading(self, idx):
        info = {}

        # Get reference resolution 
        ref = readRGB(self.anno_paths[idx])
        original_size = ref.shape[0:2]

        # Get number of objects
        cat_name = os.path.basename(os.path.dirname(self.anno_paths[idx]))
        info["num_obj"] = self.val_objnum[cat_name]

        # Read optical flow
        flow_images = []
        for flow_gap in self.flow_gaps[0]:
            flow_path = self.flow_paths[idx].replace("FlowImages_gap1", "FlowImages_gap{}".format(flow_gap))
            if os.path.exists(flow_path):
                flow_image = readRGB(flow_path, gt_resolution = original_size)
            else:
                flow_path = self.flow_paths[idx].replace("FlowImages_gap1", "FlowImages_gap{}".format(-1 * flow_gap))
                flow_image = readRGB(flow_path, gt_resolution = original_size)
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

        # Read gt annotation and gt centroid
        annos = list(processMultiSeg(self.anno_paths[idx], gt_resolution = None))[:info["num_obj"]]
        centroid_coords_batch = []
        for anno in annos:
            if np.sum(anno) == 0: # If no anno, set [0, 0] as placeholder
                centroid_coords_batch.append(torch.tensor([0., 0.]))
            else:
                centroid_raw = ndimage.center_of_mass(anno)
                centroid = np.array([centroid_raw[1], centroid_raw[0]])
                centroid_coords = self.transform.apply_coords(centroid, original_size)
                centroid_coords_batch.append(torch.from_numpy(centroid_coords))
        annos = torch.from_numpy(np.stack(annos, 0))
        centroid_coords_batch = torch.stack(centroid_coords_batch, 0).unsqueeze(1)
        
        info["rgb_image"] = rgb_image.float()    # RGB image
        info["flow_image"] = flow_images.float() # Flow images with different frame gaps
        info["anno"] = annos.float()             # GT annotation
        info["centroid"] = centroid_coords_batch.float()   # gt centroid points, size N, 1, 2
        info["size"] = original_size             # Original image size
        info["path"] = os.path.join(*self.anno_paths[idx].split("/")[-2:])  # Filename reference
        return info
        
    def __getitem__(self, idx):
        if self.train:
            return self.train_loading(idx)
        else:
            return self.val_loading(idx)

