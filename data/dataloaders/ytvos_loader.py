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



class YTVOS18m_eval_dataset(Dataset):
    def __init__(
        self, 
        data_dirs, 
        ref_sam, 
        dataset, 
        seqs = None, 
        flow_gaps = [3, -3, 6, -6],
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
        self.val_objnum = {}
        if seqs is None:
            self.seqs = ['0a275e7f12', '0b285c47f6', '0b1627e896', '0f74ec5599', '0f202e9852', '0f9683715b', '0fc958cde2',  '1dfbe6f586', '1f3876f8d0', 
                        '2e42410932', '2f17e4fc1e', '2f710f66bd', '03e2b57b0e',  '11cbcb0b4d', '11e53de6f2', '11feabe596', '14a6b5a139', '3113140ee8',
                        '19e8bc6178', '0065b171f9', '115ee1072c', '258b3b33c6', '1233ac8596', '02264db755', '04194e1248', '04667fabaa',  '39d584b09f', 
                        '2125235a49', '3b512dad74', '3cc6f90cb2', '3cdf03531b', '3db907ac77',  '3e2845307e', '3ea4c49bbe', '3f3b15f083', '3fff16d112', 
                        '4a885fa3ef', '4aa2e0f865', '4c397b6fd4', '4c7710908f', '4ca2ffc361', '4dfaee19e5','4e7a28907f', '4f37e86797', '4fffec8479', 
                        '39bc49a28c',  '346aacf439', '404d02e0c0', '472da1c484', '4894b3b9ea', '35195a56a1', '47269e0499', '450787ac97', '31539918c4',
                        '1cb44b920d', '1d6ff083aa', '2e2bf37e6d', '08d50b926c', '22f02efe3a', '3ae2deaec2', '3bb4e10ed7', '3e3a819865', '4d67c59727', 
                        '02d28375aa', 
                        '0c3a04798c', '0cfe974a89', '0d97fba242', '0f17fa6fcb',  '2a18873352', '2d03593bdc', '18bad52083', '00917dcfc4', 
                        '2f680909e6', '005a527edd', '05a569d8aa', '07a11a35e8', '10eced835e', '11a0c3b724', '16afd538ad', '0a8c467cc3', 
                        '20c6d8b362', '26ddd2ef12', '045f00aed2', '101caff7d4', '142d2621f5',  '125d1b19dd', '211bc5d102', '224e7c833e', 
                        '2481ceafeb', '07129e14a4', '013099c098', '04474174a4', '0693719753', '37d4ae24fc',  '3ab9b1a85a', '4350f3ab60',
                        '3beef45388', '3c5ff93faf', '3e7d2aeb07', '3e680622d7', '3ec273c8d5', '3efc6e9892', '3f0b0dfddd', '499bf07002', 
                        '4a4b50571c', '4a6e3faaa1', '4a088fe99a', '4b19c529fb', '4b97cc7b8d', '4e824b9247', '30dbdb2cd6', '0eefca067f',
                        '36f5cc68fd', '357a710863', '365f459863', '375a49d38f', '377db65f60', '446b8b5f7a', '456fb9362e', '1e1a18c45a']
        else: 
            self.seqs = seqs

        self.rgb_paths = []
        self.flow_paths = []
        self.anno_paths = []

        for idx, seq in enumerate(sorted(self.seqs)):
            if idx % 100 == 0:  print("Loading validation sequence: {}".format(idx))
            seq_dir = os.path.join(self.anno_dir, seq)

            objnum_ref = []
            for filename in sorted(os.listdir(seq_dir)):
                flow_path = os.path.join(seq_dir.replace(self.anno_dir, self.flow_dir), filename.replace(".jpg", ".png"))
                rgb_path = os.path.join(seq_dir.replace(self.anno_dir, self.rgb_dir), filename.replace(".png", ".jpg"))
                self.anno_paths.append(os.path.join(seq_dir, filename.replace(".jpg", ".png")))
                self.flow_paths.append(flow_path)
                self.rgb_paths.append(rgb_path)

                # Check max number of objects in the sequence
                annos_ref = list(processMultiSeg(os.path.join(seq_dir, filename.replace(".jpg", ".png")), gt_resolution = None, out_channels = 4, dataset = "ytvos"))
                num_obj_binary = [(np.sum(anno_ref) > 0).astype(np.float32) for anno_ref in annos_ref]
                objnum_ref.append(int(np.sum(np.array(num_obj_binary))))
            if seq not in self.val_objnum.keys():
                self.val_objnum[seq] = max(objnum_ref)

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
        annos = list(processMultiSeg(self.anno_paths[idx], gt_resolution = None, out_channels = 4, dataset = "ytvos"))[:info["num_obj"]]
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
      
 