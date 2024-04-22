import sys
sys.path.append('core')

import os
import cv2
import glob
import torch
import argparse
import numpy as np

import PIL
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from io import BytesIO
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def writeFlowFile(filename, uv):
    """
    According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
    Contact: dqsun@cs.brown.edu
    Contact: schar@middlebury.edu
    """
    TAG_STRING = np.array(202021.25, dtype=np.float32)
    if uv.shape[2] != 2:
        sys.exit("writeFlowFile: flow must have two bands!")
    H = np.array(uv.shape[0], dtype=np.int32)
    W = np.array(uv.shape[1], dtype=np.int32)
    with open(filename, 'wb') as f:
        f.write(TAG_STRING.tobytes())
        f.write(W.tobytes())
        f.write(H.tobytes())
        f.write(uv.tobytes())


def load_image(imfile, resolution=None):
    img = Image.open(imfile)
    if resolution:
        img = img.resize(resolution, PIL.Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.to(DEVICE)


def pil_from_raw_rgb(raw):
    return np.array(Image.open(BytesIO(raw))).astype(np.uint8)


class ImgPair(Dataset):
    def __init__(self, data_dir, gap, reverse):
        self.data_dir = data_dir
        self.images = None
        self.images_ = None
        self.gap = gap
        self.reverse = reverse

        images = glob.glob(os.path.join(self.data_dir, '*.png')) + \
                 glob.glob(os.path.join(self.data_dir, '*.jpg'))
        self.images = sorted(images)
        self.images_ = self.images[:-gap]

        
    def __len__(self):
        return len(self.images_)

    def __getitem__(self, index):
        images = self.images
        images_ = self.images_
        gap = self.gap
        if self.reverse:
            image1 = load_image(images[index + gap])
            image2 = load_image(images_[index])
            svfile = images[index + gap]
        else:
            image1 = load_image(images_[index])
            image2 = load_image(images[index + gap])
            svfile = images_[index]
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1[None], image2[None])
        return image1[0], image2[0], svfile
   

def predict_batch(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module

    model.to(DEVICE)
    model.eval()

    folder = os.path.basename(args.path)
    floout = os.path.join(args.outroot, folder)
    rawfloout = os.path.join(args.raw_outroot, folder)
    os.makedirs(floout, exist_ok=True)
    os.makedirs(rawfloout, exist_ok=True)

    imgpair_dataset = ImgPair(data_dir=args.path, gap = args.gap, reverse = args.reverse)
    imgpair_loader = DataLoader(imgpair_dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
       
        for _, data in enumerate(imgpair_loader):
            image1, image2, svfiles = data
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            for k, svfile in enumerate(svfiles):
                flopath = os.path.join(floout, os.path.basename(svfile))
                rawflopath = os.path.join(rawfloout, os.path.basename(svfile))

                flo = flow_up[k].permute(1, 2, 0).cpu().numpy()
                
                # save raw flow
                writeFlowFile(rawflopath[:-4]+'.flo', flo)

                # save image.
                flo = flow_viz.flow_to_image(flo)
                cv2.imwrite(flopath[:-4]+'.png', flo[:, :, [2, 1, 0]])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--resolution', nargs='+', type=int)
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for prediction")
    parser.add_argument('--gap', type=int, help="gap between frames")
    parser.add_argument('--reverse', type=int, help="video forward or backward")
    parser.add_argument('--outroot', help="path for output flow as image")
    parser.add_argument('--raw_outroot', help="path for output flow as xy displacement")
    parser.add_argument('--batch_size', type=int, default=4)    

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    predict_batch(args)
