import os
import cv2
from argparse import ArgumentParser
from scipy.optimize import linear_sum_assignment
from utils import imwrite_indexed

def save_indexed(filename, img):
    """ Save image with given colour palette """
    c,h,w = np.shape(img)
    output_mask = np.zeros_like(img[0])
    for score_idx in range(c):
        output_mask = output_mask * (1 - img[score_idx]) + img[score_idx] * (score_idx + 1)
    img = np.clip(output_mask, 0, c + 1).astype(np.uint8)
    img = (5+img) * (img > 0) 
    color_palette = np.array([[0,0,0],[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [64, 128, 0], [191, 0, 0], [0, 64, 0], [0, 191, 0], [0, 0, 64], [0, 0, 191], [0, 64, 128], [128, 0, 64], [128, 64, 0], [0, 128, 64], [64, 0, 128], [191, 191, 0], [0, 191, 191]]).astype(np.uint8)
    imwrite_indexed(filename, img, color_palette)

def threeway_hungarian(m1, m2, m3, thres = 0.5, emp=False):
    m1 = (m1>thres)
    m2 = (m2>thres)
    m3 = (m3>thres)
    
    ious_23 = iou(m2[:, None], m3[None], emp=False)
    orig_idx, hung_idx = linear_sum_assignment(-ious_23)
    m3_aligned = m3[hung_idx]
    i23 = ious_23[orig_idx, hung_idx]

    ious_13 = iou(m1[:, None], m3_aligned[None], emp=False)
    ious_12 = iou(m1[:, None], m2[None], emp=False)
    orig_idx_13, hung_idx_13 = linear_sum_assignment(-ious_13)
    orig_idx_12, hung_idx_12 = linear_sum_assignment(-ious_12)
    return hung_idx_12, (hung_idx_12 == hung_idx_13) *1.

def multiway_hungarian(p, m, b1, b2, f1, f2):
    hung_idx, s1 = threeway_hungarian(p, m, b1)
    _, s2 = threeway_hungarian(p, m, b2)
    _, s3 = threeway_hungarian(p, m, f1)
    _, s4 = threeway_hungarian(p, m, f2)
    m_aligned = m[hung_idx]
    out = []
    s = s1+s2+s3+s4
    for i in range(s.shape[0]):
        if s[i] >= 3: #use 4 for flow-only
            out.append(m_aligned[i])
        elif s[i] == 0:
            out.append(p[i])
        else:
            out.append(m_aligned[i]+p[i])
    out = (np.stack(out,0) > 0.5) * 1.
    return out

def main(args):
    save_mask = True
    orig_iouss = []
    seq_ious = []
    
    for cat in args.cats:
        image_dir = os.path.join(args.rgb_dir, cat)
        obj_num = args.obj_nums[cat]
        orig_ious = []
        preds = []
        gts = []
        masks = []
        last_pred = []
        for idx, image_path in enumerate(sorted(os.listdir(image_dir))):
            m = processMultiSeg(os.path.join(args.pred_dir, 'modal', cat, image_path.replace(".jpg", ".png")))
            f = processMultiSeg(os.path.join(args.flow_pred_dir, 'modal', cat, image_path.replace(".jpg", ".png")))[:5]
            masks.append(remove_overlapping_masks(np.concatenate([m,f],0)))
                
            gt = processMultiSeg(os.path.join(args.gt_dir, cat, image_path.replace(".jpg", ".png")))[:obj_num]
            gts.append(gt)
        
        for idx, image_path in enumerate(sorted(os.listdir(image_dir))):
            mask = masks[idx]
            if idx > 0:
                flow = read(os.path.join(args.flow_b1_dir, cat, image_path.replace('jpg','flo')))
                prev_mask = np.stack([warp_flow(last_pred[-1][i], flow) for i in range(last_pred[-1].shape[0])], 0)
                b1_mask = masks[idx-1]
                b1_mask = np.stack([warp_flow(b1_mask[i], flow) for i in range(b1_mask.shape[0])], 0)
            else:
                prev_mask = mask
                b1_mask = mask

            if idx > 1:
                flow = read(os.path.join(args.flow_b2_dir, cat, image_path.replace('jpg','flo')))
                b2_mask = masks[idx-2]
                b2_mask = np.stack([warp_flow(b2_mask[i], flow) for i in range(b2_mask.shape[0])], 0)
                last_pred = last_pred[-2:]
            else:
                b2_mask = mask
            
            if idx < len(masks) - 1:
                flow = read(os.path.join(args.flow_f1_dir, cat, image_path.replace('jpg','flo')))
                f1_mask = masks[idx+1]
                f1_mask = np.stack([warp_flow(f1_mask[i], flow) for i in range(f1_mask.shape[0])], 0)
            else:
                f1_mask = mask           

            if idx < len(masks) - 2:
                flow = read(os.path.join(args.flow_f2_dir, cat, image_path.replace('jpg','flo')))
                f2_mask = masks[idx+2]
                f2_mask = np.stack([warp_flow(f2_mask[i], flow) for i in range(f2_mask.shape[0])], 0)
            else:
                f2_mask = mask        

            output_masks = multiway_hungarian(prev_mask, mask, b1_mask, b2_mask, f1_mask, f2_mask)

            last_pred.append(output_masks)
            output_masks = remove_overlapping_masks(output_masks)
            preds.append(output_masks)

            orig_iou, _ = hungarian_iou(mask, gts[idx], emp=True)
        
            orig_ious.append(orig_iou)
        
            if save_mask:
                save_dir = "{}".format(args.save_dir)
                os.makedirs(save_dir, exist_ok=True)

                save_dir2 = "{}/{}_seq/".format(args.save_dir,args.dataset)
                os.makedirs(save_dir2, exist_ok=True)
                os.makedirs(os.path.join(save_dir2,cat), exist_ok=True)
                save_indexed(os.path.join(save_dir2, cat, os.path.basename(image_path).replace(".jpg", ".png")), output_masks.astype(np.uint8))

                save_dir2 = "{}/{}_seq_hung/".format(args.save_dir,args.dataset)
                os.makedirs(save_dir2, exist_ok=True)
                os.makedirs(os.path.join(save_dir2,cat), exist_ok=True)
                save_indexed(os.path.join(save_dir2, cat, os.path.basename(image_path).replace(".jpg", ".png")), hungarian_iou(output_masks, gts[idx], emp=True)[1].astype(np.uint8))
            
        
        seq_iou, preds = seq_hungarian_iou(preds, gts)
        orig_ious = np.stack(orig_ious, 0).mean(0)
        orig_iouss.extend(orig_ious)
        seq_ious.extend(seq_iou)
        print(cat, orig_ious, seq_iou)

    print(sum(orig_iouss)/len(orig_iouss))
    print(sum(seq_ious)/len(seq_ious))        


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['dvs16', 'dvs17m', 'dvs17'], default='dvs17')
    parser.add_argument('--pred_dir', type=str, default='')
    parser.add_argument('--flow_pred_dir', type=str, default='')
    parser.add_argument('--save_path', type=str, default='seq_masks')
    
    args = parser.parse_args()

    args.flow_b1_dir = ''
    args.flow_f1_dir= ''
    args.flow_b2_dir = ''
    args.flow_f2_dir= ''


    if args.dataset == 'dvs17' or args.dataset == 'dvs17m'
        args.cats = ['bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow',
                    'cows', 'dance-twirl', 'dog', 'dogs-jump', 'drift-chicane', 'drift-straight', 'goat', 'gold-fish',
                    'horsejump-high', 'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
                        'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black', 'shooting', 'soapbox']
    else:
        args.cats = ['dog', 'cows', 'goat', 'camel', 'libby', 'parkour', 'soapbox', 'blackswan', 'bmx-trees', 
                            'kite-surf', 'car-shadow', 'breakdance', 'dance-twirl', 'scooter-black', 'drift-chicane', 
                            'motocross-jump', 'horsejump-high', 'drift-straight', 'car-roundabout', 'paragliding-launch']   
    
    if args.dataset == 'dvs17':
        args.obj_nums = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 2, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 ,
                    'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 5,
                        'horsejump-high': 2, 'india': 8, 'judo': 2, 'kite-surf': 3, 'lab-coat': 5, 'libby': 1, 'loading': 3, 'mbike-trick': 2,
                        'motocross-jump': 2, 'paragliding-launch': 3, 'parkour': 1, 'pigs': 3, 'scooter-black': 2, 'shooting': 2, 'soapbox': 3}
        args.gt_dir = ''
    else:
        args.obj_nums = {'bike-packing': 2, 'blackswan': 1, 'bmx-trees': 1, 'breakdance': 1, 'camel': 1, 'car-roundabout': 1, 'car-shadow':1 ,
                    'cows': 1, 'dance-twirl': 1, 'dog': 1, 'dogs-jump': 3, 'drift-chicane': 1, 'drift-straight': 1, 'goat': 1, 'gold-fish': 5,
                        'horsejump-high': 1, 'india': 3, 'judo': 2, 'kite-surf': 1, 'lab-coat': 1, 'libby': 1, 'loading': 3, 'mbike-trick': 1,
                        'motocross-jump': 1, 'paragliding-launch': 1, 'parkour': 1, 'pigs': 3, 'scooter-black': 1, 'shooting': 1, 'soapbox': 1}
        args.gt_dir = ''

    main(args)