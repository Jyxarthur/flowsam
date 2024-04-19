import os
import cv2
import torch
import numpy as np
from model.flowpsam import FlowPSAM
from model.flowisam import FlowISAM
from segment_anything.build_sam import sam_model_registry
from model.flowpsam_src.build_sam import sam_model_registry as flowp_sam_model_registry
from model.flowisam_src.build_sam import sam_model_registry as flowi_sam_model_registry

def config_model(args):
    if args.model == "flowpsam":
        ref_rgb_sam = sam_model_registry[args.rgb_encoder](checkpoint=args.rgb_encoder_ckpt_path)
        rgb_image_encoder = ref_rgb_sam.image_encoder
        prompt_encoder = ref_rgb_sam.prompt_encoder
        flow_image_encoder = sam_model_registry[args.flow_encoder](checkpoint=args.flow_encoder_ckpt_path).image_encoder
        '''
        For flowp-sam, the mask_decoder here includes the both (i) the flow transformer (in flow prompt generator)
                                                            and (ii) the mask decoder (in segmentation module)
        '''
        mask_decoder = flowp_sam_model_registry[args.rgb_encoder](checkpoint=args.rgb_encoder_ckpt_path).mask_decoder
        if args.ckpt_path is not None:
            mask_decoder_ckpt = torch.load(args.ckpt_path)['model_state_dict']
            mask_decoder.load_state_dict(mask_decoder_ckpt)
        flowsam = FlowPSAM(rgb_encoder=rgb_image_encoder, flow_encoder=flow_image_encoder, 
                            mask_decoder=mask_decoder, prompt_encoder=prompt_encoder).to(device = "cuda")
        return flowsam

    elif args.model == "flowisam":
        ref_flow_sam = sam_model_registry[args.flow_encoder](checkpoint=args.flow_encoder_ckpt_path)
        flow_image_encoder = ref_flow_sam.image_encoder
        prompt_encoder = ref_flow_sam.prompt_encoder
        mask_decoder = flowi_sam_model_registry[args.flow_encoder](checkpoint=args.flow_encoder_ckpt_path).mask_decoder
        if args.ckpt_path is not None:
            mask_decoder_ckpt = torch.load(args.ckpt_path)['model_state_dict']
            mask_decoder.load_state_dict(mask_decoder_ckpt)
        flowsam = FlowISAM(flow_encoder=flow_image_encoder, 
                            mask_decoder=mask_decoder, prompt_encoder=prompt_encoder).to(device = "cuda")
        return flowsam