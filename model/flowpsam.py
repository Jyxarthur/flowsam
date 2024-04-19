import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class FlowPSAM(nn.Module):
    def __init__(
        self,
        rgb_encoder,
        flow_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.flow_encoder = flow_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.rgb_feature = None
        self.flow_feature = None

    def forward(self, rgb, flow, points, use_cache = False):
        if (self.rgb_feature is None or self.flow_feature is None) and use_cache:
            g = flow.shape[1]
            flow = einops.rearrange(flow, "B g c h w -> (B g) c h w")
            flow_embedding = self.flow_encoder(flow) # ((B g), 256, 64, 64)
            flow_embedding = einops.rearrange(flow_embedding, "(B g) c h w -> B g c h w", g = g)
            flow_embedding = torch.mean(flow_embedding, 1)  # averaging over different frame gaps
            self.flow_feature = flow_embedding

            rgb_embedding = self.rgb_encoder(rgb) # (B, 256, 64, 64)
            self.rgb_feature = rgb_embedding

        elif not (self.rgb_feature is None or self.flow_feature is None) and use_cache:
            rgb_embedding = self.rgb_feature
            flow_embedding = self.flow_feature
            
        else:  # The cases that do not need cache (i.e., the training case)
            g = flow.shape[1]
            flow = einops.rearrange(flow, "B g c h w -> (B g) c h w")
            flow_embedding = self.flow_encoder(flow) # ((B g), 256, 64, 64)
            flow_embedding = einops.rearrange(flow_embedding, "(B g) c h w -> B g c h w", g = g)
            flow_embedding = torch.mean(flow_embedding, 1)  # averaging over different frame gaps
            rgb_embedding = self.rgb_encoder(rgb) # (B, 256, 64, 64)
            
        
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        low_res_masks, fiou_predictions, mos_predictions = self.mask_decoder(
            image_embeddings=rgb_embedding,  # (B, 256, 64, 64)
            flow_embeddings=flow_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(rgb.shape[-2], rgb.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        return ori_res_masks, fiou_predictions, mos_predictions
