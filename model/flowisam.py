import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

class FlowISAM(nn.Module):
    def __init__(
        self,
        flow_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.flow_encoder = flow_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.flow_feature = None
    
    def forward(self, flow, points, use_cache = False):
        if self.flow_feature is None and use_cache:
            g = flow.shape[1]
            flow = einops.rearrange(flow, "B g c h w -> (B g) c h w")
            flow_embedding = self.flow_encoder(flow) # ((B g), 256, 64, 64)
            flow_embedding = einops.rearrange(flow_embedding, "(B g) c h w -> B g c h w", g = g)
            flow_embedding = torch.mean(flow_embedding, 1)  # averaging over different frame gaps
            self.flow_feature = flow_embedding

        elif not (self.flow_feature is None) and use_cache:
            flow_embedding = self.flow_feature
            
        else:  # The cases that do not need cache (i.e., the training case)
            g = flow.shape[1]
            flow = einops.rearrange(flow, "B g c h w -> (B g) c h w")
            flow_embedding = self.flow_encoder(flow) # ((B g), 256, 64, 64)
            flow_embedding = einops.rearrange(flow_embedding, "(B g) c h w -> B g c h w", g = g)
            flow_embedding = torch.mean(flow_embedding, 1)  # averaging over different frame gaps
            

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        low_res_masks, fiou_predictions = self.mask_decoder(
            image_embeddings=flow_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=True,
        )
        
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(flow.shape[-2], flow.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        
        return ori_res_masks, fiou_predictions


