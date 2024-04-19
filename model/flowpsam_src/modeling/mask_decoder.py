# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)  # fIoU tokens
        self.obj_token = nn.Embedding(1, transformer_dim)  # MOS tokens
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.prompt_tokens = nn.Embedding(4, transformer_dim)  # 4 flow prompt tokens input to flow transformer (in flow prompt generator)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # For fIoU prediction
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # For MOS prediction with sigmoid output
        self.obj_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, 1, iou_head_depth, sigmoid_output = True
        )


    def forward(
        self,
        image_embeddings: torch.Tensor,
        flow_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          flow_embeddings (torch.Tensor): the flow embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of fiou
          torch.Tensor: batched predictions of mos
        """
        masks, iou_pred, obj_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            flow_embeddings=flow_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        # Select the correct mask or masks for output
        if multimask_output: # output 4 channels
            #mask_slice = slice(1, None)
            mask_slice = slice(0, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred, obj_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        flow_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.obj_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)  # 1 256    # 4 256   
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # B 5 256
        prompt_tokens = self.prompt_tokens.weight.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)  # B 4 256
        tokens = torch.cat((output_tokens, prompt_tokens, sparse_prompt_embeddings), dim=1)  # B 7 256
        # tokens consists of --- mos token x 1; fiou token x 1; mask tokens x 4; flow prompt tokens x 4; position prompt tokens x 2 (point prompt token x 1 + padded token x 1)
        
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] == tokens.shape[0]: # Conditioning on whether batch size matches (if batch size matches --- train)
            src = image_embeddings
            flow_src = flow_embeddings
        else: # if batch size does not match --- eval
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)  
            flow_src = torch.repeat_interleave(flow_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # B 256 64 64
        b, c, h, w = src.shape
        # Run the transformer
        # This transformer include: flow transformer in flow prompt generator + mask decoder in segmentation module
        flow_hs, hs, src = self.transformer(src, flow_src, pos_src, tokens)  # flow_hs is the output from flow transformer 
        obj_token_out = flow_hs[:, 0, :]  # B 256, for MOS (moving object score)
        iou_token_out = hs[:, 0, :]  # B 256, for fIoU (foreground IoU)
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] # B 4 256, for mask prediction


        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # B 256 64 64
        upscaled_embedding = self.output_upscaling(src)  # B 32 256 256
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))  # B 32 each
        hyper_in = torch.stack(hyper_in_list, dim=1)  # B 4 32
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)  # B 4 32 x B 32 65536 -> B 4 256 256

        # Generate fIoU (foreground IoU) scores
        iou_pred = self.iou_prediction_head(iou_token_out)  # B 4
        # Generate MOS (moving object scores)
        obj_pred = self.obj_prediction_head(obj_token_out) # B 1

        return masks, iou_pred, obj_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
