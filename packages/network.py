import torch
from torch import nn
from torch.nn import Transformer
from torch.nn import Embedding
from torch.nn import Parameter
from torchvision.models.vision_transformer import VisionTransformer as ViT

from packages.constants import *


class H4vdmNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.intra_net = ViT(
            image_size = (FRAME_WIDTH, FRAME_HEIGHT, 3),
            patch_size = (VIT1_PATCH_SIZE, VIT1_PATCH_SIZE),
            num_layers = VIT1_DEPTH,
            num_heads = VIT1_NUM_HEADS,
            hidden_dim = VIT1_PROJECTION_DIMENSION,
            mlp_dim = VIT1_OUTPUT_DIMENSION
            )

        self.intra_net = ViT(
            image_size = (FRAME_WIDTH, FRAME_HEIGHT, 3),
            patch_size = (VIT1_PATCH_SIZE, VIT1_PATCH_SIZE),
            num_layers = VIT1_DEPTH,
            num_heads = VIT1_NUM_HEADS,
            hidden_dim = VIT1_PROJECTION_DIMENSION,
            mlp_dim = VIT1_OUTPUT_DIMENSION
            )

        self.frame_types_net = Embedding(EMBEDDING_VOCABULARY_SIZE, EMBEDDING_DIMENSION)
        
        self.mb_net = ViT(
            image_size = (FRAME_WIDTH, FRAME_HEIGHT),
            patch_size = (VIT2_PATCH_SIZE, VIT2_PATCH_SIZE),
            num_layers = VIT2_DEPTH,
            num_heads = VIT2_NUM_HEADS,
            hidden_dim = VIT2_PROJECTION_DIMENSION,
            mlp_dim = VIT2_OUTPUT_DIMENSION
            )
        
        self.luma_net = ViT(
            image_size = (FRAME_WIDTH, FRAME_HEIGHT),
            patch_size = (VIT2_PATCH_SIZE, VIT2_PATCH_SIZE),
            num_layers = VIT2_DEPTH,
            num_heads = VIT2_NUM_HEADS,
            hidden_dim = VIT2_PROJECTION_DIMENSION,
            mlp_dim = VIT2_OUTPUT_DIMENSION
            )

        self.special_vectors = Parameter(torch.randn(INTERMEDIATE_OUTPUTS_DIMENSION, 4))

        self.joint_net = Transformer(JAN_INPUT_SIZE, JAN_N_HEADS, JAN_N_LAYERS, JAN_N_LAYERS)

    def forward(self, gop):
        intra = self.intra_net(gop.intra_frame)
        inter = self.inter_net(gop.inter_frames)
        frame_types = self.frame_types_net(gop.frame_types)
        mb = self.mb_net(gop.mb_types)
        luma = self.luma_net(gop.luma_qps)

        # append intermediate vectors
        # TODO: might need to transpose special_vectors
        intra = torch.cat(intra, self.special_vectors[:,0], dim = 1)
        inter = torch.cat(inter, self.special_vectors[:,1], dim = 1)
        frame_types = torch.cat(self.frame_types_net, self.special_vectors[:,2], dim = 1)
        mb = torch.cat(self.mb_net, self.special_vectors[:,3], dim = 1)

        # concatenate all inputs to form jan_input
        jan_input = torch.cat(intra, inter, frame_types, mb, luma, dim = 1)

        return self.joint_net(jan_input)
        pass