import torch
from torch import nn
from torch.nn import Transformer
from torch.nn import Embedding
from torch.nn import Parameter
from torchvision.models.vision_transformer import VisionTransformer as ViT

from packages.constants import *


class H4vdmNet(nn.Module):
    def __init__(self, gop, ):
        super().__init__()

        # self.intra_net = ViT(
        #     image_size = (gop.frame_width, gop.frame_height),
        #     patch_size = (VIT1_PATCH_SIZE, VIT1_PATCH_SIZE),
        #     num_classes = I_DONT_KNOW_WHAT_TO_DO_WITH_THIS,
        #     dim = 
            # )

        # self.inter_net = ViT(
        #     self,
        #     image_size = (gop.frame_width, gop.frame_height),
        #     patch_size = (VIT2_PATCH_SIZE, VIT2_PATCH_SIZE),
        #     num_layers: int,
        #     num_heads: int,
        #     hidden_dim: int,
        #     mlp_dim: int,
        #     dropout: float = 0.0,
        #     attention_dropout: float = 0.0,
        #     num_classes: int = 1000,
        #     representation_size: Optional[int] = None,
        #     norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        #     conv_stem_configs: Optional[List[ConvStemConfig]] = None,

        self.intra_net = 0
        # self.inter_net = ViT(__vit_constructor_parameters_tbd__)
        self.frame_types_net = Embedding(EMBEDDING_VOCABULARY_SIZE, EMBEDDING_DIMENSION)
        # self.frame_types_net = Embedding(__embedding_constructor_parameters_tbd__)
        # self.mb_net = ViT(__vit_constructor_parameters_tbd__)
        # self.luma_net = ViT(__vit_constructor_parameters_tbd__)

        self.special_vectors = Parameter(torch.randn(INTERMEDIATE_OUTPUTS_DIMENSION, 4))

        # jan_input_size = 4 * L + 5
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