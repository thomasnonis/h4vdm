import torch
from torch import nn
from torch.nn import Transformer
from vit import ViT
# from vit import Transformer

class H4vdmNet(nn.Module):
    def __init__(self, __constructor_parameters_tbd__):
        super().__init__()

        # self.intra_net = ViT(__vit_constructor_parameters_tbd__)
        # self.inter_net = ViT(__vit_constructor_parameters_tbd__)
        # self.frame_types_net = Embedding(__embedding_constructor_parameters_tbd__)
        # self.mb_net = ViT(__vit_constructor_parameters_tbd__)
        # self.luma_net = ViT(__vit_constructor_parameters_tbd__)

        # jan_input_size = 4 * L + 5
        # self.joint_net = Transformer(jan_input_size, ...)

    def forward(self, gop):
        # intra = self.intra_net(gop.intra_frame)
        # inter = self.inter_net(gop.inter_frame)
        # frame_types = self.frame_types_net(gop.frame_types)
        # mb = self.mb_net(gop.mb_types)
        # luma = self.luma_net(gop.luma_qps)

        # special_vectors = ???
        # return self.joint_net(intra, inter, frame_types, mb, luma, special_vectors)
        pass