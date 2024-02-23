import torch
from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import Embedding
from torch.nn import Parameter
from torchvision.models.vision_transformer import VisionTransformer as ViT

from packages.constants import *


class H4vdmNet(nn.Module):
    def __init__(self):
        """Constructor for the H4vdmNet class. This class is a PyTorch module that implements the H4VDM network.
        """
        super().__init__()

        self.intra_net = ViT(
            image_size = max(FRAME_WIDTH, FRAME_HEIGHT),
            patch_size = VIT1_PATCH_SIZE,
            num_layers = VIT1_DEPTH,
            num_heads = VIT1_NUM_HEADS,
            hidden_dim = VIT1_PROJECTION_DIMENSION,
            mlp_dim = VIT1_MLP_DIMENSION,
            num_classes=VIT1_OUTPUT_DIMENSION
            )

        self.inter_net = ViT(
            image_size = max(FRAME_WIDTH, FRAME_HEIGHT),
            patch_size = VIT1_PATCH_SIZE,
            num_layers = VIT1_DEPTH,
            num_heads = VIT1_NUM_HEADS,
            hidden_dim = VIT1_PROJECTION_DIMENSION,
            mlp_dim = VIT1_MLP_DIMENSION,
            num_classes=VIT1_OUTPUT_DIMENSION
            )

        self.frame_types_net = Embedding(EMBEDDING_VOCABULARY_SIZE, EMBEDDING_DIMENSION)
        
        self.mb_net = ViT(
            image_size = max(FRAME_WIDTH, FRAME_HEIGHT),
            patch_size = VIT2_PATCH_SIZE,
            num_layers = VIT2_DEPTH,
            num_heads = VIT2_NUM_HEADS,
            hidden_dim = VIT2_PROJECTION_DIMENSION,
            mlp_dim = VIT2_MLP_DIMENSION,
            num_classes=VIT2_OUTPUT_DIMENSION
            )
        
        self.luma_net = ViT(
            image_size = max(FRAME_WIDTH, FRAME_HEIGHT),
            patch_size = VIT2_PATCH_SIZE,
            num_layers = VIT2_DEPTH,
            num_heads = VIT2_NUM_HEADS,
            hidden_dim = VIT2_PROJECTION_DIMENSION,
            mlp_dim = VIT2_OUTPUT_DIMENSION,
            num_classes=VIT2_OUTPUT_DIMENSION
            )

        self.special_vectors = Parameter(torch.randn(4, 1, INTERMEDIATE_OUTPUTS_DIMENSION))

        self.joint_net = TransformerEncoder(torch.nn.TransformerEncoderLayer(INTERMEDIATE_OUTPUTS_DIMENSION, JAN_N_HEADS), JAN_N_LAYERS)

        self.linear = nn.Linear(INTERMEDIATE_OUTPUTS_DIMENSION*JAN_INPUT_SIZE, OUTPUT_DIMENSION)

    def forward(self, gop, debug = False, device = "cpu"):
        """Forward pass of the H4VDM network.

        Args:
            gop (Gop): The GOP to be processed.
            debug (bool, optional): Choose whether to print shapes at all steps for debug purposes. Defaults to False.

        Returns:
            Tensor: The output of the H4VDM network.
        """
        # Inter
        tmp = gop.get_intra_frame_as_tensor()
        tmp = tmp.to(device)
        if debug:
            print("Input intra frame shape:", tmp.shape)
        intra = self.intra_net(tmp)
        if debug:
            print("Output intra feature shape:", intra.shape)
        
        # Intra
        tmp = gop.get_inter_frames_as_tensor()
        tmp = tmp.to(device)
        if debug:
            print("Input inter frames shape:", tmp.shape)
        inter = self.inter_net(tmp)
        if debug:
            print("Output inter feature shape:", inter.shape)
        
        # Frame types
        tmp = gop.get_frame_types_as_tensor()
        tmp = tmp.to(device)
        if debug:
            print("Input frame types shape:", tmp.shape)
        frame_types = self.frame_types_net(tmp)
        if debug:
            print("Output frame types feature shape:", frame_types.shape)
        
        # MB types
        tmp = gop.get_macroblock_images_as_tensor()
        tmp = tmp.to(device)
        if debug:
            print("Input MB types shape:", tmp.shape)
        mb = self.mb_net(tmp)
        if debug:
            print("Output MB types feature shape:", mb.shape)

        # Luma QPs
        tmp = gop.get_luma_qp_images_as_tensor()
        tmp = tmp.to(device)
        if debug:
            print("Input luma QPs shape:", tmp.shape)
        luma = self.luma_net(tmp)
        if debug:
            print("Output luma QPs feature shape:", luma.shape)

        # Concatenate all inputs and intermediate vectors to form jan_input
        if debug:
            print("Special vectors shape:", self.special_vectors.shape)

        jan_input = torch.cat(
            (intra,
             self.special_vectors[0],
             inter,
             self.special_vectors[1],
             frame_types,
             self.special_vectors[2],
             mb,
             self.special_vectors[3],
             luma),
                dim = 0)
        
        if debug:
            print("Input Joint Analysis Network shape:", jan_input.shape)

        result = self.joint_net(jan_input)
        result = result.reshape(-1)
        if debug:
            print("Output Joint Analysis Network shape:", result.shape)

        # Linear projection
        result = self.linear(result)
        if debug:
            print("Output linear projection shape:", result.shape)
        return result