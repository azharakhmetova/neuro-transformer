import math
import torch
import typing as t
from einops import repeat
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from v1t.models.layers.embeddings import PositionalEncoding




class PatchShifting(nn.Module):
    """Patch shifting for Shifted Patch Tokenization"""

    def __init__(self, patch_size: int):
        super(PatchShifting, self).__init__()
        self.shift = int(patch_size * (1 / 2))

    def forward(self, inputs: torch.Tensor):
        """4 diagonal directions padding"""
        padded_inputs = F.pad(
            input=inputs,
            pad=(self.shift, self.shift, self.shift, self.shift),
            mode="constant",
            value=0,
        )
        left_upper = padded_inputs[..., : -self.shift * 2, : -self.shift * 2]
        right_upper = padded_inputs[..., : -self.shift * 2, self.shift * 2 :]
        left_bottom = padded_inputs[..., self.shift * 2 :, : -self.shift * 2]
        right_bottom = padded_inputs[..., self.shift * 2 :, self.shift * 2 :]
        outputs = torch.cat(
            [inputs, left_upper, right_upper, left_bottom, right_bottom],
            dim=1,
        )
        return outputs


class Image2Patches(nn.Module):
    """
    patch embedding mode:
        0 - nn.Unfold to extract patches
        1 - nn.Conv2D to extract patches
        2 - Shifted Patch Tokenization https://arxiv.org/abs/2112.13492v1
        3 - nn.Unfold with Dual PatchNorm https://openreview.net/forum?id=jgMqve6Qhw
    """

    def __init__(
        self,
        image_shape: t.Tuple[int, int, int],
        patch_mode: int,
        patch_size: int,
        stride: int,
        emb_dim: int,
        dropout: float = 0.0,
        pe_mode: str = "1d",
    ):
        super(Image2Patches, self).__init__()
        assert 1 <= stride <= patch_size
        c, h, w = image_shape
        self.input_shape = image_shape
        self.pe_mode = pe_mode

        num_patches = self.unfold_dim(h, w, patch_size=patch_size, stride=stride)
        match patch_mode:
            case 0:
                patch_dim = patch_size * patch_size * c
                self.projection = nn.Sequential(
                    nn.Unfold(kernel_size=patch_size, stride=stride),
                    Rearrange("b c l -> b l c"),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                )
            case 1:
                self.projection = nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=emb_dim,
                        kernel_size=patch_size,
                        stride=stride,
                    ),
                    Rearrange("b c h w -> b (h w) c"),
                )
            case 2:
                patch_dim = patch_size * patch_size * (c + 4)
                self.projection = nn.Sequential(
                    PatchShifting(patch_size=patch_size),
                    nn.Unfold(kernel_size=patch_size, stride=stride),
                    Rearrange("b c l -> b l c"),
                    nn.LayerNorm(normalized_shape=patch_dim),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                )
            case 3:
                patch_dim = patch_size * patch_size * c
                self.projection = nn.Sequential(
                    nn.Unfold(kernel_size=patch_size, stride=stride),
                    Rearrange("b c l -> b l c"),
                    nn.LayerNorm(normalized_shape=patch_dim),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                    nn.LayerNorm(normalized_shape=emb_dim),
                )
            case _:
                raise NotImplementedError(f"--patch_mode {patch_mode} not implemented.")
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # num_patches += 1
        if pe_mode == "1d":
            self.pos_embedding = PositionalEncoding(
                d_model=emb_dim,
                dropout=dropout,
                max_len=num_patches,
                learned=False,
                mode=pe_mode,
                ) #nn.Parameter(torch.randn(num_patches, emb_dim))
        elif pe_mode == "2d":
            height, width = self.find_shape(num_patches)
            self.height = height
            self.width = width
            self.pos_embedding = PositionalEncoding(
                d_model=emb_dim,
                dropout=dropout,
                height=height,
                width=width,
                learned=False,
                mode=pe_mode,
                )
        self.dropout = nn.Dropout(p=dropout)
        self.num_patches = num_patches
        self.output_shape = (num_patches, emb_dim)

        self.apply(self.init_weight)

    @staticmethod
    def find_shape(num_patches: int):
        dim1 = math.ceil(math.sqrt(num_patches))
        while num_patches % dim1 != 0 and dim1 > 0:
            dim1 -= 1
        dim2 = num_patches // dim1
        return dim1, dim2
    
    @staticmethod
    def unfold_dim(h: int, w: int, patch_size: int, padding: int = 0, stride: int = 1):
        l = lambda s: math.floor(((s + 2 * padding - patch_size) / stride) + 1)
        return l(h) * l(w)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

    def forward(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        patches = self.projection(inputs)
        # cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        # outputs = torch.cat((cls_tokens, patches), dim=1)
        if self.pe_mode == "1d":
            outputs = patches + self.pos_embedding(outputs)
        elif self.pe_mode == "2d":
            patches = patches.reshape(batch_size, self.height, self.width, -1)
            # print("outputs shape before pos_embedding", outputs.shape)
            outputs = (patches + self.pos_embedding(patches)).reshape(batch_size, self.num_patches, -1)
        outputs = self.dropout(outputs)
        return outputs
