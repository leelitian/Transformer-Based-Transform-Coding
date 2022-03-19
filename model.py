from compressai.models.google import CompressionModel
from compressai.models.google import get_scale_table
from compressai.models.utils import update_registered_buffers

from layers.swin_transformer import *

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class SwinHyperprior(CompressionModel):
    """
    TRANSFORMER-BASED TRANSFORM CODING (ICLR 2022), Model: SwinT-Hyperprior (small)
    (wg; wh) = (8; 4)
    (C1; C2; C3; C4; C5; C6) = (96; 128; 160; 192; 96; 128),
    (d1; d2; d3; d4; d5; d6) = (2; 2; 6; 2; 5; 1)
    """

    def __init__(self, N=128, **kwargs):
        super().__init__(N, **kwargs)

        self.channels = [96, 128, 160, 192, 96, 128]
        self.depths = [2, 2, 6, 2, 5, 1]
        self.window_sizes = [8, 4]
        self.num_heads = [c // 32 for c in self.channels]

        self.gaussian_conditional = GaussianConditional(None)

        self.g_a = nn.Sequential(
            # stage 1
            PatchMerging(dims=(3, self.channels[0]), norm_layer=None),
            *[SwinTransformerBlock(dim=self.channels[0],
                                   num_heads=self.num_heads[0], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[0])],
            # stage 2
            PatchMerging(dims=(self.channels[0], self.channels[1])),
            *[SwinTransformerBlock(dim=self.channels[1],
                                   num_heads=self.num_heads[1], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[1])],
            # stage 3
            PatchMerging(dims=(self.channels[1], self.channels[2])),
            *[SwinTransformerBlock(dim=self.channels[2],
                                   num_heads=self.num_heads[2], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[2])],
            # stage 4
            PatchMerging(dims=(self.channels[2], self.channels[3])),
            *[SwinTransformerBlock(dim=self.channels[3],
                                   num_heads=self.num_heads[3], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[3])],
        )

        self.h_a = nn.Sequential(
            # stage 5
            PatchMerging(dims=(self.channels[3], self.channels[4])),
            *[SwinTransformerBlock(dim=self.channels[4],
                                   num_heads=self.num_heads[4], window_size=self.window_sizes[1],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[1] // 2) for i in range(self.depths[4])],
            # stage 6
            PatchMerging(dims=(self.channels[4], self.channels[5])),
            *[SwinTransformerBlock(dim=self.channels[5],
                                   num_heads=self.num_heads[5], window_size=self.window_sizes[1],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[1] // 2) for i in range(self.depths[5])],
        )

        self.h_s = nn.Sequential(
            # stage 6
            *[SwinTransformerBlock(dim=self.channels[5],
                                   num_heads=self.num_heads[5], window_size=self.window_sizes[1],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[1] // 2) for i in range(self.depths[5])],
            PatchSplitting(dims=(self.channels[5], self.channels[4])),
            # stage 5
            *[SwinTransformerBlock(dim=self.channels[4],
                                   num_heads=self.num_heads[4], window_size=self.window_sizes[1],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[1] // 2) for i in range(self.depths[4])],
            PatchSplitting(dims=(self.channels[4], 2 * self.channels[3])),
        )

        self.g_s = nn.Sequential(
            # stage 4
            *[SwinTransformerBlock(dim=self.channels[3],
                                   num_heads=self.num_heads[3], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[3])],
            PatchSplitting(dims=(self.channels[3], self.channels[2])),
            # stage 3
            *[SwinTransformerBlock(dim=self.channels[2],
                                   num_heads=self.num_heads[2], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[2])],
            PatchSplitting(dims=(self.channels[2], self.channels[1])),
            # stage 2
            *[SwinTransformerBlock(dim=self.channels[1],
                                   num_heads=self.num_heads[1], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[1])],
            PatchSplitting(dims=(self.channels[1], self.channels[0])),
            # stage 1
            *[SwinTransformerBlock(dim=self.channels[0],
                                   num_heads=self.num_heads[0], window_size=self.window_sizes[0],
                                   shift_size=0 if (i % 2 == 0) else self.window_sizes[0] // 2) for i in range(self.depths[0])],
            PatchSplitting(dims=(self.channels[0], 3), norm_layer=None),
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)           # B, H, W, C
        y = self.g_a(x)
        z = self.h_a(y)

        z = z.permute(0, 3, 1, 2)                           # B, C, H, W
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        z_hat = z_hat.permute(0, 2, 3, 1)   # B, H, W, C
        gaussian_params = self.h_s(z_hat)

        scales_hat, means_hat = gaussian_params.chunk(2, dim=-1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        x_hat = x_hat.permute(0, 3, 1, 2)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        x = x.permute(0, 2, 3, 1)
        y = self.g_a(x)
        z = self.h_a(y)

        z = z.permute(0, 3, 1, 2)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat.permute(0, 2, 3, 1)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=-1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat.permute(0, 2, 3, 1)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=-1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        x_hat = x_hat.permute(0, 3, 1, 2)

        return {"x_hat": x_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(
            scale_table, force=force)
        updated |= super().update(force=force)
        return updated


if __name__ == '__main__':
    net = SwinHyperprior().eval().cuda()

    with torch.no_grad():
        x = torch.rand((1, 3, 768, 512)).cuda()
        out = net(x)

    print(out['x_hat'].shape)
