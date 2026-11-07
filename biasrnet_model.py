"""BIASRNet (Burst Intermediate Alignment Super-Resolution Net) -- multi-frame
x4 super-resolution for real thermal video bursts.

NOTE: this is a vendored copy of the model definition from the separate
BIASRNet repo (BIADSRNet's frozen backbone comes from a BIASRNet checkpoint
trained with that repo's train.py). It's duplicated here, rather than
imported across repos, so this repo can be cloned and run on its own. Keep
this file in sync with the BIASRNet repo's model.py if that one changes.

Pipeline: per-frame RRDB feature extraction -> 3-level feature pyramid ->
EDVR-style PCD (Pyramid, Cascading, Deformable-conv) alignment of every
neighbor frame's features onto the center frame's features -> channel-concat
fusion -> RRDB body -> pixel-shuffle x4 upsample, with the center frame's
bicubic upsample kept as a long residual skip.

Reference (PCD alignment module): Wang et al., "EDVR: Video Restoration with
Enhanced Deformable Convolutional Networks", CVPRW 2019 (Fig. 2/3).

Expects an already-aligned burst as input -- see dataset.py's
`align_to_center` (classical Farneback optical flow) for how the burst
fed into this model should be prepared; PCD then performs a second,
learned, feature-space re-alignment on top of that classical warp.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * growth, num_feat, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, num_feat=64, growth=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, growth)
        self.rdb2 = ResidualDenseBlock(num_feat, growth)
        self.rdb3 = ResidualDenseBlock(num_feat, growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + 0.2 * out


class PixelShuffleUpsample(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        self.conv = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


def _match(x, ref):
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    return x


class PCDAlignment(nn.Module):
    """Aligns one neighbor frame's 3-level feature pyramid onto the center
    (reference) frame's pyramid, coarse-to-fine, plus one cascading
    refinement pass at full resolution."""

    def __init__(self, num_feat=64, deform_ks=3, deform_groups=1):
        super().__init__()
        pad = deform_ks // 2
        offset_ch = 2 * deform_groups * deform_ks * deform_ks

        self.offset_conv1_l3 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.offset_out_l3 = nn.Conv2d(num_feat, offset_ch, 3, 1, 1)
        self.dcn_l3 = DeformConv2d(num_feat, num_feat, deform_ks, padding=pad)

        self.offset_conv1_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_out_l2 = nn.Conv2d(num_feat, offset_ch, 3, 1, 1)
        self.dcn_l2 = DeformConv2d(num_feat, num_feat, deform_ks, padding=pad)
        self.feat_conv_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.offset_conv1_l1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2_l1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_out_l1 = nn.Conv2d(num_feat, offset_ch, 3, 1, 1)
        self.dcn_l1 = DeformConv2d(num_feat, num_feat, deform_ks, padding=pad)
        self.feat_conv_l1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_offset_out = nn.Conv2d(num_feat, offset_ch, 3, 1, 1)
        self.cas_dcn = DeformConv2d(num_feat, num_feat, deform_ks, padding=pad)

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, nbr_feat_l, ref_feat_l):
        # L3 (coarsest)
        offset_l3_feat = self.lrelu(self.offset_conv1_l3(torch.cat([nbr_feat_l[2], ref_feat_l[2]], dim=1)))
        offset_l3_feat = self.lrelu(self.offset_conv2_l3(offset_l3_feat))
        feat_l3 = self.lrelu(self.dcn_l3(nbr_feat_l[2], self.offset_out_l3(offset_l3_feat)))

        # L2
        offset = self.lrelu(self.offset_conv1_l2(torch.cat([nbr_feat_l[1], ref_feat_l[1]], dim=1)))
        offset_up = _match(self.upsample(offset_l3_feat) * 2, offset)
        offset_l2_feat = self.lrelu(self.offset_conv2_l2(torch.cat([offset, offset_up], dim=1)))
        feat_l2 = self.dcn_l2(nbr_feat_l[1], self.offset_out_l2(offset_l2_feat))
        feat_l3_up = _match(self.upsample(feat_l3), feat_l2)
        feat_l2 = self.lrelu(self.feat_conv_l2(torch.cat([feat_l2, feat_l3_up], dim=1)))

        # L1 (finest)
        offset = self.lrelu(self.offset_conv1_l1(torch.cat([nbr_feat_l[0], ref_feat_l[0]], dim=1)))
        offset_up = _match(self.upsample(offset_l2_feat) * 2, offset)
        offset_l1_feat = self.lrelu(self.offset_conv2_l1(torch.cat([offset, offset_up], dim=1)))
        feat_l1 = self.dcn_l1(nbr_feat_l[0], self.offset_out_l1(offset_l1_feat))
        feat_l2_up = _match(self.upsample(feat_l2), feat_l1)
        feat_l1 = self.feat_conv_l1(torch.cat([feat_l1, feat_l2_up], dim=1))

        # cascading refinement against the full-res reference feature
        offset = self.lrelu(self.cas_offset_conv1(torch.cat([feat_l1, ref_feat_l[0]], dim=1)))
        offset_feat = self.lrelu(self.cas_offset_conv2(offset))
        offset_cas = self.cas_offset_out(offset_feat)
        return self.lrelu(self.cas_dcn(feat_l1, offset_cas))


class BIASRNet(nn.Module):
    """Input:  (B, n_frames, H, W) Farneback-aligned LR burst, [0, 1] float.
    Output: (B, 1, 4H, 4W) restored HR center frame, [0, 1] float.

    deform_groups=1 is the released/best-PSNR configuration -- deform_groups=8
    was tested and gave a negligible extra gain (+0.01dB) for +3% params, so
    it isn't worth the extra cost here.
    """

    def __init__(self, n_frames=5, num_feat=64, growth=32,
                 num_extract_blocks=2, num_body_blocks=10, scale=4, deform_groups=1):
        super().__init__()
        self.n_frames = n_frames
        self.center_idx = n_frames // 2
        self.scale = scale

        self.conv_first = nn.Conv2d(1, num_feat, 3, 1, 1)
        self.feat_extract = nn.Sequential(*[RRDB(num_feat, growth) for _ in range(num_extract_blocks)])
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.pcd = PCDAlignment(num_feat, deform_groups=deform_groups)
        self.fusion = nn.Conv2d(n_frames * num_feat, num_feat, 1, 1)

        self.body = nn.Sequential(*[RRDB(num_feat, growth) for _ in range(num_body_blocks)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample = nn.Sequential(*[PixelShuffleUpsample(num_feat) for _ in range(2)])
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.act_hr = nn.LeakyReLU(0.2, inplace=True)
        self.conv_last = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def _pyramid(self, feat_l1):
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))
        return feat_l1, feat_l2, feat_l3

    def forward(self, burst):
        b, n, h, w = burst.shape
        x = burst.reshape(b * n, 1, h, w)
        feat = self.lrelu(self.conv_first(x))
        feat = self.feat_extract(feat)

        l1, l2, l3 = self._pyramid(feat)
        l1 = l1.view(b, n, -1, h, w)
        l2 = l2.view(b, n, -1, l2.shape[-2], l2.shape[-1])
        l3 = l3.view(b, n, -1, l3.shape[-2], l3.shape[-1])

        ref = [l1[:, self.center_idx], l2[:, self.center_idx], l3[:, self.center_idx]]

        aligned = []
        for i in range(n):
            if i == self.center_idx:
                aligned.append(l1[:, i])
            else:
                nbr = [l1[:, i], l2[:, i], l3[:, i]]
                aligned.append(self.pcd(nbr, ref))

        fused_in = torch.cat(aligned, dim=1)
        fused = self.lrelu(self.fusion(fused_in))

        body_feat = self.conv_body(self.body(fused))
        feat = fused + body_feat

        feat = self.upsample(feat)
        out = self.conv_last(self.act_hr(self.conv_hr(feat)))

        center = burst[:, self.center_idx:self.center_idx + 1]
        base = F.interpolate(center, scale_factor=self.scale, mode='bicubic', align_corners=False)
        return (out + base).clamp(0, 1)


if __name__ == '__main__':
    model = BIASRNet()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'BIASRNet parameters: {n_params:.2f}M')
    x = torch.rand(2, 5, 64, 64)
    out = model(x)
    print('input', x.shape, '-> output', out.shape)
