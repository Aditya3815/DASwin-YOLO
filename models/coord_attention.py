import torch
import torch.nn as nn
import torch.nn.functional as F


class h_swish(nn.Module):
    """Hard-swish activation: x * ReLU6(x+3) / 6 — fast approximation of swish."""

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return x * self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    """
    Coordinate Attention (Hou et al. 2021, CVPR).

    Encodes spatial position information along X and Y axes independently via
    directional average pooling, then generates H- and W-wise attention maps.

    Args:
        inp (int): Input channels.
        oup (int): Output channels. When inp != oup, a 1×1 projection is added
                   to the identity branch so that skip + attention are compatible.
        reduction (int): Channel reduction ratio for the intermediate bottleneck.

    Shape:
        Input:  (N, inp, H, W)  or  (inp, H, W)   — 3D tensors are handled gracefully
        Output: (N, oup, H, W)  or  (oup, H, W)
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (N, C, H, 1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (N, C, 1, W)

        mip = max(8, inp // reduction)  # bottleneck width

        self.conv1  = nn.Conv2d(inp, mip, kernel_size=1, bias=False)
        self.bn1    = nn.BatchNorm2d(mip)
        self.act    = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, bias=False)

        # Projection for the identity branch when channel dims differ.
        # Required so that  identity (inp channels) * attention (oup channels) works.
        self.proj = (
            nn.Conv2d(inp, oup, kernel_size=1, bias=False)
            if inp != oup
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 3-D inputs (no batch dim) transparently
        was_3d = x.dim() == 3
        if was_3d:
            x = x.unsqueeze(0)

        identity = self.proj(x)         # (N, oup, H, W)

        n, c, h, w = x.size()
        x_h = self.pool_h(x)            # (N, c, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (N, c, W, 1) → transposed for concat

        # Joint encoding of H and W directions
        y = torch.cat([x_h, x_w], dim=2)  # (N, c, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # restore W dimension

        a_h = self.conv_h(x_h).sigmoid()  # (N, oup, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (N, oup, 1, W)

        out = identity * a_h * a_w         # broadcast over H and W axes

        if was_3d:
            out = out.squeeze(0)

        return out


class CoordAttMulti(nn.Module):
    """
    Applies Coordinate Attention to a list of multi-scale feature maps,
    using a dedicated :class:`CoordAtt` layer for each scale.

    Args:
        inp (int): Input channels (must match each feature map's channel count).
        oup (int): Output channels.
        reduction (int): Channel reduction ratio (default 32).
        num_levels (int): Number of feature pyramid levels (default 4).

    Shape:
        Input:  List of (N, inp, H_i, W_i) tensors, length = num_levels
        Output: List of (N, oup, H_i, W_i) tensors, length = num_levels
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32, num_levels: int = 4):
        super().__init__()
        self.ca_layers = nn.ModuleList(
            [CoordAtt(inp, oup, reduction) for _ in range(num_levels)]
        )

    def forward(self, x):
        return [layer(xi) for layer, xi in zip(self.ca_layers, x)]


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CoordAtt self-test:")

    # Test 1 — standard (inp == oup)
    m = CoordAtt(256, 256)
    out = m(torch.randn(1, 256, 40, 40))
    assert out.shape == (1, 256, 40, 40), f"Test 1 failed: {out.shape}"
    print(f"  ✅ inp==oup  : {out.shape}")

    # Test 2 — channel projection (inp != oup)
    m2 = CoordAtt(256, 128)
    out2 = m2(torch.randn(1, 256, 20, 20))
    assert out2.shape == (1, 128, 20, 20), f"Test 2 failed: {out2.shape}"
    print(f"  ✅ inp!=oup  : {out2.shape}")

    # Test 3 — 3D input (no batch dim)
    m3 = CoordAtt(64, 64)
    out3 = m3(torch.randn(64, 10, 10))
    assert out3.dim() == 3 and out3.shape == (64, 10, 10), f"Test 3 failed: {out3.shape}"
    print(f"  ✅ 3D input  : {out3.shape}")

    # Test 4 — CoordAttMulti
    multi = CoordAttMulti(inp=256, oup=256, num_levels=4)
    feats = [torch.randn(1, 256, s, s) for s in [160, 80, 40, 20]]
    outs = multi(feats)
    assert len(outs) == 4
    for i, (o, f) in enumerate(zip(outs, feats)):
        assert o.shape == f.shape, f"CoordAttMulti scale {i}: {o.shape} != {f.shape}"
    print(f"  ✅ CoordAttMulti: {[list(o.shape) for o in outs]}")

    print("All CoordAtt tests passed.")
