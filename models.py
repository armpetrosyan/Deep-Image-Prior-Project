"""The Deep Image Prior "skip" network (paper Section 4, Fig. 21).

The network is an hourglass made of nested levels. Each level halves the
resolution, runs the next level, doubles the resolution back and joins the
result with a small skip branch.
"""

import torch
import torch.nn as nn


def conv(in_ch, out_ch, k, stride=1):
    """k x k convolution with reflection padding instead of zero padding."""
    if k == 1:
        return nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=0)
    pad = nn.ReflectionPad2d(k // 2)
    layer = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=0)
    return nn.Sequential(pad, layer)


def bn_act(ch):
    """BatchNorm followed by LeakyReLU(0.2)."""
    return nn.Sequential(nn.BatchNorm2d(ch), nn.LeakyReLU(0.2))


class Level(nn.Module):
    """One resolution level of the hourglass."""

    def __init__(self, in_ch, levels_left, ch_down, ch_up, ch_skip, upsample):
        super().__init__()

        # Deeper branch: downsample with a stride-2 conv, then one more conv.
        self.down = nn.Sequential(
            conv(in_ch, ch_down, 3, stride=2),
            bn_act(ch_down),
            conv(ch_down, ch_down, 3),
            bn_act(ch_down),
        )

        # The next level down, if there is one.
        if levels_left > 1:
            self.deeper = Level(ch_down, levels_left - 1,
                                ch_down, ch_up, ch_skip, upsample)
            deeper_ch = ch_up
        else:
            self.deeper = nn.Identity()
            deeper_ch = ch_down

        self.up = nn.Upsample(scale_factor=2, mode=upsample)

        # Skip branch: a cheap 1x1 conv at this level's resolution.
        if ch_skip > 0:
            self.skip = nn.Sequential(conv(in_ch, ch_skip, 1), bn_act(ch_skip))
        else:
            self.skip = None

        # Layers applied after joining the skip and deeper branches.
        joined_ch = ch_skip + deeper_ch
        self.after = nn.Sequential(
            nn.BatchNorm2d(joined_ch),
            conv(joined_ch, ch_up, 3),
            bn_act(ch_up),
            conv(ch_up, ch_up, 1),
            bn_act(ch_up),
        )

    def forward(self, x):
        deeper = self.down(x)
        deeper = self.deeper(deeper)
        deeper = self.up(deeper)
        if self.skip is not None:
            joined = torch.cat([self.skip(x), deeper], dim=1)
        else:
            joined = deeper
        return self.after(joined)


class SkipNet(nn.Module):
    """Hourglass of `depth` levels, then a 1x1 conv and a Sigmoid (output in [0, 1])."""

    def __init__(self, in_ch=32, out_ch=3, depth=5, ch_down=128, ch_up=128,
                 ch_skip=4, upsample="bilinear"):
        super().__init__()
        self.depth = depth
        self.body = Level(in_ch, depth, ch_down, ch_up, ch_skip, upsample)
        self.head = nn.Sequential(conv(ch_up, out_ch, 1), nn.Sigmoid())

    def forward(self, z):
        # Each level halves the size, so H and W must survive `depth` halvings.
        height, width = z.shape[-2:]
        factor = 2 ** self.depth
        if height % factor != 0 or width % factor != 0:
            raise ValueError(f"Input size {height}x{width} must be divisible by "
                             f"2**depth = {factor} (depth={self.depth}).")
        return self.head(self.body(z))


def skip_net(depth=5, ch=128, skip_ch=4, upsample="bilinear"):
    """SkipNet used in the experiments. skip_ch=0 removes all skip connections."""
    return SkipNet(in_ch=32, out_ch=3, depth=depth, ch_down=ch, ch_up=ch,
                   ch_skip=skip_ch, upsample=upsample)


def count_params(net):
    """Number of trainable parameters."""
    total = 0
    for p in net.parameters():
        if p.requires_grad:
            total += p.numel()
    return total


if __name__ == "__main__":
    # Quick self-check on CPU with a small input.
    torch.manual_seed(0)
    z = torch.rand(1, 32, 64, 64) * 0.1

    net = skip_net()
    out = net(z)
    in_range = out.min().item() >= 0 and out.max().item() <= 1
    print("default:   out", tuple(out.shape),
          f"range [{out.min():.3f}, {out.max():.3f}]",
          "in [0, 1]:", in_range,
          f"params={count_params(net):,}")

    net_no_skip = skip_net(skip_ch=0)
    out = net_no_skip(z)
    print("no skips   out", tuple(out.shape), f"params={count_params(net_no_skip):,}")

    net_shallow = skip_net(depth=3)
    out = net_shallow(z)
    print("depth=3    out", tuple(out.shape), f"params={count_params(net_shallow):,}")

    try:
        net(torch.rand(1, 32, 50, 50))
    except ValueError as error:
        print("50x50 input -> ValueError:", error)
