"""CNN architectures for real / Model A / Model B classification (trained from scratch).

Design notes
------------
* Stride-1 stem: the discriminative evidence is pixel-level (high-frequency
  energy, periodic upsampling traces), so the first layers run at full input
  resolution. The first downsample happens only after the stem.
* Global average pooling: the network is fully convolutional up to the head,
  so it can be trained on crops and evaluated on the full 512x512 image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, Config


# Building blocks

def conv3x3(c_in: int, c_out: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)


class ConvBNReLU(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__(conv3x3(c_in, c_out, stride), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

# ResNet basic block
class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(c_in, c_out, stride)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = conv3x3(c_out, c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


# Network

class ForensicResNet(nn.Module):
# input (B, 3, H, W)
#     -> stem: 2x ConvBNReLU, s=1   full res        (stem_ch)
#     -> stage1 (s=2)               H/2             (widths[0])
#     -> stage2 (s=2)               H/4             (widths[1])
#     -> stage3 (s=2)               H/8             (widths[2])
#     -> stage4 (s=2)               H/16            (widths[3])
#     -> global avg pool -> dropout -> FC(num_classes)

    def __init__(self, widths=(32, 64, 128, 256), depths=(2, 2, 2, 2), stem_ch: int = 32,
                 dropout: float = 0.2, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(ConvBNReLU(3, stem_ch), ConvBNReLU(stem_ch, stem_ch))

        stages, c = [], stem_ch
        for w, d in zip(widths, depths):
            blocks = [BasicBlock(c, w, stride=2)] + [BasicBlock(w, w) for _ in range(d - 1)]
            stages.append(nn.Sequential(*blocks))
            c = w
        self.stages = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01); nn.init.zeros_(m.bias)
        # zero-init the last BN of each residual branch: every block starts as identity,
        # which noticeably stabilises training from scratch
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def features(self, x):
        return self.pool(self.stages(self.stem(x))).flatten(1)

    def forward(self, x):
        return self.fc(self.dropout(self.features(x)))


# Builder

def build_model() -> nn.Module:
    """The final model: stride-1-stem ResNet, widths 32-64-128-256, depths 2-2-2-2 (2.81M params)."""
    return ForensicResNet()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)