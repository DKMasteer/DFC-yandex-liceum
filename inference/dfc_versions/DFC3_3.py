import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out


class MultiSRMLayer(nn.Module):
    def __init__(self, clamp_value=3.0):
        super().__init__()

        kernels = torch.tensor([
            [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 1, -2, 1, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],

            [[0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0],
             [0, 2, -4, 2, 0],
             [0, -1, 2, -1, 0],
             [0, 0, 0, 0, 0]],

            [[-1, 2, -2, 2, -1],
             [2, -6, 8, -6, 2],
             [-2, 8, -12, 8, -2],
             [2, -6, 8, -6, 2],
             [-1, 2, -2, 2, -1]]
        ], dtype=torch.float32)

        kernels[0] /= 2.0
        kernels[1] /= 4.0
        kernels[2] /= 12.0

        weight = kernels.unsqueeze(1).repeat(3, 1, 1, 1)   # 9, 1, 5, 5
        self.register_buffer("weight", weight)

        self.clamp = nn.Hardtanh(min_val=-clamp_value, max_val=clamp_value)

    def forward(self, x):
        x = F.conv2d(x, self.weight, bias=None, stride=1, padding=2, groups=3)
        x = self.clamp(x)
        return x


def make_stage(in_channels: int, out_channels: int, block_count: int, stride: int):
    layers = [SEResidualBlock(in_channels, out_channels, stride=stride)]
    for _ in range(block_count - 1):
        layers.append(SEResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class DFC_3_3(nn.Module):
    def __init__(self, num_classes: int = 1, dropout_p: float = 0.2):
        super().__init__()

        self.rgb_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.rgb_layer1 = make_stage(64, 64, block_count=3, stride=1)
        self.rgb_layer2 = make_stage(64, 128, block_count=4, stride=2)
        self.rgb_layer3 = make_stage(128, 256, block_count=6, stride=2)
        self.rgb_layer4 = make_stage(256, 512, block_count=3, stride=2)

        self.srm = MultiSRMLayer()

        self.hf_stem = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # HF-ветку делаем легче, чтобы не убить время и VRAM
        self.hf_layer1 = make_stage(32, 32, block_count=1, stride=1)
        self.hf_layer2 = make_stage(32, 64, block_count=2, stride=2)
        self.hf_layer3 = make_stage(64, 128, block_count=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RGB branch
        rgb = self.rgb_stem(x)
        rgb = self.rgb_layer1(rgb)
        rgb = self.rgb_layer2(rgb)
        rgb = self.rgb_layer3(rgb)
        rgb = self.rgb_layer4(rgb)
        rgb = self.avgpool(rgb)
        rgb = torch.flatten(rgb, 1)

        # HF branch
        hf = self.srm(x)
        hf = self.hf_stem(hf)
        hf = self.hf_layer1(hf)
        hf = self.hf_layer2(hf)
        hf = self.hf_layer3(hf)
        hf = self.avgpool(hf)
        hf = torch.flatten(hf, 1)

        # Fusion
        feats = torch.cat([rgb, hf], dim=1)
        out = self.head(feats)
        return out