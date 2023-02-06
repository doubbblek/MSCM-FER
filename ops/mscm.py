from einops import rearrange
from functools import partial
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import numpy as np
import torch.nn.functional as F 

from torchvision.transforms._presets import ImageClassification
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param, _ModelURLs
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._api import WeightsEnum, Weights


class Correlation(nn.Module):
    def __init__(self, max_displacement=4):
        super().__init__()
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad2d(max_displacement, 0)

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                           torch.arange(0, 2 * self.max_hdisp + 1)])
        hei, wid = in1.size(2), in1.size(3)
        output = torch.cat([
            torch.sum(in1 * in2_pad[:, :, dy:dy+hei, dx:dx+wid], dim=1, keepdim=True) \
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))], 1)
        return output


class CorrelationModule(nn.Module):
    """generate motion vector based on two given frames"""
    def __init__(self, corr_range, n_segment):
        super().__init__()
        print('Add corr')
        self.n_segment = n_segment
        self.corr_range = corr_range
        self.correlation = Correlation(corr_range)
        self._construct_motion_grid()
    
    def _construct_motion_grid(self):
        self.h_grid = np.arange(-self.corr_range, self.corr_range+1)
        self.v_grid = np.arange(-self.corr_range, self.corr_range+1)
        self.h_grid_t = torch.from_numpy(self.h_grid).float()
        self.v_grid_t = torch.from_numpy(self.v_grid).float()
    
    def _gau_kernel(self, x, sigma=0.1):
        b, _, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h*w, (self.corr_range*2+1), (self.corr_range*2+1)) # b, hw, r, r
        _, idx = x.view(b, h*w, -1).max(dim=-1) # b, n
        h_idx = (idx // (self.corr_range*2+1) - self.corr_range) \
                .view(b, h*w, 1, 1).repeat(1, 1, (self.corr_range*2+1), (self.corr_range*2+1)) # b, n, r_h, r_w
        v_idx = (idx % (self.corr_range*2+1) - self.corr_range) \
                .view(b, h*w, 1, 1).repeat(1, 1, (self.corr_range*2+1), (self.corr_range*2+1))

        x_cord = self.h_grid_t.to(x.device) \
                 .view(1, 1, self.corr_range*2+1, 1).repeat(b, h*w, 1, (self.corr_range*2+1))
        y_cord = self.v_grid_t.to(x.device) \
                 .view(1, 1, 1, self.corr_range*2+1).repeat(b, h*w, (self.corr_range*2+1), 1)
        gauss_kernel = torch.exp(-((x_cord-h_idx)**2 + (y_cord-v_idx)**2) / (2 * sigma**2))
        return gauss_kernel * x
        # motion = torch.cat([h_idx, v_idx], dim=-1).permute(0, 2, 1).view(b, 2, h, w)
        # return motion
    
    def _softmax2d(self, x, beta=10):
        b, n, rh, rw = x.size()
        x_flat = x.view(b, n, -1) * beta
        _sum = x_flat.sum(dim=-1, keepdim=True) + 1e-12
        x_flat = (x_flat / _sum).view(b, n, rh, rw)

        sum_h = x_flat.sum(dim=3) # b, n, rh
        sum_w = x_flat.sum(dim=2) # b, n, rw
        h_motion = self.h_grid_t.to(x.device).view(1, 1, self.corr_range*2+1).repeat(b, n, 1)
        v_motion = self.v_grid_t.to(x.device).view(1, 1, self.corr_range*2+1).repeat(b, n, 1)
        h_cord = (sum_h * h_motion).sum(dim=-1, keepdim=True)
        v_cord = (sum_w * v_motion).sum(dim=-1, keepdim=True)
        return torch.cat([h_cord, v_cord], dim=-1)
    
    def _get_motion(self, x1, x2):
        _, _, h, w = x1.size()
        r = self.correlation(x1, x2) # b, r*r, h, w
        conf, _ = r.max(dim=1, keepdim=True)
        r = self._gau_kernel(r) # b, hw, r, r
        # r = rearrange(r, 'b (x y) h w -> b (h w) x y', x=self.corr_range*2+1, y=self.corr_range*2+1)
        r = self._softmax2d(r).permute(0, 2, 1).view(-1, 2, h, w) # b, n, 2
        r = torch.cat([r, conf], dim=1)
        return r 

    def forward(self, x):
        bt, c, h, w = x.size()
        x_m = F.normalize(x.detach(), p=2, dim=1)
        x_m_post = torch.zeros_like(x_m)
        x_m = x_m.view(-1, self.n_segment, c, h, w)
        x_m_post = x_m_post.view(-1, self.n_segment, c, h, w)
        x_m_post[:, 1:] = x_m[:, :-1]
        # x_m_post = x_m[:, :-1]
        # x_m_post = torch.cat([x_m[:, :1], x_m_post], dim=1)
        x_m = x_m.view(bt, c, h, w)
        x_m_post = x_m_post.view(bt, c, h, w)
        
        r = self._get_motion(x_m, x_m_post)
        return r


class TSM(nn.Module):
    def __init__(self, n_segment=8, n_div=8):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=8):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment 
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class SingleMotionEncoder(nn.Module):
    def __init__(self, planes_1, planes_2, stride=[2, 2]):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, planes_1, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes_1), nn.ReLU(True),
            nn.Conv2d(planes_1, planes_1, 3, stride[0], 1, bias=False, groups=planes_1),
            nn.BatchNorm2d(planes_1), nn.ReLU(True),
            nn.Conv2d(planes_1, planes_2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(planes_2), nn.ReLU(True),
            nn.Conv2d(planes_2, planes_2, 3, stride[1], 1, bias=False, groups=planes_2),
            nn.BatchNorm2d(planes_2), nn.ReLU(True),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x 


class AdaFusionTwoMotion(nn.Module):
    def __init__(self, planes_1, planes_2):
        super().__init__()
        # print('scadafuse')
        # self.planes_2 = planes_2
        # self.linear = nn.Sequential(
        #     nn.Conv1d(planes_2, planes_2//8, 3, 1, 1, bias=False),
        #     nn.BatchNorm1d(planes_2//8), nn.SiLU(True),
        #     nn.Conv1d(planes_2//8, 2*planes_2, 1, bias=False),)
        
        # self.conv = nn.Sequential(
        #     nn.Conv3d(planes_2, planes_2//16, 3, 1, 1, bias=False),
        #     nn.BatchNorm3d(planes_2//16), nn.SiLU(True),
        #     nn.Conv3d(planes_2//16, 2, 1, bias=False),)

        self.mo_endcoder_1 = SingleMotionEncoder(planes_1, planes_2, [1, 2])
        self.mo_endcoder_2 = SingleMotionEncoder(planes_1, planes_2, [1, 1])
    
    def forward(self, m1, m2, x):
        m1 = self.mo_endcoder_1(m1)
        m2 = self.mo_endcoder_2(m2)

        # tmp = m1 + m2 + x 
        # tmp = rearrange(tmp, '(b t) c h w -> b c t h w', t=16)
        # att = torch.mean(tmp, dim=[3, 4]) # b, c, t
        # att = self.linear(att) # b, 2c, t
        # att = rearrange(att, 'b (n c) t -> (b t) n c', n=2).unsqueeze(-1).unsqueeze(-1) # bt, 2, c, 1, 1
        
        # satt = self.conv(tmp) # b, 2, t, h, w
        # satt = rearrange(satt, 'b c t h w -> (b t) c h w').unsqueeze(2) # bt, 2, 1, h, w
        
        # all_att = torch.sigmoid(satt + att) # bt, 2, c, h, w

        # return all_att[:, 0] * m1 + all_att[:, 1] * m2 + x 
        return m1 + m2 + x 


class SAdaFusion(nn.Module):
    def __init__(self, planes):
        super().__init__()
        print('SADA')
        self.mo_endcoder_1 = SingleMotionEncoder(planes//2, planes, [1, 2])
        self.mo_endcoder_2 = SingleMotionEncoder(planes//2, planes, [1, 1])
        self.conv = nn.Sequential(
            nn.Conv3d(planes, planes//16, 3, 1, 1, bias=False),
            nn.BatchNorm3d(planes//16), nn.SiLU(True),
            nn.Conv3d(planes//16, 2, 1, bias=False),)
    
    def forward(self, m1, m2, x):
        m1 = self.mo_endcoder_1(m1)
        m2 = self.mo_endcoder_2(m2)

        tmp = m1 + m2 + x 
        tmp = rearrange(tmp, '(b t) c h w -> b c t h w', t=16)
        att = self.conv(tmp) # b, 2, t, h, w
        att = rearrange(att, 'b n t h w -> (b t) n h w')
        # att = F.softmax(att, dim=1)
        att = torch.sigmoid(att)
        return m1 * att[:, :1] + m2 * att[:, 1:] + x 


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_frames: int = 16,
        ts = True
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        if ts == True:
            self.tsm = TSM(n_segment=n_frames, n_div=8)
        else:
            self.tsm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.tsm(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_frames: int = 16
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        n_frames: int = 16
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], n_frames=n_frames, ts=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], n_frames=n_frames, ts=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], n_frames=n_frames, ts=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], n_frames=n_frames, ts=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.last_dim = 512 * block.expansion
        # self.corr = CorrelationModule(2, n_frames)
        # self.fuse = AdaFusionTwoMotion(32, 64)
        # self.fuse = SAdaFusion(64)
        # self.m_encoder1 = SingleMotionEncoder(64, 64, [1, 2])
        # self.m_encoder2 = SingleMotionEncoder(64, 64, [1, 1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        n_frames: int = 16,
        ts = True
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, n_frames, ts
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    n_frames=n_frames,
                    ts=ts
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # m1 = self.corr(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # m2 = self.corr(x)
        # x = self.fuse(m1, m2, x)
        # x = x + self.m_encoder1(m1) + self.m_encoder2(m2)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        b = x.size(0)
        x = rearrange(x, 'b (t c) h w -> (b t) c h w', c=3)
        x = self._forward_impl(x)
        x = rearrange(x, '(b t) c -> b t c', b=b)
        x = x.mean(dim=1)
        return x
    
    def reset_fc(self, num_classes):
        """Run after pretrained loaded"""
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.last_dim, num_classes)
        )


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress), strict=False)

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt50_32X4D_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25028904,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


class Wide_ResNet50_2_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 68883240,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:

    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:

    weights = ResNet50_Weights.verify(weights)

    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", ResNeXt50_32X4D_Weights.IMAGENET1K_V1))
def resnext50_32x4d(
    *, weights: Optional[ResNeXt50_32X4D_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:

    weights = ResNeXt50_32X4D_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


@handle_legacy_interface(weights=("pretrained", Wide_ResNet50_2_Weights.IMAGENET1K_V1))
def wide_resnet50_2(
    *, weights: Optional[Wide_ResNet50_2_Weights] = None, progress: bool = True, **kwargs: Any
) -> ResNet:

    weights = Wide_ResNet50_2_Weights.verify(weights)

    _ovewrite_named_param(kwargs, "width_per_group", 64 * 2)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)


model_urls = _ModelURLs(
    {
        "resnet18": ResNet18_Weights.IMAGENET1K_V1.url,
        "resnet50": ResNet50_Weights.IMAGENET1K_V1.url,
        "resnext50_32x4d": ResNeXt50_32X4D_Weights.IMAGENET1K_V1.url,
        "wide_resnet50_2": Wide_ResNet50_2_Weights.IMAGENET1K_V1.url,
    }
)


def ts_resnet50():
    return resnet50(weights=ResNet50_Weights.DEFAULT, n_frames=16)


if __name__ == '__main__':
    # net = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT, n_frames=8)
    net = resnet18(weights=None, n_frames=16)
    net.reset_fc(99)
    # out = net(torch.rand(1, 24, 224, 224))
    # import time

    # for i in range(20):
    #     t1 = time.time()
    #     net(torch.randn(1, 48, 112, 112))
    #     t2 = time.time()
    #     print(t2 - t1)

    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 48, 112, 112)
    macs, params = profile(net, inputs=(input,))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)
    