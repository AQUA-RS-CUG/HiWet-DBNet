from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """
    3x3 convolution with padding
    多大的dilation就多大的padding，保证输入和输出的特征大小相同
    return: nn.Conv2d
    """
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

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """
    1x1 convolution
    只管输入和输出的特征数
    return: nn.Conv2d
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def maxpool(kernel_size: int, stride: int, padding: int, dilation: int = 1):
    """
    MaxPool2d
    return: nn.MaxPool2d
    """
    return nn.MaxPool2d(
        kernel_size=kernel_size, 
        stride=stride,
        padding=padding,
        dilation=dilation
    )

# 常规残差模块
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
    ):
        """return: None"""
        super().__init__()
        if norm_layer is None: # 如果norm_layer为None，则norm_layer为nn.BatchNorm2d
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: # 保证卷积分组为1，和输入特征数为64
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1: # 保证不是空洞卷积
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        """return: Tensor"""
        identity = x # 恒等映射，短路连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入和输出维度不一样时，identity和out不能直接相加，需要进行downsample
        # 方法一：采用zero-padding增加维度，可以采用stride=2的pooling，保证不增加参数
        # 方法二：采用新的映射（projection shortcut），一般采用1×1的卷积，但是会增加新的参数
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 瓶颈残差模块
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

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
    ):
        """return: None"""
        super().__init__()
        if norm_layer is None: # 如果norm_layer为None，则norm_layer为nn.BatchNorm2d
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor):
        """reutrn: Tensor"""
        identity = x # 恒等映射，短路连接

        out = self.conv1(x) # 1×1 conv
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3×3 conv
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1×1 conv
        out = self.bn3(out)

        # 与BasicBlock相同
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
        num_class2, 
        num_class3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        """return: None"""
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None: # 如果未指定norm_layer，则norm_layer为nn.BatchNorm2d
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 # 输入特征数
        self.dilation = 1 # 扩展膨胀率大小
        if replace_stride_with_dilation is None: # 是否使用扩展卷积代替2×2步幅
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(16, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False) # 原始参数:kernel_size=7, stride=2, padding=3, bias=False 此处10可以改成12
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0) # 原始参数:kernel_size=3, stride=2, padding=1
        self.layer1 = self._make_layer(block, 64, layers[0]) # 原始参数:block, 64, layers[0]
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]) # 原始参数:block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        # 下面两句默认没有注释
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]) # 原始参数:block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]) # 原始参数:block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes) # 默认没有注释
        # self.fc3 = nn.Linear(256 * block.expansion, num_class3)
        # self.fc2 = nn.Linear(128 * block.expansion, num_class2)
        # 不要avgpool
        self.fc3 = nn.Linear(256 * 4 * 4 * block.expansion, num_class3)
        self.fc2 = nn.Linear(128 * 7 * 7 * block.expansion, num_class2)
        # 实验性添加
        # self.fc2 = nn.Linear(64 * 13 * 13 * block.expansion, num_class2)
        # self.fc3_2 = nn.Linear(256 * 4 * 4 * block.expansion, 4096)
        # self.fc2_2 = nn.Linear(128 * 7 * 7 * block.expansion, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    # 构建网络层序列
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ):
        """return: nn.Sequential"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion: # 如果步幅不为1或者输入的特征不等于输出的特征，定义降采样层
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     maxpool(kernel_size=2, stride=1, padding=0),
            #     norm_layer(planes * block.expansion),
            # )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
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
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor):
        """return: Tensor"""
        # # See note [TorchScript super()]
        # # print(x.shape)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x) # (64, 64, 13, 13)
        # # x_level1 = self.avgpool(x)
        # # x_level1_connect = torch.flatten(x_level1, 1)
        # # x_level1 = self.fc1(x_level1_connect)

        # x = self.layer2(x) # (64, 128, 7, 7)
        # x_level2 = self.avgpool(x)
        # x_level2_connect = torch.flatten(x_level2, 1)
        # x_level2 = self.fc2(x_level2_connect)

        # x = self.layer3(x) # (64, 256, 4, 4)
        # x_level3 = self.avgpool(x)
        # x_level3_connect = torch.flatten(x_level3, 1)
        # x_level3 = self.fc3(x_level3_connect)
        # # x = self.layer4(x)
        # # print(x.shape)
        
        # # x = self.fc(x) # 去掉注释

        # 不要avgpool
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # (64, 64, 13, 13)
        x = self.layer2(x) # (64, 128, 7, 7)
        x_level2_connect = torch.flatten(x, 1) # 6727
        x_level2 = self.fc2(x_level2_connect) # 4

        return x_level2 # resnet模型

    def forward(self, x: Tensor):
        """return: Tensor"""
        return self._forward_impl(x)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
):
    """return: ResNet"""
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #     model.load_state_dict(state_dict)
    return model

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    
    return: ResNet
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    return: ResNet
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    
    return: ResNet
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    return: ResNet
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    return: ResNet
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    
    return: ResNet
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    
    return: ResNet
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    return: ResNet
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr

    return: ResNet
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def resnet_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    return _resnet("resnet18", BasicBlock, [3, 3, 3], pretrained, progress, **kwargs)



if __name__ == "__main__":
    # model = resnet18(pretrained=False, progress=True, num_classes=7)
    model = resnet_backbone(pretrained=False, progress=True, num_class2=4, num_class3=6)
    # summary(model, input_size=[(3, 112, 112)], batch_size=32, device="cpu")
    summary(model, input_size=[(16, 15, 15)], batch_size=32, device="cpu")
    # print(out_level3.shape)
    # print(out_level3_connect.shape)