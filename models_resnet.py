import torch
import torch.nn as nn
from torchvision.models import resnet


def replace_first_conv(model, in_c):
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(in_c, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                            stride=old_conv.stride, padding=old_conv.padding, bias=False)
    model.conv1.weight.data[:, :3, :, :] = old_conv.weight.data
    return model


@torch.jit.ignore
def no_weight_decay(self):
    return {}


def resnet18(in_c, num_classes=62, pretrained=True, **kwargs):
    model = resnet.resnet18(pretrained=pretrained, **kwargs)
    model = replace_first_conv(model, in_c)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes, bias=True)
    model.no_weight_decay = no_weight_decay.__get__(model)
    return model


def resnet34(in_c, num_classes=62, pretrained=True, **kwargs):
    model = resnet.resnet34(pretrained=pretrained, **kwargs)
    model = replace_first_conv(model, in_c)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes, bias=True)
    model.no_weight_decay = no_weight_decay.__get__(model)
    return model


def resnet50(in_c, num_classes=62, pretrained=True, **kwargs):
    model = resnet.resnet50(pretrained=pretrained, **kwargs)
    model = replace_first_conv(model, in_c)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes, bias=True)
    model.no_weight_decay = no_weight_decay.__get__(model)
    return model


def resnet101(in_c, num_classes=62, pretrained=True, **kwargs):
    model = resnet.resnet101(pretrained=pretrained, **kwargs)
    model = replace_first_conv(model, in_c)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes, bias=True)
    model.no_weight_decay = no_weight_decay.__get__(model)
    return model


def resnet152(in_c, num_classes=62, pretrained=True, **kwargs):
    model = resnet.resnet152(pretrained=pretrained, **kwargs)
    model = replace_first_conv(model, in_c)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes, bias=True)
    model.no_weight_decay = no_weight_decay.__get__(model)
    return model
