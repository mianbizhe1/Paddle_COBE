import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2D(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias_attr=bias)


class MeanShift(nn.Conv2D):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = paddle.to_tensor(rgb_std)
        self.weight.set_value(paddle.eye(3).reshape([3, 3, 1, 1]))
        self.weight.set_value(self.weight / std.reshape([3, 1, 1, 1]))
        self.bias.set_value(sign * rgb_range * paddle.to_tensor(rgb_mean))
        self.bias.set_value(self.bias / std)
        for param in self.parameters():
            param.stop_gradient = True


class BasicBlock(nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU()):

        m = [nn.Conv2D(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size // 2), stride=stride, bias_attr=bias)
        ]
        if bn: m.append(nn.BatchNorm2D(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x

        return res


class Upsampler(nn.Layer):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2D(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2D(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## add SELayer
class SELayer(nn.Layer):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sequential(
                nn.Conv2D(channel, channel // reduction, 1, padding=0, bias_attr=True),
                nn.ReLU(),
                nn.Conv2D(channel // reduction, channel, 1, padding=0, bias_attr=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## add SEResBlock
class SEResBlock(nn.Layer):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(), res_scale=1):

        super(SEResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2D(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SELayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res
