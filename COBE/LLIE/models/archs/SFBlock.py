import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SpaBlock(nn.Layer):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2D(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        return x + self.block(x)


class SpaBlock_res(nn.Layer):
    def __init__(self, nc):
        super(SpaBlock_res, self).__init__()
        self.conv = nn.Conv2D(nc, nc, 3, 1, 1)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self, x):
        res = self.conv(x)
        res = self.LeakyReLU(res)
        res = self.conv(res)
        x = x + res
        return x


class FreBlock(nn.Layer):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.fpre = nn.Conv2D(nc, nc, 1, 1, 0)
        self.process1 = nn.Sequential(
            nn.Conv2D(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 1, 1, 0))
        self.process2 = nn.Sequential(
            nn.Conv2D(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = paddle.fft.rfft2(self.fpre(x))
        mag = paddle.abs(x_freq)
        pha = paddle.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * paddle.cos(pha)
        imag = mag * paddle.sin(pha)
        x_out = paddle.complex(real, imag)
        x_out = paddle.fft.irfft2(x_out, s=(H, W))

        return x_out + x


class FreBlock_rcab(nn.Layer):
    def __init__(self, nc):
        super(FreBlock_rcab, self).__init__()
        self.fpre = nn.Conv2D(nc, nc, 1, 1, 0)
        self.conv = nn.Conv2D(nc, nc, 1, 1, 0)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.cal = CALayer(nc)

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = paddle.fft.rfft2(self.fpre(x))
        mag = paddle.abs(x_freq)
        pha = paddle.angle(x_freq)

        mag_res = self.conv(mag)
        mag_res = self.LeakyReLU(mag_res)
        mag_res = self.conv(mag_res)
        mag_res = self.cal(mag_res)
        mag = mag_res + mag

        pha_res = self.conv(pha)
        pha_res = self.LeakyReLU(pha_res)
        pha_res = self.conv(pha_res)
        pha_res = self.cal(pha_res)
        pha = pha_res + pha

        real = mag * paddle.cos(pha)
        imag = mag * paddle.sin(pha)
        x_out = paddle.complex(real, imag)
        x_out = paddle.fft.irfft2(x_out, s=(H, W))

        return x_out + x


class ProcessBlock(nn.Layer):
    def __init__(self, in_nc, spatial=True):
        super(ProcessBlock, self).__init__()
        self.spatial = spatial
        self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
        self.frequency_process = FreBlock(in_nc)
        self.cat = nn.Conv2D(2 * in_nc, in_nc, 1, 1, 0) if spatial else nn.Conv2D(in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        xori = x
        x_freq = self.frequency_process(x)
        x_spatial = self.spatial_process(x)
        xcat = paddle.concat([x_spatial, x_freq], axis=1)
        x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)

        return x_out + xori


class SFNet(nn.Layer):
    def __init__(self, nc, n=1):
        super(SFNet, self).__init__()
        self.conv1 = ProcessBlock(nc, spatial=False)
        self.conv2 = ProcessBlock(nc, spatial=False)
        self.conv3 = ProcessBlock(nc, spatial=False)
        self.conv4 = ProcessBlock(nc, spatial=False)
        self.conv5 = ProcessBlock(nc, spatial=False)

    def forward(self, x):
        x_ori = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        xout = x_ori + x5
        return xout


class AmplitudeNet_skip(nn.Layer):
    def __init__(self, nc=8, n=1):
        super(AmplitudeNet_skip, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2D(1, nc, 1, 1, 0),
            ProcessBlock(nc),
        )
        self.conv1 = ProcessBlock(nc)
        self.conv2 = ProcessBlock(nc)
        self.conv3 = ProcessBlock(nc)
        self.conv4 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2D(nc * 2, nc, 1, 1, 0),
        )
        self.conv5 = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2D(nc * 2, nc, 1, 1, 0),
        )
        self.convout = nn.Sequential(
            ProcessBlock(nc * 2),
            nn.Conv2D(nc * 2, 1, 1, 1, 0),
        )

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(paddle.concat([x2, x3], axis=1))
        x5 = self.conv5(paddle.concat([x1, x4], axis=1))
        xout = self.convout(paddle.concat([x, x5], axis=1))

        return xout


# 这里是修改后，增加RDB模块
class DenseLayer(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return paddle.concat([x, self.relu(self.conv(x))], axis=1)


class RDB(nn.Layer):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        self.lff = nn.Conv2D(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))


# Channel Attention Layer (CALayer)
class CALayer(nn.Layer):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sequential(
            nn.Conv2D(channel, channel // reduction, 1, padding=0),
            nn.ReLU(),
            nn.Conv2D(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
