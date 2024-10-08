"""
## ECCV 2022
"""

# --- Imports --- #
# --- Imports --- #
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class UNetConvBlock(nn.Layer):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2D(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2D(in_size, out_size, kernel_size=3, padding=1, bias_attr=True)
        self.relu_1 = nn.LeakyReLU(relu_slope)
        self.conv_2 = nn.Conv2D(out_size, out_size, kernel_size=3, padding=1, bias_attr=True)
        self.relu_2 = nn.LeakyReLU(relu_slope)

        if use_HIN:
            self.norm = nn.InstanceNorm2D(out_size // 2, weight_attr=True, bias_attr=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = paddle.chunk(out, 2, axis=1)
            out = paddle.concat([self.norm(out_1), out_2], axis=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class InvBlock(nn.Layer):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock(self.split_len2, self.split_len1)
        self.G = UNetConvBlock(self.split_len1, self.split_len2)
        self.H = UNetConvBlock(self.split_len1, self.split_len2)

    def forward(self, x):
        x1, x2 = paddle.split(x, [self.split_len1, self.split_len2], axis=1)

        y1 = x1 + self.F(x2)
        self.s = self.clamp * (paddle.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2 * paddle.exp(self.s) + self.G(y1)

        out = paddle.concat((y1, y2), axis=1)
        return out


class SpaBlock(nn.Layer):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc, nc // 2)

    def forward(self, x):
        yy = self.block(x)
        return x + yy


class FreBlock(nn.Layer):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2D(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 1, 1, 0)
        )
        self.processpha = nn.Sequential(
            nn.Conv2D(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 1, 1, 0)
        )

    def forward(self, x):
        mag = paddle.abs(x)
        pha = paddle.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * paddle.cos(pha)
        imag = mag * paddle.sin(pha)
        x_out = paddle.complex(real, imag)
        return x_out


class FreBlockAdjust(nn.Layer):
    def __init__(self, nc):
        super(FreBlockAdjust, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2D(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 1, 1, 0)
        )
        self.processpha = nn.Sequential(
            nn.Conv2D(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 1, 1, 0)
        )
        self.sft = SFT(nc)
        self.cat = nn.Conv2D(2 * nc, nc, 1, 1, 0)

    def forward(self, x, y_amp, y_phase):
        mag = paddle.abs(x)
        pha = paddle.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        mag = self.sft(mag, y_amp)
        pha = self.cat(paddle.concat([y_phase, pha], axis=1))
        real = mag * paddle.cos(pha)
        imag = mag * paddle.sin(pha)
        x_out = paddle.complex(real, imag)
        return x_out


class ProcessBlock(nn.Layer):
    def __init__(self, in_nc):
        super(ProcessBlock, self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlock(in_nc)
        self.frequency_spatial = nn.Conv2D(in_nc, in_nc, 3, 1, 1)
        self.spatial_frequency = nn.Conv2D(in_nc, in_nc, 3, 1, 1)
        self.cat = nn.Conv2D(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x):
        x_ori = x
        _, _, H, W = x.shape
        x_freq = paddle.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = paddle.fft.irfft2(x_freq, s=(H, W), norm='backward')

        x_cat = paddle.concat([x, x_freq_spatial], axis=1)
        x_out = self.cat(x_cat)
        return x_out + x_ori


class ProcessBlockAdjust(nn.Layer):
    def __init__(self, in_nc):
        super(ProcessBlockAdjust, self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlockAdjust(in_nc)
        self.cat = nn.Conv2D(2 * in_nc, in_nc, 1, 1, 0)

    def forward(self, x, y_amp, y_phase):
        x_ori = x
        _, _, H, W = x.shape
        x_freq = paddle.fft.rfft2(x, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq, y_amp, y_phase)
        x_freq_spatial = paddle.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_cat = paddle.concat([x, x_freq_spatial], axis=1)
        x_out = self.cat(x_cat)
        return x_out + x_ori


class SFT(nn.Layer):
    def __init__(self, nc):
        super(SFT, self).__init__()
        self.convmul = nn.Conv2D(nc, nc, 3, 1, 1)
        self.convadd = nn.Conv2D(nc, nc, 3, 1, 1)
        self.convfuse = nn.Conv2D(2 * nc, nc, 1, 1, 0)

    def forward(self, x, res):
        mul = self.convmul(res)
        add = self.convadd(res)
        fuse = self.convfuse(paddle.concat([x, mul * x + add], axis=1))
        return fuse


def coeff_apply(InputTensor, CoeffTensor, isoffset=True):
    if not isoffset:
        raise ValueError("No-offset is not implemented.")
    bIn, cIn, hIn, wIn = InputTensor.shape
    bCo, cCo, hCo, wCo = CoeffTensor.shape
    assert hIn == hCo and wIn == wCo, f'Wrong dimension: In:{hIn}x{wIn} != Co:{hCo}x{wCo}'
    if isoffset:
        assert cCo % (cIn + 1) == 0, 'The dimension of Coeff and Input is mismatching with offset.'
        cOut = cCo // (cIn + 1)
    else:
        assert cCo % cIn == 0, 'The dimension of Coeff and Input is mismatching without offset.'
        cOut = cCo // cIn
    outList = []

    if isoffset:
        for i in range(int(cOut)):
            Oc = CoeffTensor[:, cIn + (cIn + 1) * i:cIn + (cIn + 1) * i + 1, :, :]
            Oc = Oc + paddle.sum(CoeffTensor[:, (cIn + 1) * i:(cIn + 1) * i + cIn, :, :] * InputTensor,
                                 axis=1, keepdim=True)
            outList.append(Oc)

    return paddle.concat(outList, axis=1)


class HighNet(nn.Layer):
    def __init__(self, nc):
        super(HighNet, self).__init__()
        self.conv0 = nn.PixelUnshuffle(2)
        self.conv1 = ProcessBlockAdjust(12)
        # self.conv2 = ProcessBlockAdjust(nc)
        self.conv3 = ProcessBlock(12)
        self.conv4 = ProcessBlock(12)
        self.conv5 = nn.PixelShuffle(2)
        self.convout = nn.Conv2D(3, 3, 3, 1, 1)
        self.trans = nn.Conv2D(6, 32, 1, 1, 0)
        self.con_temp1 = nn.Conv2D(32, 32, 3, 1, 1)
        self.con_temp2 = nn.Conv2D(32, 32, 3, 1, 1)
        self.con_temp3 = nn.Conv2D(32, 3, 3, 1, 1)
        self.LeakyReLU = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x, y_down, y_down_amp, y_down_phase):
        x_ori = x
        x = self.conv0(x)  # 3*4=12

        x1 = self.conv1(x, y_down_amp, y_down_phase)
        # x2 = self.conv2(x1, y_down_amp, y_down_phase)

        x3 = self.conv3(x1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        xout_temp = self.convout(x5)
        y_aff = self.trans(paddle.concat([F.interpolate(y_down, scale_factor=2, mode='bilinear'), xout_temp], axis=1))
        con_temp1 = self.con_temp1(y_aff)
        con_temp2 = self.con_temp2(con_temp1)
        xout = self.con_temp3(con_temp2)
        # xout = coeff_apply(x_ori, y_aff) + xout

        return xout


class LowNet(nn.Layer):
    def __init__(self, in_nc, nc):
        super(LowNet, self).__init__()
        self.conv0 = nn.Conv2D(in_nc, nc, 1, 1, 0)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2D(nc, nc * 2, stride=2, kernel_size=2, padding=0)
        self.conv2 = ProcessBlock(nc * 2)
        self.downsample2 = nn.Conv2D(nc * 2, nc * 3, stride=2, kernel_size=2, padding=0)
        self.conv3 = ProcessBlock(nc * 3)
        self.up1 = nn.ConvTranspose2D(nc * 5, nc * 2, 1, 1)
        self.conv4 = ProcessBlock(nc * 2)
        self.up2 = nn.ConvTranspose2D(nc * 3, nc * 1, 1, 1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2D(nc, 12, 1, 1, 0)
        self.convoutfinal = nn.Conv2D(12, 3, 1, 1, 0)

        self.transamp = nn.Conv2D(12, 12, 1, 1, 0)
        self.transpha = nn.Conv2D(12, 12, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x01 = self.conv1(x)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2)
        x34 = self.up1(paddle.concat([F.interpolate(x3, size=(x12.shape[2], x12.shape[3]), mode='bilinear'), x12], axis=1))
        x4 = self.conv4(x34)
        x4 = self.up2(paddle.concat([F.interpolate(x4, size=(x01.shape[2], x01.shape[3]), mode='bilinear'), x01], axis=1))
        x5 = self.conv5(x4)
        xout = self.convout(x5)
        xout_fre = paddle.fft.rfft2(xout, norm='backward')
        xout_fre_amp, xout_fre_phase = paddle.abs(xout_fre), paddle.angle(xout_fre)
        xfinal = self.convoutfinal(xout)

        return xfinal, self.transamp(xout_fre_amp), self.transpha(xout_fre_phase)


class InteractNet(nn.Layer):
    def __init__(self, nc=16):
        super(InteractNet, self).__init__()
        self.extract = nn.Conv2D(3, nc // 2, 1, 1, 0)
        self.lownet = LowNet(nc // 2, nc * 12)
        self.highnet = HighNet(nc)

    def forward(self, x):
        x_f = self.extract(x)
        x_f_down = F.interpolate(x_f, scale_factor=0.5, mode='bilinear')
        y_down, y_down_amp, y_down_phase = self.lownet(x_f_down)
        y = self.highnet(x, y_down, y_down_amp, y_down_phase)

        return y, y_down