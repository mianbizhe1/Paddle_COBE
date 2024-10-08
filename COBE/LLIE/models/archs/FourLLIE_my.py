import functools
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# import kornia  # Kornia 在Paddle中没有直接的等价物，可以考虑自定义实现或找等价操作
import COBE.LLIE.models.archs.arch_util as arch_util
from COBE.LLIE.models.archs.SFBlock import *
from COBE.LLIE.models.archs.FSIB import FuseBlock
from COBE.LLIE.models.archs.myblock import FSAIO

from COBE.LLIE.models.archs.SFBlock import AmplitudeNet_skip, SFNet


###############################

class FourLLIE(nn.Layer):
    def __init__(self, nf=64):
        super(FourLLIE, self).__init__()

        # AMPLITUDE ENHANCEMENT
        self.AmpNet = nn.Sequential(
            AmplitudeNet_skip(8),
            nn.Sigmoid()
        )

        self.nf = nf
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)

        self.conv_first_1 = nn.Conv2D(3 * 2, nf, 3, 1, 1, bias_attr=True)
        self.conv_first_2 = nn.Conv2D(nf, nf, 3, 2, 1, bias_attr=True)
        self.conv_first_3 = nn.Conv2D(nf, nf, 3, 2, 1, bias_attr=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)

        self.upconv1 = nn.Conv2D(nf*2, nf * 4, 3, 1, 1, bias_attr=True)
        self.upconv2 = nn.Conv2D(nf*2, nf * 4, 3, 1, 1, bias_attr=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2D(nf*2, nf, 3, 1, 1, bias_attr=True)
        self.conv_last = nn.Conv2D(nf, 3, 3, 1, 1, bias_attr=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.transformer = SFNet(nf)
        self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)
        self.fuseblock = FuseBlock(64)
        self.fourget = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.fsaio = FSAIO(nc=nf)

    def get_mask(self, dark):
        light = paddle.vision.transforms.functional.gaussian_blur(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = paddle.abs(dark - light)

        mask = paddle.divide(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = paddle.max(mask.reshape([batch_size, -1]), axis=1).unsqueeze(-1).unsqueeze(-1)
        mask_max = mask_max.tile([1, 1, height, width])
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = paddle.clip(mask, min=0, max=1.0)
        return mask.astype(paddle.float32)

    def forward(self, x):
        # AMPLITUDE ENHANCEMENT
        _, _, H, W = x.shape
        image_fft = paddle.fft.fft2(x, norm='backward')
        mag_image = paddle.abs(image_fft)
        pha_image = paddle.angle(image_fft)
        curve_amps = self.AmpNet(x)
        mag_image = mag_image / (curve_amps + 0.00000001)  # * d4
        real_image_enhanced = mag_image * paddle.cos(pha_image)
        imag_image_enhanced = mag_image * paddle.sin(pha_image)
        img_amp_enhanced = paddle.fft.ifft2(paddle.complex(real_image_enhanced, imag_image_enhanced), s=(H, W), norm='backward').real()

        x_center = img_amp_enhanced

        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, [0, pad_w, 0, pad_h], mode="reflect")
            x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        L1_fea_1 = self.lrelu(self.conv_first_1(paddle.concat((x_center,x), axis=1)))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)

        fea_fre, fea_spa = self.fsaio(fre=fea, spa=fea)
        for _ in range(6):  # 重复调用fsaio
            fea_fre, fea_spa = self.fsaio(fre=fea_fre, spa=fea_spa)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = self.get_mask(x_center)
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        channel = fea.shape[1]
        mask = mask.tile([1, channel, 1, 1])

        fea = fea_fre * (1 - mask) + fea_spa * mask

        out_noise = self.recon_trunk(fea)
        out_noise = paddle.concat([out_noise, L1_fea_3], axis=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = paddle.concat([out_noise, L1_fea_2], axis=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = paddle.concat([out_noise, L1_fea_1], axis=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :H, :W]

        return out_noise, mag_image, x_center, mask


if __name__ == '__main__':
    a = paddle.randn([1, 3, 400, 600])
    model = FourLLIE()
    out_noise, mag_image, x_center, mask = model(a)
    print(mag_image.shape)
