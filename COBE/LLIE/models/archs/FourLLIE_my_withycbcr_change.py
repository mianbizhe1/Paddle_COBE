import functools
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.transforms import functional as Fv
from LLIE.models.archs.arch_util import ResidualBlock_noBN, make_layer
from LLIE.models.archs.SFBlock import AmplitudeNet_skip, SFNet
from LLIE.models.archs.FSIB import FuseBlock
from LLIE.models.archs.myblock import FSAIO
from LLIE.models.archs.torch_rgbto import rgb_to_ycbcr, ycbcr_to_rgb

class FourLLIE_my(nn.Layer):
    def __init__(self, nf=64):
        super(FourLLIE_my, self).__init__()

        # AMPLITUDE ENHANCEMENT
        self.AmpNet = nn.Sequential(
            AmplitudeNet_skip(8),
            nn.Sigmoid()
        )

        self.nf = nf
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)

        self.conv_first_1 = nn.Conv2D(3 * 2, nf, 3, stride=1, padding=1)
        self.conv_first_2 = nn.Conv2D(nf, nf, 3, stride=2, padding=1 )
        self.conv_first_3 = nn.Conv2D(nf, nf, 3, stride=2, padding=1)

        self.feature_extraction = make_layer(ResidualBlock_noBN_f, 1)
        self.recon_trunk = make_layer(ResidualBlock_noBN_f, 1)

        self.upconv1 = nn.Conv2D(nf*2, nf * 4, 3, stride=1, padding=1)
        self.upconv2 = nn.Conv2D(nf*2, nf * 4, 3, stride=1, padding=1)
        self.myupconv = nn.Conv2D(nf*2, nf*2, 3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2D(nf*2, nf, 3, stride=1, padding=1)
        self.conv_last = nn.Conv2D(nf, 3, 3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.transformer = SFNet(nf)
        self.recon_trunk_light = make_layer(ResidualBlock_noBN_f, 6)
        self.fuseblock = FuseBlock(64)
        self.fourget = make_layer(ResidualBlock_noBN_f, 1)
        self.fsaio1 = FSAIO(nc=nf)
        self.fsaio2 = FSAIO(nc=nf)
        self.fsaio3 = FSAIO(nc=nf)
        self.fsaio4 = FSAIO(nc=nf)
        self.fsaio5 = FSAIO(nc=nf)
        self.fsaio6 = FSAIO(nc=nf)
        self.fsaio7 = FSAIO(nc=nf)

    def gaussian_blur_2d(self, input, kernel_size, sigma):
        # Ensure input is a paddle.Tensor
        if not isinstance(input, paddle.Tensor):
            raise TypeError(f"Expected input to be a paddle.Tensor, got: {type(input)}")

        def get_gaussian_kernel(kernel_size, sigma):
            ax = paddle.arange(kernel_size, dtype=paddle.float32) - (kernel_size - 1) / 2.0
            xx, yy = paddle.meshgrid(ax, ax)
            kernel = paddle.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
            kernel /= kernel.sum()  # Normalize
            return kernel

        channels = input.shape[1]
        if isinstance(kernel_size, (tuple, list)):
            kernel_size = kernel_size[0]  # Assuming square kernel
        if isinstance(sigma, (tuple, list)):
            sigma = sigma[0]  # Assuming same sigma for all channels

        gaussian_kernel = get_gaussian_kernel(kernel_size, sigma)

        # Expand kernel to match the input channels
        gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.tile([channels, 1, 1, 1])  # Shape: (channels, 1, kernel_size, kernel_size)

        # Perform convolution using paddle.nn.functional
        output = F.conv2d(input, gaussian_kernel, padding=kernel_size // 2, groups=channels)

        return output

    def get_mask(self, dark):
        # 使用 paddle 实现高斯模糊替代 Kornia 的实现
        light = self.gaussian_blur_2d(dark, (5, 5), (1.5, 1.5))
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = paddle.abs(dark - light)

        mask = paddle.divide(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = paddle.max(mask.reshape((batch_size, -1)), axis=1)
        mask_max = mask_max.reshape((batch_size, 1, 1, 1))
        mask_max = mask_max.tile([1, 1, height, width])
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = paddle.clip(mask, min=0, max=1.0)
        return mask.astype(paddle.float32)

    def forward(self, x):
        # AMPLITUDE ENHANCEMENT
        _, _, H, W = x.shape
        # transform here
        Y, Cb, Cr = rgb_to_ycbcr(x)
        # enhance Y
        image_fft = paddle.fft.fft2(Y, norm='backward')
        mag_image = paddle.abs(image_fft)
        pha_image = paddle.angle(image_fft)
        curve_amps = self.AmpNet(Y)
        mag_image = mag_image / (curve_amps + 1e-8)  # * d4
        real_image_enhanced = mag_image * paddle.cos(pha_image)
        imag_image_enhanced = mag_image * paddle.sin(pha_image)
        img_amp_enhanced = paddle.fft.ifft2(paddle.complex(real_image_enhanced, imag_image_enhanced), s=(H, W), norm='backward').real()
        # enhance Cb
        enhenced_Cb = Cb
        # enhance Cr
        enhenced_Cr = Cr
        enhenced_all = ycbcr_to_rgb(img_amp_enhanced, enhenced_Cb, enhenced_Cr)

        x_center = enhenced_all

        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            x_center = F.pad(x_center, (0, pad_w, 0, pad_h), mode='reflect')
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        L1_fea_1 = self.lrelu(self.conv_first_1(paddle.concat((x_center, x), axis=1)))
        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1))
        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2))

        fea = self.feature_extraction(L1_fea_3)

        fea_fre, fea_spa = self.fsaio1(fea, fea)
        fea_fre, fea_spa = self.fsaio2(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio3(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio4(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio5(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio6(fea_fre, fea_spa)
        fea_fre, fea_spa = self.fsaio7(fea_fre, fea_spa)

        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = self.get_mask(x_center)
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest')

        channel = fea.shape[1]
        mask = mask.tile([1, channel, 1, 1])

        fea = fea_spa

        out_noise = self.recon_trunk(fea)  # nf
        out_noise = paddle.concat([out_noise, L1_fea_3], axis=1)  # 2*nf
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))  # upconv1->4*nf,pixel_shuffle->nf
        out_noise = paddle.concat([out_noise, L1_fea_2], axis=1)  # 2*nf
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))  # upconv1->4*nf,pixel_shuffle->nf
        out_noise = paddle.concat([out_noise, L1_fea_1], axis=1)  # 2*nf
        out_noise = self.lrelu(self.HRconv(out_noise))  # nf
        out_noise = self.conv_last(out_noise)  # 3
        out_noise = out_noise + x
        out_noise = out_noise[:, :, :H, :W]

        return out_noise, mag_image, x_center, mask

if __name__ == '__main__':
    a = paddle.randn([1, 3, 400, 600])
    model = FourLLIE_my()
    out_noise, mag_image, x_center, mask = model(a)
    print(mag_image.shape)
