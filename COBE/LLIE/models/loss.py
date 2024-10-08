import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle import ParamAttr

class CharbonnierLoss(nn.Layer):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = paddle.sum(paddle.sqrt(diff * diff + self.eps))
        return loss


class CharbonnierLoss2(nn.Layer):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = paddle.mean(paddle.sqrt(diff * diff + self.eps))
        return loss


class VGG19(nn.Layer):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = paddle.vision.models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Layer):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


def gaussian(window_size, sigma):
    gauss = paddle.to_tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / paddle.sum(gauss)


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.matmul(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand([channel, 1, window_size, window_size]).contiguous()
    return paddle.to_tensor(window)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return paddle.mean(ssim_map)
    else:
        return paddle.mean(ssim_map, axis=[1, 2, 3])


class SSIM(nn.Layer):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.shape

        if channel == self.channel and self.window.type() == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if paddle.is_cuda():
                window = window.cuda()
            window = window.astype(img1.dtype)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.shape
    window = create_window(window_size, channel)

    if paddle.is_cuda():
        window = window.cuda()
    window = window.astype(img1.dtype)

    return _ssim(img1, img2, window, window_size, channel, size_average)
