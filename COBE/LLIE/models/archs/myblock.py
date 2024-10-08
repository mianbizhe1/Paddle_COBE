import paddle
import paddle.nn as nn
from einops import rearrange


class FreBlock(nn.Layer):
    '''
    The Fourier Processing (FP) Block in paper
    '''
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
        x_freq = paddle.fft.rfft2(self.fpre(x), norm='backward')
        mag = paddle.abs(x_freq)
        pha = paddle.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        real = mag * paddle.cos(pha)
        imag = mag * paddle.sin(pha)
        x_out = paddle.complex(real, imag)
        x_out = paddle.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_out + x


class SpaBlock(nn.Layer):
    '''
    The Spatial Processing (SP) Block in paper
    '''
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2D(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2D(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1))

    def forward(self, x):
        return x + self.block(x)


class Attention(nn.Layer):
    '''
    Attention module: A part in the frequency-spatial interaction block
    '''
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = self.create_parameter([num_heads, 1, 1], default_initializer=nn.initializer.Constant(1.0))

        self.kv = nn.Conv2D(dim, dim * 2, kernel_size=1, bias_attr=bias)
        self.kv_dwconv = nn.Conv2D(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias_attr=bias)
        self.q = nn.Conv2D(dim, dim, kernel_size=1, bias_attr=bias)
        self.q_dwconv = nn.Conv2D(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias_attr=bias)
        self.project_out = nn.Conv2D(dim, dim, kernel_size=1, bias_attr=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = paddle.split(kv, 2, axis=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = paddle.nn.functional.normalize(q, axis=-1)
        k = paddle.nn.functional.normalize(k, axis=-1)

        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.temperature
        attn = paddle.nn.functional.softmax(attn, axis=-1)

        out = paddle.matmul(attn, v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FuseBlock(nn.Layer):
    '''
    The FuseBlock: The module to make frequency-spatial interaction block
    '''
    def __init__(self, nc):
        super(FuseBlock, self).__init__()
        self.fre = nn.Conv2D(nc, nc, 3, 1, 1)
        self.spa = nn.Conv2D(nc, nc, 3, 1, 1)
        self.fre_att = Attention(dim=nc)
        self.spa_att = Attention(dim=nc)
        self.fuse = nn.Sequential(nn.Conv2D(2*nc, nc, 3, 1, 1), nn.Conv2D(nc, 2*nc, 3, 1, 1), nn.Sigmoid())

    def forward(self, fre, spa):
        ori = spa
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa) + fre
        spa = self.spa_att(spa, fre) + spa
        fuse = self.fuse(paddle.concat([fre, spa], axis=1))
        fre_a, spa_a = paddle.split(fuse, 2, axis=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        res = paddle.where(paddle.isnan(res), paddle.to_tensor(1e-5), res)
        return res


class FSAIO(nn.Layer):
    '''
    Frequency Spatial All in One Module
    '''
    def __init__(self, nc):
        super(FSAIO, self).__init__()
        self.freblock = FreBlock(nc=nc)
        self.spablock = SpaBlock(nc=nc)
        self.fuseblock = FuseBlock(nc=nc)

    def forward(self, fre, spa):
        fre_out = self.freblock(fre)
        spa_out = self.spablock(spa)
        spa_out = self.fuseblock(fre=fre_out, spa=spa_out)
        return fre_out, spa_out


if __name__ == '__main__':
    spa = paddle.randn([1, 64, 64, 64])
    fre = paddle.randn([1, 64, 64, 64])
    model = FSAIO(nc=64)
    fre_out, spa_out = model(spa=spa, fre=fre)
    print(fre_out.shape)
    print(spa_out.shape)
