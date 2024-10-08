import paddle
import paddle.nn as nn
from einops import rearrange

# 将频域的部分和空域的部分融合起来，得到空频融合的模块
class Attention(nn.Layer):
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
    def __init__(self, channels):
        super(FuseBlock, self).__init__()
        self.fre = nn.Conv2D(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2D(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)
        self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2D(2*channels, channels, 3, 1, 1), nn.Conv2D(channels, 2*channels, 3, 1, 1), nn.Sigmoid())

    def forward(self, spa, fre):
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

if __name__ == '__main__':
    spa = paddle.randn([1, 64, 64, 64])
    fre = paddle.randn([1, 64, 64, 64])
    model = FuseBlock(channels=64)
    out = model(spa=spa, fre=fre)
    print(out.shape)
