import paddle

# 创建随机张量 a 和 b
a = paddle.randn([1, 62, 50, 74])
b = paddle.randn([1, 62, 50, 75])

# 使用 reshape 替代 resize_
b = b.reshape([1, 62, 50, 74])
print(b.shape)  # 打印输出形状

