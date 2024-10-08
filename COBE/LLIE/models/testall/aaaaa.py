import paddle
import paddle.nn as nn

# 定义参数
dim_level = 31

# 创建卷积层
conv1 = nn.Conv2D(in_channels=dim_level, out_channels=dim_level * 2, kernel_size=4, stride=2, padding=1, bias_attr=False)

# 创建输入张量
input = paddle.randn([1, 31, 100, 150])

# 通过第一个卷积层
output = conv1(input)
print(output.shape)  # 打印输出形状

# 更新 dim_level
dim_level = dim_level * 2

# 创建第二个卷积层
conv2 = nn.Conv2D(in_channels=dim_level, out_channels=dim_level * 2, kernel_size=4, stride=2, padding=1, bias_attr=False)

# 更新 dim_level
dim_level = dim_level * 2

# 创建转置卷积层
conv3 = nn.Conv2DTranspose(in_channels=dim_level, out_channels=dim_level // 2, kernel_size=2, stride=2, padding=0, output_padding=0)

# 通过第二个卷积层
output2 = conv2(output)
print(output2.shape)  # 打印输出形状

# 通过转置卷积层
output3 = conv3(output2)
print(output3.shape)  # 打印输出形状

# 创建另一个转置卷积层
conv4 = nn.Conv2DTranspose(in_channels=31, out_channels=31 // 2, kernel_size=2, stride=2, padding=0, output_padding=0)

# 通过转置卷积层
outout = conv4(input)
print(outout.shape)  # 打印输出形状
