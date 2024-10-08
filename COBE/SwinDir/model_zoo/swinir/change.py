import torch
import paddle

# 步骤 1：加载 PyTorch 模型
torch_model_path = '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth'  # 确保路径正确
torch_model = torch.load(torch_model_path, weights_only=True)  # 使用 weights_only=True

# 打印模型内容以确认
print(torch_model)  # 确认加载的模型是否有参数

# 步骤 2：创建一个字典来存储 Paddle 参数
paddle_model = {}

# 步骤 3：从 PyTorch 模型转换并复制参数到 Paddle 模型
for k, v in torch_model.items():
    print(f"Key: {k}, Value: {v}")  # 打印键和值
    if isinstance(v, torch.Tensor):
        # 将 PyTorch 张量转换为 NumPy 然后转换为 Paddle 张量
        paddle_model[k] = paddle.to_tensor(v.cpu().numpy())
    else:
        # 保留非张量项
        paddle_model[k] = v  # 直接保存非张量项

# 步骤 4：保存转换后的模型
if paddle_model:
    paddle.save(paddle_model, '002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pdparams')
    print("模型成功转换并保存！")
else:
    print("没有参数被保存，因为 paddle_model 为空。")
