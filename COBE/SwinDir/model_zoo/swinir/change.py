import torch
import torch.onnx
from SwinDir.models.network_swinir import SwinIR as net


upscale = 4
window_size = 8
height = (1024 // upscale // window_size + 1) * window_size
width = (720 // upscale // window_size + 1) * window_size

# 创建一个随机输入张量
dummy_input = torch.randn((1, 3, height, width))


model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
            mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')

param_key_g = 'params'

pretrained_model = torch.load("002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth")
model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                      strict=True)

# 创建一个虚拟输入来模拟模型推理时的输入
torch.onnx.export(model, dummy_input, "nasnet.onnx", verbose=True)
