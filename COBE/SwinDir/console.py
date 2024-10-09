import onnxruntime
import numpy as np

upscale = 4
window_size = 8
height = 16
width = 16

# 加载 ONNX 模型
model_path = "nasnet.onnx"
session = onnxruntime.InferenceSession(model_path)

# 准备输入数据（NumPy 数组）
input_data = np.random.rand(1, 3, height, width).astype(np.float32)  # 示例输入

# 获取模型输入名
input_name = session.get_inputs()[0].name
print({input_name: input_data})
print(input_data.shape)
# 推理
output = session.run(None, {input_name: input_data})

# 输出结果
# print(output)
print(output[0])