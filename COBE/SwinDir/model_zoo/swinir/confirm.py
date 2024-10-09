import onnx

# 加载ONNX模型
onnx_model = onnx.load("model.onnx")

# 验证模型的结构和格式
onnx.checker.check_model(onnx_model)

print("ONNX模型格式正确")
