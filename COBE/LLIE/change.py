import torch
import paddle

# Step 1: Load the PyTorch model
torch_model_path = '83000_G.pth'
torch_model = torch.load(torch_model_path)

# Step 2: Create a dictionary to hold Paddle parameters
paddle_model = {}

# Step 3: Convert and copy parameters from torch model to paddle model
for k, v in torch_model.items():
    paddle_model[k] = paddle.to_tensor(v.cpu().numpy())  # Convert torch tensor to numpy and then to paddle tensor

# Step 4: Save the converted model
paddle.save(paddle_model, '83000_G.pdparams')

print("Model successfully converted and saved!")
