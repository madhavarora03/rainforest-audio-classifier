import torch
from torch import nn
import torch.nn.functional as F
import ai_edge_torch
import tensorflow as tf
import numpy as np

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

model = M5(n_input=1, n_output=2)
model.load_state_dict(torch.load("./checkpoints/model_M5.pth", map_location=torch.device('cpu')))
model = model.cpu()

sample_input = (torch.randn(1, 1, 8000),)

edge_model = ai_edge_torch.convert(model.eval(), sample_input)

edge_model.export("M5_model.tflite")

interpreter = tf.lite.Interpreter(model_path="M5_model.tflite")
interpreter.allocate_tensors()
tensor_details = interpreter.get_tensor_details()

for tensor in tensor_details:
    print(f"Tensor name: {tensor['name']}, Tensor index: {tensor['index']}, Tensor shape: {tensor['shape']}, Tensor type: {tensor['dtype']}")

edge_output = edge_model(*sample_input)
torch_output = model(*sample_input)

assert np.allclose(torch_output.detach().numpy(), edge_output, atol=1e-5)