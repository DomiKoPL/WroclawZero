import torch
import torch.nn as nn
from torchsummary import summary

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=4, stride=1, padding=0),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(384, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 11)
)

input = torch.zeros((1, 1, 9, 7)).cuda()
model.to('cuda')
model(input)
summary(model, (1, 9, 7))