import torch

from attention import AdditiveAttention

x = torch.randn(4, 25, 2048)

model = AdditiveAttention(2048)

context, weights = model(x)

print("Input:", x.shape)
print("Context:", context.shape)
print("Weights:", weights.shape)