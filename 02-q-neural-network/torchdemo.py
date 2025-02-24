import torch
from torch import nn
from sklearn.datasets import load_iris

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 32)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        print(f"DEBUG: x: {x}")
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
model=MLP()

print(f"DEBUG: model: {model}")

for p in model.parameters():
    print(f"DEBUG: p: {p}")
    print(f"DEBUG: p.shape: {p.shape}")

v=torch.tensor([1,2,3,4],dtype=torch.float32)

print(f"DEBUG: model(v): {model(v)}")

iris=load_iris()
X=torch.tensor(iris.data,dtype=torch.float32)
y=torch.tensor(iris.target,dtype=torch.float32)

print(f"DEBUG: X.shape: {X.shape}")
print(f"DEBUG: y.shape: {y.shape}")

opt=torch.optim.Adam(model.parameters(),lr=0.0001)

pred=model(X)
print(f"DEBUG: pred.shape: {pred.shape}")

for i in range(10000):
    pred=model(X)
    loss=((pred[:,0]-y)**2).sum()/pred.shape[0]

    opt.zero_grad()
    loss.backward()
    opt.step()

accuracy=(torch.abs(pred[:,0]-y)<.49).sum()/150
print(f"DEBUG: accuracy: {accuracy}")
