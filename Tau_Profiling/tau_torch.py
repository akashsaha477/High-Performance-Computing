import os
import sys
import pytau
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


pytau.setNode(0)

data_timer = pytau.profileTimer("DATA_LOADING")
model_timer = pytau.profileTimer("MODEL_INIT")
train_timer = pytau.profileTimer("TRAINING_LOOP")
forward_timer = pytau.profileTimer("FORWARD")
backward_timer = pytau.profileTimer("BACKWARD")
optim_timer = pytau.profileTimer("OPTIMIZER")
eval_timer = pytau.profileTimer("EVALUATION")


INTRA = int(sys.argv[1])
INTER = int(sys.argv[2])
BATCH = int(sys.argv[3])

torch.set_num_threads(INTRA)
torch.set_num_interop_threads(INTER)


pytau.start(data_timer)

transform = transforms.ToTensor()

train_ds = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=0
)

pytau.stop(data_timer)


pytau.start(model_timer)

class BasicBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        r = x
        x = torch.relu(self.c1(x))
        x = self.c2(x)
        return torch.relu(x + r)

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 32, 3, padding=1)
        self.r1 = BasicBlock(32)
        self.r2 = BasicBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.c(x))
        x = self.r1(x)
        x = self.r2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SmallResNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

pytau.stop(model_timer)


pytau.start(train_timer)

model.train()

for x, y in train_loader:

    # Forward
    pytau.start(forward_timer)
    out = model(x)
    loss = criterion(out, y)
    pytau.stop(forward_timer)

    # Backward
    pytau.start(backward_timer)
    optimizer.zero_grad()
    loss.backward()
    pytau.stop(backward_timer)

    # Optimizer
    pytau.start(optim_timer)
    optimizer.step()
    pytau.stop(optim_timer)

pytau.stop(train_timer)



pytau.start(eval_timer)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in test_loader:
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

pytau.stop(eval_timer)

print("Accuracy:", correct / total)


pytau.dbDump()