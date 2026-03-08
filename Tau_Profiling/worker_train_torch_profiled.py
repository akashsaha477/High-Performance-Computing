import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import pytau

# =============================
# ARGUMENTS
# =============================

INTRA = int(sys.argv[1])
INTER = int(sys.argv[2])
BATCH = int(sys.argv[3])

# =============================
# THREAD SETTINGS
# =============================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = str(INTRA)
os.environ["MKL_NUM_THREADS"] = str(INTRA)

import torch
torch.set_num_threads(INTRA)
torch.set_num_interop_threads(INTER)

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# =============================
# REPRODUCIBILITY
# =============================

SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# =============================
# TAU TIMERS
# =============================

data_timer = pytau.profileTimer("DATA_LOADING")
model_timer = pytau.profileTimer("MODEL_INIT")
train_timer = pytau.profileTimer("TRAINING_LOOP")
forward_timer = pytau.profileTimer("FORWARD")
backward_timer = pytau.profileTimer("BACKWARD")
optim_timer = pytau.profileTimer("OPTIMIZER")
eval_timer = pytau.profileTimer("EVALUATION")

# =============================
# DATA LOADING
# =============================

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

# =============================
# MODEL
# =============================

pytau.start(model_timer)

class BasicBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        r = x
        x = torch.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return torch.relu(x + r)

class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(32)
        self.r1 = BasicBlock(32)
        self.r2 = BasicBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.b(self.c(x)))
        x = self.r1(x)
        x = self.r2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SmallResNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

pytau.stop(model_timer)

# =============================
# TRAINING
# =============================

EPOCHS = 5

pytau.start(train_timer)

model.train()
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for x, y in tqdm(train_loader):

        pytau.start(forward_timer)
        out = model(x)
        loss = criterion(out, y)
        pytau.stop(forward_timer)

        pytau.start(backward_timer)
        optimizer.zero_grad()
        loss.backward()
        pytau.stop(backward_timer)

        pytau.start(optim_timer)
        optimizer.step()
        pytau.stop(optim_timer)

pytau.stop(train_timer)

# =============================
# EVALUATION
# =============================

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

print("Accuracy:", correct/total)