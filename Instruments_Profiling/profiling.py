# worker_train_torch_instruments_correct.py

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# ---------------- CONFIG ----------------
INTRA = 4
INTER = 2
BATCH = 128
EPOCHS = 5


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = str(INTRA)
os.environ["MKL_NUM_THREADS"] = str(INTRA)

torch.set_num_threads(INTRA)
torch.set_num_interop_threads(INTER)


# ---------------- PATH ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------- SEED ----------------
SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------- ATTACH WINDOW ----------------
print("\n[INFO] Start recording in Instruments NOW...")
time.sleep(6)


# ---------------- DATA ----------------
transform = transforms.ToTensor()

train_ds = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=0
)


# ---------------- MODEL ----------------
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
        self.c = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(64)
        self.r1 = BasicBlock(64)
        self.r2 = BasicBlock(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.b(self.c(x)))
        x = self.r1(x)
        x = self.r2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


model = SmallResNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


# ---------------- WARMUP ----------------
model.train()
for x, y in train_loader:
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    break


# ---------------- PROFILE REGION ----------------
print("[INFO] Profiling region started")

start = time.perf_counter()

for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(train_loader):

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

end = time.perf_counter()

print(f"[DONE] Time: {end - start:.2f} sec")