import os
import sys
import time
import random
import numpy as np
import pandas as pd

INTRA = int(sys.argv[1])
INTER = int(sys.argv[2])
BATCH = int(sys.argv[3])

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = str(INTRA)
os.environ["MKL_NUM_THREADS"] = str(INTRA)

import torch
torch.set_num_threads(INTRA)
torch.set_num_interop_threads(INTER)

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

SEED = 12345
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

t_data_start = time.perf_counter()

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

data_time = time.perf_counter() - t_data_start

t_model_start = time.perf_counter()

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

model_init_time = time.perf_counter() - t_model_start

forward_time = 0.0
backward_time = 0.0
optim_time = 0.0

t_train_start = time.perf_counter()

model.train()
for _ in range(5):
    for x, y in train_loader:

        t0 = time.perf_counter()
        out = model(x)
        loss = criterion(out, y)
        forward_time += time.perf_counter() - t0

        t1 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        backward_time += time.perf_counter() - t1

        t2 = time.perf_counter()
        optimizer.step()
        optim_time += time.perf_counter() - t2

train_time_total = time.perf_counter() - t_train_start

t_eval_start = time.perf_counter()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

eval_time = time.perf_counter() - t_eval_start
test_acc = correct / total

BASE_DIR = "/Users/akashsaha/Desktop/High-Performance-Computing/Tau_Profiling/hpc_torch_benchmarks"
CSV_DIR = os.path.join(BASE_DIR, "results_torch", "csv")
os.makedirs(CSV_DIR, exist_ok=True)

df = pd.DataFrame([{
    "intra_threads": INTRA,
    "inter_threads": INTER,
    "batch_size": BATCH,
    "data_load_time_sec": data_time,
    "model_init_time_sec": model_init_time,
    "train_time_total_sec": train_time_total,
    "forward_time_sec": forward_time,
    "backward_time_sec": backward_time,
    "optimizer_time_sec": optim_time,
    "eval_time_sec": eval_time,
    "test_accuracy": test_acc
}])

fname = os.path.join(
    CSV_DIR,
    f"profile_intra{INTRA}_inter{INTER}_batch{BATCH}.csv"
)

df.to_csv(fname, index=False)
print("PROFILE SAVED:", fname)