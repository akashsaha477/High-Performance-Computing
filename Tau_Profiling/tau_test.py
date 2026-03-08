import os
import sys
import time
import random
import numpy as np
from tqdm import tqdm
import pytau

# =============================
# TAU HELPERS
# =============================

def tau_timer(name):
    return pytau.profileTimer(name)

def start(timer):
    pytau.start(timer)

def stop(timer):
    pytau.stop(timer)

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

DATA_LOADING = tau_timer("DATA_LOADING")
MODEL_INIT = tau_timer("MODEL_INIT")
TRAINING_LOOP = tau_timer("TRAINING_LOOP")
FORWARD = tau_timer("FORWARD")
BACKWARD = tau_timer("BACKWARD")
OPTIMIZER = tau_timer("OPTIMIZER")
EVALUATION = tau_timer("EVALUATION")

# =============================
# DATA LOADING
# =============================

start(DATA_LOADING)

print("\nLoading CIFAR10 dataset...")
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
print("Data loading time:", data_time)

stop(DATA_LOADING)

# =============================
# MODEL
# =============================

start(MODEL_INIT)

print("\nInitializing model...")
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
print("Model initialization time:", model_init_time)

stop(MODEL_INIT)

# =============================
# TRAINING
# =============================

EPOCHS = 5

forward_time = 0
backward_time = 0
optim_time = 0

print("\nStarting training...")

model.train()

start(TRAINING_LOOP)

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for x, y in tqdm(
        train_loader,
        desc="Training",
        ncols=100,
        ascii=True,
        file=sys.stdout,
        disable=False
    ):

        # Forward
        start(FORWARD)
        t0 = time.perf_counter()
        out = model(x)
        loss = criterion(out, y)
        forward_time += time.perf_counter() - t0
        stop(FORWARD)

        # Backward
        start(BACKWARD)
        t1 = time.perf_counter()
        optimizer.zero_grad()
        loss.backward()
        backward_time += time.perf_counter() - t1
        stop(BACKWARD)

        # Optimizer
        start(OPTIMIZER)
        t2 = time.perf_counter()
        optimizer.step()
        optim_time += time.perf_counter() - t2
        stop(OPTIMIZER)

stop(TRAINING_LOOP)

# =============================
# EVALUATION
# =============================

start(EVALUATION)

print("\nEvaluating model...")
t_eval_start = time.perf_counter()

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in tqdm(
        test_loader,
        desc="Testing",
        ncols=100,
        ascii=True,
        file=sys.stdout,
        disable=False
    ):
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

eval_time = time.perf_counter() - t_eval_start

stop(EVALUATION)

test_acc = correct / total

# =============================
# RESULTS
# =============================

print("\n==============================")
print("RESULTS")
print("==============================")

print("Forward time:", forward_time)
print("Backward time:", backward_time)
print("Optimizer time:", optim_time)
print("Evaluation time:", eval_time)
print("Test accuracy:", test_acc)

# =============================
# WRITE DUMP FILE
# =============================

pytau.dbDump()