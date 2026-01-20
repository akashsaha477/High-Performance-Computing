
# Apple Silicon M1


import os
import sys
import time
import random
import numpy as np
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


if len(sys.argv) != 3:
    print("Usage: python macbook_cpu_full_potential_fixed.py <intra_threads> <interop_threads>")
    sys.exit(1)

NUM_THREADS = int(sys.argv[1])
NUM_INTEROP = int(sys.argv[2])


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def main():


    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(NUM_INTEROP)
    torch.backends.mkldnn.enabled = True
    torch.backends.cudnn.enabled = False

    print("\n===== PYTORCH APPLE SILICON CPU =====")
    print("PyTorch:", torch.__version__)
    print("Intra threads:", torch.get_num_threads())
    print("Interop threads:", torch.get_num_interop_threads())

    
    SEED = 12345
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

  
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )


    class BasicBlock(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(8, c)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            self.gn2 = nn.GroupNorm(8, c)

        def forward(self, x):
            out = torch.relu(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            return torch.relu(out + x)

    class SmallResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1, bias=False),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True)
            )
            self.block1 = BasicBlock(64)
            self.block2 = BasicBlock(64)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = x.to(memory_format=torch.channels_last)
            x = self.stem(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    device = torch.device("cpu")

    model = SmallResNet().to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)


    EPOCHS = 10
    start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {loss_sum/len(train_loader):.4f} "
            f"Time: {time.perf_counter()-epoch_start:.2f}s"
        )

    train_time = time.perf_counter() - start


    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    acc = correct / total

    print("\n===== RESULTS =====")
    print(f"Training time: {train_time:.2f}s")
    print(f"Test accuracy: {acc:.4f}")


    df = pd.DataFrame([{
        "intra_threads": NUM_THREADS,
        "interop_threads": NUM_INTEROP,
        "train_time_sec": train_time,
        "test_accuracy": acc
    }])

    out = f"macbook_cpu_{NUM_THREADS}x{NUM_INTEROP}.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")



if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()