import os
import time
import csv
import struct
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BATCH_SIZE = 100
EPOCHS = 5
MODEL_DIR = "saved_mlp"

DATA_PATH = "/Users/akashsaha/Desktop/High-Performance-Computing/Profiling/C++/data"

def load_mnist_local(path, kind='train'):
    prefix = "train" if kind == 'train' else "t10k"
    lbl_path = os.path.join(path, f"{prefix}-labels-idx1-ubyte")
    img_path = os.path.join(path, f"{prefix}-images-idx3-ubyte")

    if not os.path.exists(lbl_path) or not os.path.exists(img_path):
        raise FileNotFoundError(f"Could not find data files at: {path}")

    with open(lbl_path, 'rb') as lb:
        lb.read(8)
        labels = np.frombuffer(lb.read(), dtype=np.uint8)

    with open(img_path, 'rb') as img:
        img.read(16)
        images = np.frombuffer(img.read(), dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

print(f"Loading data from {DATA_PATH}...")
x_train, y_train = load_mnist_local(DATA_PATH, kind='train')
x_test, y_test = load_mnist_local(DATA_PATH, kind='test')

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print(f"Training for {EPOCHS} epochs...")
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, verbose=1)

if os.path.exists(MODEL_DIR):
    import shutil
    shutil.rmtree(MODEL_DIR)
    
tf.saved_model.save(model, MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")

print("\nStarting Inference Benchmarking...")

results = []
num_samples = x_test.shape[0]
num_batches = int(np.ceil(num_samples / BATCH_SIZE))

_ = model.predict_on_batch(x_test[:BATCH_SIZE])

for i in range(num_batches):
    start_idx = i * BATCH_SIZE
    end_idx = min(start_idx + BATCH_SIZE, num_samples)
    
    batch_x = x_test[start_idx:end_idx]
    
    current_size = batch_x.shape[0]
    if current_size < BATCH_SIZE:
        padding = np.zeros((BATCH_SIZE - current_size, 784), dtype="float32")
        batch_x = np.concatenate([batch_x, padding], axis=0)

    t0 = time.perf_counter()
    _ = model.predict_on_batch(batch_x)
    t1 = time.perf_counter()
    
    latency_ms = (t1 - t0) * 1000.0
    
    results.append({
        "batch_id": i + 1,
        "batch_size": BATCH_SIZE,
        "latency_ms": latency_ms
    })
    
    if i % 10 == 0:
        print(f"Processed batch {i}/{num_batches}")

csv_file = "inference_profiling_python.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["batch_id", "batch_size", "latency_ms"])
    writer.writeheader()
    writer.writerows(results)

print(f"\nBenchmarking complete. Data saved to {csv_file}")