import os
import time
import csv
import random
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR_BEST = os.path.join(BASE_DIR, "saved_mlp_best")
EXPORT_DIR_LAST = os.path.join(BASE_DIR, "saved_mlp_last")
TIMING_CSV = os.path.join(BASE_DIR, "python_timings.csv")
ACCURACY_CSV = os.path.join(BASE_DIR, "python_accuracy.csv")
EVAL_CSV = os.path.join(BASE_DIR, "eval_data.csv")
TB_LOGDIR = os.path.join(BASE_DIR, "tb_log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

EPOCHS = 100
BATCH_SIZE = 128
N_EVAL = 1000
VALIDATION_SPLIT = 0.1
SEED = 12345
USE_TB = True
PATIENCE_ES = 12
PATIENCE_RLR = 6
MIN_LR = 1e-6

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except:
    pass

def now(): return time.perf_counter()

os.makedirs(EXPORT_DIR_BEST, exist_ok=True)
os.makedirs(EXPORT_DIR_LAST, exist_ok=True)
if USE_TB:
    os.makedirs(TB_LOGDIR, exist_ok=True)

timings = []
def record(step, t): timings.append({"step": step, "time_s": float(t)})

t0 = now()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
t1 = now(); record("data_load", t1 - t0)

t0 = now()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = x_train.reshape((-1, 784))
x_test  = x_test.reshape((-1, 784))
t1 = now(); record("preprocess", t1 - t0)

num_samples = x_train.shape[0]
val_count = int(num_samples * VALIDATION_SPLIT)
train_count = num_samples - val_count

indices = np.arange(num_samples)
np.random.shuffle(indices)
train_idx = indices[:train_count]
val_idx = indices[train_count:]

x_train_split = x_train[train_idx]
y_train_split = y_train[train_idx]
x_val = x_train[val_idx]
y_val = y_train[val_idx]

def build_model(drop=0.2):
    return keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(drop),
        layers.Dense(256, activation='relu'),
        layers.Dropout(drop),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

t0 = now()
model = build_model(0.2)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
t1 = now(); record("model_build_compile", t1 - t0)

best_weights_path = os.path.join(BASE_DIR, "best_weights.weights.h5")

callbacks = [
    keras.callbacks.ModelCheckpoint(best_weights_path,
        monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=0),
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE_ES, restore_best_weights=False, verbose=0),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=PATIENCE_RLR, min_lr=MIN_LR, verbose=0)
]

if USE_TB:
    callbacks.append(keras.callbacks.TensorBoard(log_dir=TB_LOGDIR, histogram_freq=0))

train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split)).shuffle(10000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset   = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

epoch_results = []

for epoch in range(1, EPOCHS + 1):
    e0 = now()
    history = model.fit(
        train_dataset,
        epochs=1,
        verbose=0,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    e1 = now()

    train_acc = history.history["accuracy"][-1]
    val_acc   = history.history["val_accuracy"][-1]
    epoch_time = e1 - e0

    epoch_results.append({
        "epoch": epoch,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "epoch_time_s": epoch_time
    })
    record(f"epoch_{epoch}", epoch_time)

    print(f"Epoch {epoch}/{EPOCHS} — train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}  time: {epoch_time:.2f}s")

    if getattr(model, "stop_training", False):
        print("Stopped early at epoch", epoch)
        break

t0 = now()
loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
t1 = now(); record("evaluation", t1 - t0)
print("Final test accuracy:", test_acc)

t0 = now()
model.export(EXPORT_DIR_LAST)
t1 = now(); record("export_last", t1 - t0)

if os.path.exists(best_weights_path):
    best = build_model(0.2)
    best.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    best.load_weights(best_weights_path)
    best.evaluate(x_test, y_test, verbose=0)
    t0 = now()
    best.export(EXPORT_DIR_BEST)
    t1 = now(); record("export_best", t1 - t0)

sample = x_test[:128]
t0 = now()
_ = model.predict(sample, batch_size=128, verbose=0)
t1 = now(); record("python_inference_128", t1 - t0)

with open(TIMING_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step","time_s"])
    w.writeheader()
    for r in timings:
        w.writerow(r)

with open(ACCURACY_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["epoch","train_accuracy","val_accuracy","epoch_time_s"])
    w.writeheader()
    for r in epoch_results:
        w.writerow(r)
    w.writerow({"epoch": "test", "train_accuracy": None, "val_accuracy": float(test_acc), "epoch_time_s": None})

N = min(N_EVAL, x_test.shape[0])
with open(EVAL_CSV, "w", newline="") as f:
    w = csv.writer(f)
    header = ["label"] + [f"p{i}" for i in range(784)]
    w.writerow(header)
    for i in range(N):
        w.writerow([int(y_test[i])] + [f"{float(x):.6f}" for x in x_test[i]])

print("done")
