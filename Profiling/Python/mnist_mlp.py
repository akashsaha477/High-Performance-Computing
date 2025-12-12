'''Code taken ispiration from
https://www.tensorflow.org/datasets/keras_example
https://github.com/gursky1/MNIST-Tensorflow-2
https://github.com/lnmurthy/TensorFlow-ImageClassificaiton/blob/main/ImageClassification.ipynb
'''

# paths + hyperparams
import os
import time
import csv
import random
import datetime
import numpy as np

# --- IMPORTANT: avoid loading conflicting native TF libs from your shell env
# If you must keep DYLD_LIBRARY_PATH / LIBRARY_PATH in your shell for other projects,
# remove them from the environment before importing tensorflow so the Python TF wheel
# loads its own correct native libs.
os.environ.pop("DYLD_LIBRARY_PATH", None)
os.environ.pop("LIBRARY_PATH", None)
os.environ.pop("LD_LIBRARY_PATH", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

#directory paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR_BEST = os.path.join(BASE_DIR, "saved_mlp_best")
EXPORT_DIR_LAST = os.path.join(BASE_DIR, "saved_mlp_last")
TIMING_CSV = os.path.join(BASE_DIR, "python_timings.csv")
ACCURACY_CSV = os.path.join(BASE_DIR, "python_accuracy.csv")
EVAL_CSV = os.path.join(BASE_DIR, "eval_data.csv")
TB_LOGDIR = os.path.join(BASE_DIR, "tb_log", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


#hyperparameters

EPOCHS = 100
BATCH_SIZE = 64           # reduced batch size (often helps)
N_EVAL = 1000
VALIDATION_SPLIT = 0.1
SEED = 12345
USE_TB = True
PATIENCE_ES = 12
PATIENCE_RLR = 6
MIN_LR = 1e-6

# Choose model type: 'cnn' or 'mlp'
# cnn gives much better accuracy on MNIST; mlp is a tuned MLP variant
MODEL_TYPE = 'cnn'  # options: 'cnn' or 'mlp'

# Data augmentation (only used when MODEL_TYPE == 'cnn')
USE_AUG = False

# learning rate
INITIAL_LR = 5e-4     # tuned LR for both MLP and CNN

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

#try to enable deterministic ops

try:
    tf.config.experimental.enable_op_determinism()
except:
    pass


#timer for profiling

def now(): return time.perf_counter()

os.makedirs(EXPORT_DIR_BEST, exist_ok=True)
os.makedirs(EXPORT_DIR_LAST, exist_ok=True)
if USE_TB:
    os.makedirs(TB_LOGDIR, exist_ok=True)

timings = []
def record(step, t): timings.append({"step": step, "time(s)": float(t)})


#Load MNIST dataset

t0 = now()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
t1 = now(); record("data load", t1 - t0)

t0 = now()

# scale to 0..1
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
t1 = now(); record("preprocess", t1 - t0)

# prepare flattened arrays for MLP / CSV export

t0 = now()

x_train_flat = x_train.reshape((-1, 784))
x_test_flat  = x_test.reshape((-1, 784))

t0 = now()



t0 = now()

num_samples = x_train_flat.shape[0]
val_count = int(num_samples * VALIDATION_SPLIT)
train_count = num_samples - val_count

indices = np.arange(num_samples)
np.random.shuffle(indices)
train_idx = indices[:train_count]
val_idx = indices[train_count:]

# splits for both flat and image forms
x_train_split_flat = x_train_flat[train_idx]
y_train_split = y_train[train_idx]
x_val_flat = x_train_flat[val_idx]
y_val = y_train[val_idx]

t1=now(); record("train/val split", t1 - t0)

# image-shaped variants (for CNN)
x_train_img = x_train.reshape((-1, 28, 28, 1))
x_test_img  = x_test.reshape((-1, 28, 28, 1))
x_train_split_img = x_train_img[train_idx]
x_val_img = x_train_img[val_idx]


#Return a small tuned MLP suitable for MNIST
def build_model_mlp(drop=0.1):
    t0 = now()
    
    return keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(drop),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(drop),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
        t1=now(); record("mlp", t1 - t0)
    ])

#Return a small CNN (recommended for better MNIST accuracy)
# FIX A applied: model now accepts image inputs directly
def build_model_cnn():
    return keras.Sequential([

        t0 = now()
        
        layers.Input(shape=(28, 28, 1)),            # <- accept images directly
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPool2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPool2D(2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')

        t1=now(); record("mlp", t1 - t0)
    ])


# Build model based on chosen type
t0 = now()
if MODEL_TYPE == 'cnn':
    model = build_model_cnn()
else:
    model = build_model_mlp(0.1)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
t1 = now(); record("model_build_compile", t1 - t0)

best_weights_path = os.path.join(BASE_DIR, "best_weights.weights.h5")


# we only save weights not the whole model
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


# optional simple data augmentation pipeline (works when using CNN)
if USE_AUG and MODEL_TYPE == 'cnn':
    data_augmentation = keras.Sequential([
        layers.Resizing(28,28),
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
    ])
else:
    data_augmentation = None


# input pipelines from in memory numpy arrays
if MODEL_TYPE == 'cnn':
    if USE_AUG:
        # pipeline that keeps images (not flattened) so augmentation can apply naturally
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split_img, y_train_split))
        train_dataset = train_dataset.shuffle(10000, seed=SEED).map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE
        ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val_img, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        x_test_proc = x_test_img
    else:
        # no augmentation, use pre-shaped image arrays
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split_img, y_train_split)).shuffle(10000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset   = tf.data.Dataset.from_tensor_slices((x_val_img, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        x_test_proc = x_test_img
else:
    # MLP branch uses flattened arrays
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_split_flat, y_train_split)).shuffle(10000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset   = tf.data.Dataset.from_tensor_slices((x_val_flat, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    x_test_proc = x_test_flat


# for 1 epoch per loop lets us record per-epoch times easily

epoch_results = []

for epoch in range(1, EPOCHS + 1):
    e0 = now()
    history = model.fit(
        train_dataset,
        epochs=1,
        verbose=1,            # change to 0 for quieter runs
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
loss, test_acc = model.evaluate(x_test_proc, y_test, verbose=0)
t1 = now(); record("evaluation", t1 - t0)
print("Final test accuracy:", test_acc)

t0 = now()
# use model.save for robustness
model.save(EXPORT_DIR_LAST, save_format='tf')
t1 = now(); record("export_last", t1 - t0)


# if we saved best weights earlier it makes the same architecture to load them

if os.path.exists(best_weights_path):
    best = build_model_cnn() if MODEL_TYPE == 'cnn' else build_model_mlp(0.1)
    best.compile(optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    best.load_weights(best_weights_path)
    best.evaluate(x_test_proc, y_test, verbose=0)
    t0 = now()
    best.save(EXPORT_DIR_BEST, save_format='tf')
    t1 = now(); record("export_best", t1 - t0)


#inference timing on a small batch

sample = x_test_proc[:128]
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

N = min(N_EVAL, x_test_flat.shape[0])
with open(EVAL_CSV, "w", newline="") as f:
    w = csv.writer(f)
    header = ["label"] + [f"p{i}" for i in range(784)]
    w.writerow(header)
    for i in range(N):
        #flattened pixel values (string)
        w.writerow([int(y_test[i])] + [f"{float(x):.6f}" for x in x_test_flat[i]])

print("done")