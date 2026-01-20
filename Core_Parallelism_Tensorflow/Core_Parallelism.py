import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # FORCE CPU ONLY


import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


import time
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random

print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())
print("Using 1 CPU cores / 1 threads")

SEED = 12345
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = y_train.squeeze()
y_test  = y_test.squeeze()


def resnet_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def build_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = resnet_block(x, 32)
    x = resnet_block(x, 32)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = build_model()
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


start = time.perf_counter()

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    verbose=1
)

train_time = time.perf_counter() - start


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("Test Accuracy:", test_acc)
print("Training Time (sec):", train_time)


results = pd.DataFrame([{
    "cores": 1,
    "threads": 1,
    "train_time_sec": train_time,
    "test_accuracy": test_acc
}])

results.to_csv("/kaggle/working/cpu_1core_results.csv", index=False)

print("Saved: /kaggle/working/cpu_1core_results.csv")