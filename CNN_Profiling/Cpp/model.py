import os
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


SEED = 12345
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except:
    pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KERAS_MODEL_PATH = os.path.join(BASE_DIR, "cnn_trained.keras")
CAPI_EXPORT_DIR  = os.path.join(BASE_DIR, "cnn_capi_export")


EPOCHS = 100
BATCH_SIZE = 128
LR = 5e-4
VALIDATION_SPLIT = 0.1


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Reshape for CNN
x_train = x_train.reshape((-1, 28, 28, 1))
x_test  = x_test.reshape((-1, 28, 28, 1))


def build_cnn():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPool2D(2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(10, activation="softmax")
    ])

model = build_cnn()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


print("\nTraining CNN...\n")

model.fit(
    x_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    verbose=2
)


loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc:.4f}")


model.save(KERAS_MODEL_PATH)
print("Saved Keras model to:", KERAS_MODEL_PATH)

print("\nExporting CNN for TensorFlow C API...")

model.export(CAPI_EXPORT_DIR)

print("CNN exported for C API at:")
print(CAPI_EXPORT_DIR)





print("\nVerify with:")
print(f"saved_model_cli show --dir {CAPI_EXPORT_DIR} --all")