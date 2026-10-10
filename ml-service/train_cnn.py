"""Train a basic CNN classifier from labelled jewellery image folders.

Expected layout: training/jewellery_images/<class-name>/<image-file>.
The serving pipeline uses the same 224x224 RGB, 0–1 preprocessing.
"""
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training/jewellery_images"
OUT = ROOT / "models/jewellery_cnn.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train = keras.utils.image_dataset_from_directory(
    DATA, validation_split=0.2, subset="training", seed=71,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int",
)
valid = keras.utils.image_dataset_from_directory(
    DATA, validation_split=0.2, subset="validation", seed=71,
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int",
)
classes = train.class_names
train = train.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)
valid = valid.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(), layers.Dropout(0.25),
    layers.Dense(len(classes), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train, validation_data=valid, epochs=20,
          callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
OUT.parent.mkdir(parents=True, exist_ok=True)
model.save(OUT)
(ROOT / "models/jewellery_classes.json").write_text(json.dumps(classes, indent=2))
print(f"Saved {OUT}; set JEWELLERY_CLASSES='{json.dumps(classes)}' when serving.")
