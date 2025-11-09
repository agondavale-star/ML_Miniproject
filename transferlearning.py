# ===== STEP 2A: Transfer Learning on SAME dataset (COVID vs NORMAL) =====
!pip install -q tensorflow matplotlib scikit-learn

import numpy as np, matplotlib.pyplot as plt, itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = '/content/dataset'
IMG_SIZE = (160, 160)
BATCH    = 16
VAL_SPLIT= 0.2

datagen = ImageDataGenerator(
    rescale=1./255, validation_split=VAL_SPLIT,
    rotation_range=5, width_shift_range=0.02, height_shift_range=0.02, zoom_range=0.05
)

train_ds = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='binary',
    subset='training', shuffle=True
)
val_ds = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH, class_mode='binary',
    subset='validation', shuffle=False
)

# 1) Feature extractor (frozen)
base = MobileNetV2(include_top=False, input_shape=IMG_SIZE+(3,), weights='imagenet')
base.trainable = False
model = Sequential([ base, GlobalAveragePooling2D(), Dropout(0.2), Dense(1, activation='sigmoid') ])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cbs = [
    EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint('best_transfer.keras', save_best_only=True, monitor='val_accuracy')
]

hist = model.fit(train_ds, epochs=8, validation_data=val_ds, callbacks=cbs, verbose=1)

# quick curves
plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy']); plt.title('Accuracy'); plt.legend(['train','val']); plt.show()
plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss']); plt.title('Loss'); plt.legend(['train','val']); plt.show()

# evaluate + report
val_pred = (model.predict(val_ds) > 0.5).astype(int).ravel()
y_true   = val_ds.classes
labels   = list(val_ds.class_indices.keys())
acc = (val_pred == y_true).mean()
print(f"\n✅ Validation Accuracy (frozen base): {acc*100:.2f}%")

# 2) Light fine-tuning (unfreeze last ~20 layers)
for layer in base.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
hist2 = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=cbs, verbose=1)

val_pred = (model.predict(val_ds) > 0.5).astype(int).ravel()
acc2 = (val_pred == y_true).mean()
print(f"✅ Validation Accuracy (after fine-tuning): {acc2*100:.2f}%")

cm = confusion_matrix(y_true, val_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, val_pred, target_names=labels))
