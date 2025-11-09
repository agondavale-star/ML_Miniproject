from google.colab import files
from zipfile import ZipFile
import os, glob

print("👉 Choose your dataset.zip file (it must contain COVID/ and NORMAL/ folders)")
uploaded = files.upload()

# Unzip to /content/
for fn in uploaded.keys():
    if fn.endswith('.zip'):
        with ZipFile(fn, 'r') as z:
            z.extractall('/content/')
        print(f"✅ Extracted {fn}")


for f in os.listdir('/content'):
    if os.path.isdir(f) and 'COVID' in os.listdir(f):
        os.rename(f, 'dataset')

# Verify structure
print("\nContents of /content/dataset:")
print(os.listdir('/content/dataset'))
print("COVID images:", len(glob.glob('/content/dataset/COVID/*.png')))
print("NORMAL images:", len(glob.glob('/content/dataset/NORMAL/*.png')))


# ============================================
# STEP 2: Train simple CNN on X-ray dataset
# ============================================
!pip install -q tensorflow matplotlib scikit-learn opencv-python

import numpy as np, matplotlib.pyplot as plt, itertools
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report

DATA_DIR = '/content/dataset'

# Data pipeline
IMG_SIZE = (128, 128)
BATCH = 16
VAL_SPLIT = 0.2

datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

train_ds = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', subset='training')

val_ds = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', subset='validation')

# Model (simple CNN)
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=IMG_SIZE+(3,)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=8, validation_data=val_ds)

# Accuracy curves
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy vs Epochs'); plt.show()

# Evaluate
val_pred = (model.predict(val_ds) > 0.5).astype(int).ravel()
y_true   = val_ds.classes
labels   = list(val_ds.class_indices.keys())

acc = (val_pred == y_true).mean()
print(f"\n✅ Validation Accuracy: {acc*100:.2f}%")

cm = confusion_matrix(y_true, val_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:")
print(classification_report(y_true, val_pred, target_names=labels))

# Plot confusion matrix
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix'); plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()
