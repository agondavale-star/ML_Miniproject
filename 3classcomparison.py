# ===== STEP 2B-1: Upload new 3-class dataset.zip =====
from google.colab import files
from zipfile import ZipFile
import os, glob

print("👉 Upload dataset3.zip (with COVID, NORMAL, Viral Pneumonia)")
up = files.upload()
for fn in up:
    if fn.endswith('.zip'):
        with ZipFile(fn,'r') as z: z.extractall('/content/')
        print("✅ Extracted:", fn)

# rename top folder to /content/dataset (if needed)
for f in os.listdir('/content'):
    p = os.path.join('/content', f)
    if os.path.isdir(p) and set(['COVID','NORMAL','Viral Pneumonia']).issubset(set(os.listdir(p))):
        # We’re skipping rename; just note which folder matched
        print("✅ Found dataset folder:", p)
        DATA_DIR = p  # Save this path for later use

print("Contents:", os.listdir(DATA_DIR))
for cls in ['COVID','NORMAL','Viral Pneumonia']:
    print(cls, len(glob.glob(f'/content/dataset/{cls}/*.png')))


# ===========================
# Robust 3-Class Trainer (Auto-fix dataset paths)
# ===========================
!pip install -q tensorflow matplotlib scikit-learn

import os, glob, re, itertools, shutil, zipfile, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ---- Utilities ----
IMG_EXT = ('.png','.jpg','.jpeg','.bmp','.tif','.tiff')

def extract_all_zips(root='/content'):
    zips = [f for f in os.listdir(root) if f.lower().endswith('.zip')]
    for z in zips:
        try:
            with zipfile.ZipFile(os.path.join(root, z), 'r') as Z:
                Z.extractall(root)
            print(f"✅ Extracted: {z}")
        except Exception as e:
            print(f"⚠️ Could not extract {z}: {e}")

def find_candidate_dirs(root='/content'):
    cand = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip system/sample_data
        if '/.' in dirpath or '/proc' in dirpath or '/usr' in dirpath:
            continue
        # Heuristic: folder with >= 5 images
        img_count = sum(1 for f in filenames if f.lower().endswith(IMG_EXT))
        if img_count >= 5:
            cand.append((dirpath, img_count))
    cand.sort(key=lambda x: -x[1])
    return [c[0] for c in cand]

def match_class_dir(candidates, patterns):
    # patterns: list of substrings to look for in path (case-insensitive)
    for c in candidates:
        low = c.lower()
        if any(p in low for p in patterns):
            # ensure it's a dir with images
            imgs = glob.glob(os.path.join(c, '*'))
            if any(i.lower().endswith(IMG_EXT) for i in imgs):
                return c
    return None

def ensure_clean_dataset(root='/content', target='dataset'):
    target_dir = os.path.join(root, target)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def move_images(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    moved = 0
    for f in glob.glob(os.path.join(src_dir, '*')):
        if f.lower().endswith(IMG_EXT):
            shutil.move(f, os.path.join(dst_dir, os.path.basename(f)))
            moved += 1
    return moved

# ---- 1) Extract any uploaded zips (handles "dataset 2.zip" etc.) ----
extract_all_zips('/content')

# ---- 2) Try to locate class folders anywhere under /content ----
candidates = find_candidate_dirs('/content')

# Flexible name patterns (case-insensitive)
covid_patterns  = ['covid']
normal_patterns = ['normal', 'healthy', 'non-covid', 'noncovid']
vp_patterns     = ['viral pneumonia', 'viral_pneumonia', 'viralpneumonia', 'pneumonia viral']

covid_dir = match_class_dir(candidates, covid_patterns)
normal_dir = match_class_dir(candidates, normal_patterns)
vp_dir = match_class_dir(candidates, vp_patterns)

# If a top-level contains "dataset" with subfolders, prefer that
for d in [d for d in os.listdir('/content') if os.path.isdir(os.path.join('/content', d))]:
    p = os.path.join('/content', d)
    subs = [s for s in os.listdir(p) if os.path.isdir(os.path.join(p, s))]
    subs_low = [s.lower() for s in subs]
    if any('covid' in s for s in subs_low) and any('normal' in s for s in subs_low) and any('viral' in s and 'pneumonia' in s for s in subs_low):
        # Map within this folder
        maybe = [os.path.join(p, s) for s in subs]
        covid_dir  = covid_dir  or match_class_dir(maybe, covid_patterns)
        normal_dir = normal_dir or match_class_dir(maybe, normal_patterns)
        vp_dir     = vp_dir     or match_class_dir(maybe, vp_patterns)

# ---- 3) Assemble a clean /content/dataset/{COVID,NORMAL,Viral Pneumonia} ----
DATA_DIR = '/content/dataset'
os.makedirs(DATA_DIR, exist_ok=True)

def pick_or_fail(found_dir, fallback_hint):
    if found_dir and os.path.isdir(found_dir):
        return found_dir
    # Try common exact subfolder spellings under /content
    commons = [
        f"/content/dataset/{fallback_hint}",
        f"/content/{fallback_hint}",
        f"/content/COVID-19_Radiography_Dataset/{fallback_hint}",
        f"/content/COVID-19 Radiography Database/{fallback_hint}",
    ]
    for c in commons:
        if os.path.isdir(c):
            return c
    return None

covid_dir  = pick_or_fail(covid_dir, 'COVID')
normal_dir = pick_or_fail(normal_dir, 'NORMAL')
vp_dir     = pick_or_fail(vp_dir, 'Viral Pneumonia')

missing = []
if not covid_dir:  missing.append('COVID')
if not normal_dir: missing.append('NORMAL')
if not vp_dir:     missing.append('Viral Pneumonia')

if missing:
    raise AssertionError(f"Missing class folder(s): {', '.join(missing)}. Make sure your zip has these three class folders.")

# Clean old if exists
for cls in ['COVID','NORMAL','Viral Pneumonia']:
    cls_dir = os.path.join(DATA_DIR, cls)
    if os.path.isdir(cls_dir):
        # leave existing, but it's fine—will still work
        pass

moved_counts = {}
moved_counts['COVID'] = move_images(covid_dir,  os.path.join(DATA_DIR, 'COVID'))
moved_counts['NORMAL'] = move_images(normal_dir, os.path.join(DATA_DIR, 'NORMAL'))
moved_counts['Viral Pneumonia'] = move_images(vp_dir, os.path.join(DATA_DIR, 'Viral Pneumonia'))

# If nothing moved (perhaps already in place), just count existing
def count_images(d): return len([f for f in glob.glob(os.path.join(d, '*')) if f.lower().endswith(IMG_EXT)])

cov_ct  = count_images(os.path.join(DATA_DIR, 'COVID'))
norm_ct = count_images(os.path.join(DATA_DIR, 'NORMAL'))
vp_ct   = count_images(os.path.join(DATA_DIR, 'Viral Pneumonia'))

print(f"📁 DATA_DIR = {DATA_DIR}")
print(f"Images → COVID: {cov_ct}, NORMAL: {norm_ct}, Viral Pneumonia: {vp_ct}")

# ---- 4) Build and train model ----
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

IMG_SIZE=(160,160); BATCH=16; VAL_SPLIT=0.2
datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

classes_order = ['COVID','NORMAL','Viral Pneumonia']
train_ds = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='training', shuffle=True, classes=classes_order
)
val_ds = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='categorical', subset='validation', shuffle=False, classes=classes_order
)
labels = list(val_ds.class_indices.keys())

base = MobileNetV2(include_top=False, input_shape=IMG_SIZE+(3,), weights='imagenet')
base.trainable = False
model = Sequential([ base, GlobalAveragePooling2D(), Dropout(0.2), Dense(3, activation='softmax') ])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=8, validation_data=val_ds, verbose=1)

# ---- 5) Accuracy curve ----
plt.figure()
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy vs Epochs'); plt.legend(); plt.show()

# ---- 6) Evaluate + normalized confusion matrix ----
probs  = model.predict(val_ds)
y_true = val_ds.classes
y_pred = probs.argmax(axis=1)

overall_acc = (y_pred == y_true).mean()
rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
macro_f1 = rep['macro avg']['f1-score']
print(f"✅ Overall Validation Accuracy: {overall_acc*100:.2f}%")
print(f"✅ Macro F1-score: {macro_f1:.3f}")

cm = confusion_matrix(y_true, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure()
plt.imshow(cm_norm, interpolation='nearest')
plt.title('Normalized Confusion Matrix')
plt.colorbar()
ticks = np.arange(len(labels))
plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
th = cm_norm.max()/2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
             ha="center", va="center",
             color="white" if cm_norm[i,j] > th else "black", fontsize=8)
plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout(); plt.show()
