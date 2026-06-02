# train_local.py — streaming (no big X,y arrays)
import os, random, argparse, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, precision_recall_curve

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument('--resize', type=int, default=672, help='Square resize before patching.')
parser.add_argument('--patch', type=int, default=224, help='Patch size (square).')
parser.add_argument('--stride', type=int, default=112, help='Patch stride.')
parser.add_argument('--batch', type=int, default=32, help='Batch size (patches).')
parser.add_argument('--warmup', type=int, default=5, help='Warmup epochs (frozen backbone).')
parser.add_argument('--finetune', type=int, default=15, help='Finetune epochs.')
parser.add_argument('--top-train', type=int, default=40, help='Unfreeze last N layers.')
parser.add_argument('--random-init', action='store_true', help='Use weights=None (skip ImageNet download).')
parser.add_argument('--limit-per-class', type=int, default=0, help='Limit number of images per class (0=all).')
parser.add_argument('--num-parallel-calls', type=int, default=2, help='tf.data parallelism.')
args, _ = parser.parse_known_args()

# ----------------- Repro -----------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# ----------------- Paths -----------------
ACNE_FOLDER = 'Acne'
CLEAR_FOLDER = 'non-acne'
MODEL_OUT = 'acne_mobilenet_patch_model.h5'
THRESH_FILE = 'best_threshold.json'

# ----------------- Params -----------------
RESIZE = args.resize
PATCH = args.patch
STRIDE = args.stride
BATCH = args.batch
WARMUP_EPOCHS = args.warmup
FINETUNE_EPOCHS = args.finetune
TOP_LAYERS_TO_TRAIN = args.top_train

# ----------------- Collect files -----------------
def list_images(folder, limit=0):
    exts = ('.jpg','.jpeg','.png')
    files = [os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]
    return files[:limit] if limit else files

clear_files = list_images(CLEAR_FOLDER, args.limit_per_class)
acne_files  = list_images(ACNE_FOLDER,  args.limit_per_class)

if not clear_files and not acne_files:
    raise SystemExit("No images found. Check folders.")

all_files = clear_files + acne_files
all_labels = [0]*len(clear_files) + [1]*len(acne_files)
all_groups = [os.path.basename(p) for p in all_files]  # group by filename

# ----------------- Group split by image -----------------
gss = GroupShuffleSplit(test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(np.zeros(len(all_files)), all_labels, groups=all_groups))

train_files = [all_files[i]  for i in train_idx]
train_lbls  = [all_labels[i] for i in train_idx]
test_files  = [all_files[i]  for i in test_idx]
test_lbls   = [all_labels[i] for i in test_idx]

print(f"Train images: {len(train_files)} | Test images: {len(test_files)}")

# ----------------- TF image helpers -----------------
@tf.function
def decode_image(path):
    img_bytes = tf.io.read_file(path)
    # try png, then jpeg
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None,None,3])
    img = tf.image.resize(img, [RESIZE, RESIZE], method='bilinear')
    img = tf.cast(img, tf.float32)
    return img

def extract_patches(img):
    # img: [H,W,3]
    k = PATCH
    s = STRIDE
    img4 = tf.expand_dims(img, 0)  # [1,H,W,3]
    patches = tf.image.extract_patches(
        images=img4,
        sizes=[1, k, k, 1],
        strides=[1, s, s, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )  # [1, nH, nW, k*k*3]
    patches = tf.reshape(patches, [-1, k, k, 3])  # [N, k, k, 3]
    return patches

def preprocess_patch(p):
    # MobileNetV2 expects preprocess_input
    return preprocess_input(p)

# simple augmentations
def augment_patch(p):
    p = tf.image.random_flip_left_right(p)
    p = tf.image.random_brightness(p, max_delta=0.15)
    p = tf.image.random_contrast(p, 0.85, 1.15)
    return p

# ----------------- Build streaming datasets -----------------
def image_to_patch_ds(path, label, training=False):
    img = decode_image(path)
    patches = extract_patches(img)                   # [N, PATCH, PATCH, 3]
    if training:
        patches = tf.map_fn(augment_patch, patches, fn_output_signature=tf.float32)
    patches = tf.map_fn(preprocess_patch, patches, fn_output_signature=tf.float32)
    labels = tf.fill([tf.shape(patches)[0]], tf.cast(label, tf.int32))  # all patches inherit image label
    return tf.data.Dataset.from_tensor_slices((patches, labels))

def files_to_patch_ds(files, labels, training=False):
    # dataset of file paths + labels
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    # for each file, expand to its patches dataset
    ds = ds.interleave(
        lambda p,l: image_to_patch_ds(p, l, training=training),
        cycle_length=args.num_parallel_calls,
        num_parallel_calls=args.num_parallel_calls,
        deterministic=False if training else True
    )
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = files_to_patch_ds(train_files, train_lbls, training=True)
val_ds   = files_to_patch_ds(test_files,  test_lbls,  training=False)

# ----------------- Build model -----------------
weights_choice = None if args.random_init else 'imagenet'
base = MobileNetV2(input_shape=(PATCH, PATCH, 3), include_top=False, weights=weights_choice)
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_auc', mode='max')

# ----------------- Train: warmup -----------------
print("== Warmup training ==")
model.fit(train_ds, epochs=WARMUP_EPOCHS, validation_data=val_ds, callbacks=[early], verbose=1)

# ----------------- Finetune top layers -----------------
print("== Fine-tuning top layers ==")
base.trainable = True
for layer in base.layers[:-TOP_LAYERS_TO_TRAIN]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-5), loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.fit(train_ds, epochs=FINETUNE_EPOCHS, validation_data=val_ds, callbacks=[early], verbose=1)

# ----------------- Patch-level eval (quick) -----------------
# Compute patch metrics on val set
y_true_patch = []
y_prob_patch = []
for xb, yb in val_ds:
    pb = model.predict(xb, verbose=0).ravel()
    y_prob_patch.append(pb)
    y_true_patch.append(yb.numpy())
y_prob_patch = np.concatenate(y_prob_patch)
y_true_patch = np.concatenate(y_true_patch)
y_pred_patch = (y_prob_patch > 0.5).astype(int)
print("\n--- Patch-level (0.50 threshold) ---")
print(classification_report(y_true_patch, y_pred_patch, target_names=['Clear', 'Acne']))

# ----------------- Image-level eval (max over patches) -----------------
def predict_image_prob(path):
    img = decode_image(path)
    patches = extract_patches(img)
    patches = tf.map_fn(preprocess_patch, patches, fn_output_signature=tf.float32)
    probs = model.predict(patches, verbose=0).ravel()
    return float(np.max(probs)) if probs.size else 0.0

test_image_probs = [predict_image_prob(p) for p in test_files]
y_true_img = np.array(test_lbls, dtype=int)
img_probs = np.array(test_image_probs)

prec, rec, thr = precision_recall_curve(y_true_img, img_probs)
f1s = [0.0 if (p+r)==0 else 2*p*r/(p+r) for p, r in zip(prec, rec)]
best_idx = int(np.argmax(f1s))
best_thr = float(np.append(thr, 1.0)[best_idx])
print(f"\nChosen image-level threshold by max F1: {best_thr:.4f} (F1={f1s[best_idx]:.4f}, P={prec[best_idx]:.4f}, R={rec[best_idx]:.4f})")

img_preds = (img_probs > best_thr).astype(int)
print("\n--- Image-level (optimal threshold) ---")
print(classification_report(y_true_img, img_preds, target_names=['Clear', 'Acne']))

# ----------------- Save -----------------
model.save(MODEL_OUT)
with open(THRESH_FILE, "w") as f:
    json.dump({"best_threshold": float(best_thr)}, f)
print(f"\n✅ Saved model to {MODEL_OUT}")
print(f"💾 Saved threshold to {THRESH_FILE}: {best_thr:.4f}")
