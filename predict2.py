# predict.py — ROI (cheeks/forehead), skin mask, cluster gating to reduce FPs
import os, sys, json, argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===== Must match training =====
PATCH_SIZE = (224, 224)           # (w, h)
STRIDE = 112
RESIZE_FOR_PATCHING = (672, 672)  # (w, h)
MODEL_PATH = "acne_mobilenet_patch_model.h5"
THRESHOLD_FILE = "best_threshold.json"
THRESHOLD_FALLBACK = 0.50
# =================================

def load_threshold(default=THRESHOLD_FALLBACK):
    if os.path.exists(THRESHOLD_FILE):
        try:
            with open(THRESHOLD_FILE, "r") as f:
                return float(json.load(f).get("best_threshold", default))
        except Exception:
            pass
    return default

def extract_patches_rgb(img_rgb):
    h, w, _ = img_rgb.shape
    patches, coords = [], []
    ph, pw = PATCH_SIZE[1], PATCH_SIZE[0]
    for y in range(0, h - ph + 1, STRIDE):
        for x in range(0, w - pw + 1, STRIDE):
            patches.append(img_rgb[y:y+ph, x:x+pw])
            coords.append((x, y))
    return patches, coords  # coords are top-left patch origins

def skin_mask_hsv_ycrcb(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_hsv = cv2.inRange(hsv, np.array([0, 40, 60], np.uint8),
                                np.array([25, 255, 255], np.uint8))
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], np.uint8),
                                    np.array([255, 173, 127], np.uint8))
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    return (mask > 0).astype(np.uint8)  # HxW {0,1}

def make_center_mask(width, height, frac=0.85):
    cw, ch = int(width*frac), int(height*frac)
    x0, y0 = (width - cw)//2, (height - ch)//2
    m = np.zeros((height, width), dtype=np.uint8)
    m[y0:y0+ch, x0:x0+cw] = 1
    return m  # HxW

def detect_face_bbox(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60,60))
    if len(faces) == 0:
        return None
    # pick the largest face
    x,y,w,h = max(faces, key=lambda r:r[2]*r[3])
    return (x,y,w,h)

def roi_mask_from_face(img_rgb, roi_mode="cheeks", center_frac=0.85):
    """Return HxW binary mask for allowed ROI based on detected face."""
    H, W, _ = img_rgb.shape
    face = detect_face_bbox(img_rgb)
    if face is None:
        # fallback to central area if no face found
        return make_center_mask(W, H, center_frac)

    x, y, w, h = face
    mask = np.zeros((H, W), dtype=np.uint8)

    # heuristic regions relative to face bbox
    if "forehead" in roi_mode:
        fh_top = int(y + 0.10*h)
        fh_bot = int(y + 0.35*h)
        fh_x0  = int(x + 0.15*w)
        fh_x1  = int(x + 0.85*w)
        mask[fh_top:fh_bot, fh_x0:fh_x1] = 1

    if "cheeks" in roi_mode:
        cy_top = int(y + 0.35*h)
        cy_bot = int(y + 0.75*h)
        # left cheek
        lc_x0 = int(x + 0.05*w)
        lc_x1 = int(x + 0.45*w)
        # right cheek
        rc_x0 = int(x + 0.55*w)
        rc_x1 = int(x + 0.95*w)
        mask[cy_top:cy_bot, lc_x0:lc_x1] = 1
        mask[cy_top:cy_bot, rc_x0:rc_x1] = 1

    return mask

def tta_variants(img_rgb):
    return [
        img_rgb,
        cv2.flip(img_rgb, 1),
        np.clip(img_rgb*1.15, 0, 255).astype(np.uint8),
        np.clip(img_rgb*0.85, 0, 255).astype(np.uint8),
    ]

def aggregate_probs(probs, mode="max", topk=3):
    probs = np.asarray(probs)
    if probs.size == 0: return 0.0
    if mode == "max":  return float(np.max(probs))
    if mode == "mean": return float(np.mean(probs))
    if mode == "topk":
        k = min(topk, probs.size)
        return float(np.mean(np.sort(probs)[-k:]))
    return float(np.max(probs))

def largest_cluster_size(coords, probs, thr, grid_w, grid_h):
    """Coords are top-left pixel coords; map them to patch grid and compute largest 4-neighbor cluster."""
    if len(coords) == 0: return 0
    idx_over = [i for i,p in enumerate(probs) if p > thr]
    if not idx_over: return 0
    xs = sorted(set([x for x,_ in coords]))
    ys = sorted(set([y for _,y in coords]))
    x_to_col = {x:i for i,x in enumerate(xs)}
    y_to_row = {y:i for i,y in enumerate(ys)}
    over_cells = set((y_to_row[coords[i][1]], x_to_col[coords[i][0]]) for i in idx_over)
    visited = set(); best = 0
    for cell in over_cells:
        if cell in visited: continue
        stack = [cell]; visited.add(cell); size = 0
        while stack:
            r,c = stack.pop(); size += 1
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nb = (r+dr, c+dc)
                if nb in over_cells and nb not in visited:
                    visited.add(nb); stack.append(nb)
        best = max(best, size)
    return best

def predict_image(model, image_path, threshold=None, agg="max", topk=3,
                  min_pos=1, min_cluster=1, roi="none",
                  skin_only=False, min_skin=0.35, center_frac=0.85,
                  tta=False, show=False, save_debug=""):
    # Load & prep
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("❌ Could not load image."); return 1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, RESIZE_FOR_PATCHING)
    H, W, _ = img_rgb.shape

    # Global mask composition
    masks = []

    if roi in ("cheeks", "forehead", "cheeks+forehead"):
        roi_m = roi_mask_from_face(img_rgb, roi_mode=roi if roi!="cheeks+forehead" else "cheeksforehead",
                                   center_frac=center_frac)
        masks.append(roi_m)

    if skin_only:
        masks.append(skin_mask_hsv_ycrcb(img_rgb))

    if center_frac and roi == "none":  # if ROI already used, center mask is likely redundant
        masks.append(make_center_mask(W, H, center_frac))

    global_mask = None
    if masks:
        global_mask = masks[0].astype(np.uint8)
        for m in masks[1:]:
            global_mask = cv2.bitwise_and(global_mask, m.astype(np.uint8))

    # Extract patches & filter by masks
    patches, coords = extract_patches_rgb(img_rgb)

    keep_idx = np.arange(len(patches))
    if global_mask is not None:
        ph, pw = PATCH_SIZE[1], PATCH_SIZE[0]
        cov = []
        for (x,y) in coords:
            cov.append(global_mask[y:y+ph, x:x+pw].mean())
        cov = np.array(cov, dtype=np.float32)
        min_cov = float(min_skin if skin_only else 0.35)
        keep_idx = keep_idx[cov >= min_cov]

    if keep_idx.size == 0:
        thr = load_threshold() if threshold is None else threshold
        print(f"⚠️ No patches passed filters — defaulting to CLEAR at thr={thr:.2f}")
        return 0

    patches = [patches[i] for i in keep_idx]
    coords  = [coords[i]  for i in keep_idx]

    def predict_batch(patches_list):
        arr = preprocess_input(np.array(patches_list, dtype=np.float32))
        return model.predict(arr, verbose=0).ravel()

    if tta:
        all_probs = []
        ph, pw = PATCH_SIZE[1], PATCH_SIZE[0]
        for v in tta_variants(img_rgb):
            v_patches = [v[y:y+ph, x:x+pw] for (x,y) in coords]
            all_probs.append(predict_batch(v_patches))
        probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    else:
        probs = predict_batch(patches)

    thr = load_threshold() if threshold is None else threshold
    img_prob = aggregate_probs(probs, agg, topk)
    num_over = int((probs > thr).sum())

    # cluster gating
    grid_w = (W - PATCH_SIZE[0]) // STRIDE + 1
    grid_h = (H - PATCH_SIZE[1]) // STRIDE + 1
    cluster = largest_cluster_size(coords, probs, thr, grid_w, grid_h)

    label = int(img_prob > thr and num_over >= min_pos and cluster >= min_cluster)

    print(f"🧪 Image: {os.path.basename(image_path)}")
    print(f"• Patches kept: {len(probs)}  • Agg={agg}  • Max={probs.max():.4f}  "
          f"• TopK({topk})={aggregate_probs(probs,'topk',topk):.4f}  • Mean={probs.mean():.4f}")
    print(f"• Threshold: {thr:.4f}  • Patches over thr: {num_over}  • Largest cluster: {cluster}")
    print(f"➡️ Verdict: {'Acne' if label==1 else 'Clear'}  (image_prob={img_prob:.4f})")

    if show and probs.size:
        idxs = np.argsort(probs)[-3:][::-1]
        ph, pw = PATCH_SIZE[1], PATCH_SIZE[0]
        for rank, i in enumerate(idxs, 1):
            x, y = coords[i]
            patch_bgr = img_rgb[y:y+ph, x:x+pw][:,:,::-1]
            cv2.imshow(f"Top{rank} prob={probs[i]:.3f} at ({x},{y})", patch_bgr)
            cv2.waitKey(0); cv2.destroyAllWindows()

    if save_debug and probs.size:
        out = img_bgr.copy()
        ph, pw = PATCH_SIZE[1], PATCH_SIZE[0]
        for i, p in enumerate(probs):
            if p > thr:
                x, y = coords[i]
                cv2.rectangle(out, (x,y), (x+pw, y+ph), (0,0,255), 2)
                cv2.putText(out, f"{p:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.imwrite(save_debug, out)
        print(f"🖼️ Saved debug to {save_debug}")

    return 0

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")

    # === Your preferred defaults ===
    ap.add_argument("--th", type=float, default=0.85)
    ap.add_argument("--agg", choices=["max","mean","topk"], default="topk")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--min_pos", type=int, default=2)
    ap.add_argument("--min-cluster", type=int, default=3)
    ap.add_argument("--roi", choices=["none","cheeks","forehead","cheeks+forehead"], default="cheeks")

    # Boolean toggles with default=True but overridable
    ap.add_argument("--skin-only", dest="skin_only", action="store_true")
    ap.add_argument("--no-skin-only", dest="skin_only", action="store_false")
    ap.set_defaults(skin_only=True)

    ap.add_argument("--tta", dest="tta", action="store_true")
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.set_defaults(tta=True)

    # Other FP control defaults
    ap.add_argument("--min-skin", type=float, default=0.5)
    ap.add_argument("--center-frac", type=float, default=0.9)

    # Debug output default
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-debug", default="debug_out.jpg")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.image):
        print("❗File not found:", args.image); sys.exit(1)
    model = load_model(MODEL_PATH)
    rc = predict_image(model, args.image, threshold=args.th, agg=args.agg, topk=args.topk,
                       min_pos=args.min_pos, min_cluster=args.min_cluster, roi=args.roi,
                       skin_only=args.skin_only, min_skin=args.min_skin,
                       center_frac=args.center_frac, tta=args.tta, show=args.show,
                       save_debug=args.save_debug)
    sys.exit(rc)
