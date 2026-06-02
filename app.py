# app.py — GetGlowMind (Streamlit)
# Works with your CURRENT predict.py (that exposes predict_image + helper fns, but NOT predict_image_details):
# - NO "from predict import ..." (prevents ImportError)
# - Imports predict as a module, then runs an internal predict_image_details_local() wrapper
#   that reuses predict.py constants + helper functions and returns a dict for Streamlit
# - Sidebar hidden for users; acne settings are ENV-driven and logged

import os
import sys
import tempfile
import threading
import hashlib
import logging
import importlib
import inspect
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from deepface import DeepFace
from tensorflow.keras.models import load_model as keras_load_model

# OpenCV needed for patching + debug overlays (same dependency as predict.py)
try:
    import cv2
except Exception as e:
    st.error(f"OpenCV (cv2) is required but could not be imported: {e}")
    st.stop()

try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
except Exception as e:
    st.error(f"TensorFlow preprocess_input import failed: {e}")
    st.stop()


# -------------------- Logging: to terminal + file --------------------
BASE_DIR = Path(__file__).resolve().parent
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "app.log", mode="a"),
    ],
)
logger = logging.getLogger("GetGlowMind")


# -------------------- Streamlit config --------------------
st.set_page_config(page_title="GetGlowMind", layout="centered", initial_sidebar_state="collapsed")

# Hide sidebar for users (still exists internally; we just hide it visually)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarNav"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧴 GetGlowMind – AI Skincare Advisor")
st.write(
    "Upload a face photo. We’ll estimate your age, gender, race and mood, "
    "check for acne, then tailor product picks with links."
)

uploaded_file = st.file_uploader("Upload your face photo", type=["jpg", "png", "jpeg"])


# -------------------- Load predict.py safely --------------------
try:
    predict = importlib.import_module("predict")
except Exception as e:
    st.error(f"Could not import local predict.py: {e}")
    st.stop()

logger.info("predict module loaded from: %s", getattr(predict, "__file__", "unknown"))

MODEL_PATH_RAW = getattr(predict, "MODEL_PATH", None)
if not MODEL_PATH_RAW:
    st.error("predict.py loaded, but MODEL_PATH is missing. Ensure predict.py defines MODEL_PATH.")
    st.stop()

MODEL_PATH = Path(MODEL_PATH_RAW)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH
MODEL_PATH = str(MODEL_PATH)
predict.MODEL_PATH = MODEL_PATH

THRESHOLD_FILE_RAW = getattr(predict, "THRESHOLD_FILE", None)
if THRESHOLD_FILE_RAW:
    threshold_path = Path(THRESHOLD_FILE_RAW)
    if not threshold_path.is_absolute():
        threshold_path = BASE_DIR / threshold_path
    predict.THRESHOLD_FILE = str(threshold_path)

load_threshold = getattr(predict, "load_threshold", None)
if not callable(load_threshold):
    # fallback
    def load_threshold(default: float = 0.5) -> float:
        return default

logger.info("MODEL_PATH from predict.py: %s", MODEL_PATH)

if not Path(MODEL_PATH).exists():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Make sure acne_mobilenet_patch_model.h5 is committed to the deployed repository "
        "or stored with Git LFS."
    )
    st.stop()

# Pull constants (must match training)
PATCH_SIZE = getattr(predict, "PATCH_SIZE", (224, 224))  # (w, h)
STRIDE = getattr(predict, "STRIDE", 112)
RESIZE_FOR_PATCHING = getattr(predict, "RESIZE_FOR_PATCHING", (672, 672))  # (w, h)


# -------------------- GPU memory growth (optional) --------------------
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logger.info("TF GPUs: %s", gpus)
except Exception as e:
    logger.info("TF GPU setup skipped: %s", e)


# -------------------- Cache heavy models/resources --------------------
@st.cache_resource(show_spinner=False)
def load_acne_model_and_threshold():
    model = keras_load_model(MODEL_PATH)
    thr = float(load_threshold())

    try:
        with open(MODEL_PATH, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        digest = "unknown"

    logger.info("Loaded acne model: %s (sha256=%s), threshold=%.4f", MODEL_PATH, digest, thr)
    return model, thr, digest


@st.cache_resource(show_spinner=False)
def warmup_deepface():
    logger.info("Warming up DeepFace backbone…")
    return DeepFace.build_model("VGG-Face")


acne_model, default_thr, model_hash = load_acne_model_and_threshold()

face_lock = threading.Lock()


# -------------------- Internal (hidden) acne detection settings --------------------
def _env_bool(name: str, default: bool) -> bool:
    v = (os.getenv(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


# Defaults aligned with your predict.py CLI defaults
ACNE_ROI = os.getenv("ACNE_ROI", "cheeks")          # none/cheeks/forehead/cheeks+forehead
ACNE_AGG = os.getenv("ACNE_AGG", "topk")            # max/topk/mean
ACNE_TOPK = _env_int("ACNE_TOPK", 3)

ACNE_MIN_POS = _env_int("ACNE_MIN_POS", 2)
ACNE_MIN_CLUSTER = _env_int("ACNE_MIN_CLUSTER", 3)

ACNE_SKIN_ONLY = _env_bool("ACNE_SKIN_ONLY", True)
ACNE_MIN_SKIN = _env_float("ACNE_MIN_SKIN", 0.50)

ACNE_CENTER_FRAC = _env_float("ACNE_CENTER_FRAC", 0.90)

ACNE_TTA = _env_bool("ACNE_TTA", True)

ACNE_SAVE_DEBUG = _env_bool("ACNE_SAVE_DEBUG", False)
ACNE_SHOW_METRICS = _env_bool("ACNE_SHOW_METRICS", False)

# Match predict.py CLI default. Set ACNE_THRESHOLD_OVERRIDE="" to use best_threshold.json instead.
ACNE_THRESHOLD_OVERRIDE = os.getenv("ACNE_THRESHOLD_OVERRIDE", "0.85").strip()
ACNE_THRESHOLD = None if ACNE_THRESHOLD_OVERRIDE == "" else float(ACNE_THRESHOLD_OVERRIDE)

logger.info(
    "Acne settings (hidden) → roi=%s agg=%s topk=%s skin_only=%s min_skin=%.2f center_frac=%.2f "
    "min_pos=%s min_cluster=%s tta=%s thr_override=%s model_sha=%s default_thr=%.4f",
    ACNE_ROI, ACNE_AGG, ACNE_TOPK,
    ACNE_SKIN_ONLY, ACNE_MIN_SKIN, ACNE_CENTER_FRAC,
    ACNE_MIN_POS, ACNE_MIN_CLUSTER,
    ACNE_TTA,
    (ACNE_THRESHOLD if ACNE_THRESHOLD is not None else "None"),
    model_hash,
    default_thr,
)


# -------------------- Helpers --------------------
def st_image_compat(img_or_path, caption=None):
    # Prefer newer Streamlit
    try:
        st.image(img_or_path, caption=caption, use_container_width=True)
        return
    except TypeError:
        pass
    # Oldest fallback
    st.image(img_or_path, caption=caption, use_column_width=True)


def age_group_from_age(age: int) -> str:
    if age < 25:
        return "young"
    if age <= 40:
        return "adult"
    return "mature"


def normalize_gender(g: str) -> str:
    g = (g or "").strip().lower()
    if g in ("man", "male"):
        return "Male"
    if g in ("woman", "female"):
        return "Female"
    return "Any"


def normalize_race(r: str) -> str:
    r = (r or "").strip().lower()
    if "black" in r:
        return "Black"
    if "asian" in r:
        return "Asian"
    return "Any"


def normalize_mood(emotion_raw: str) -> str:
    e = (emotion_raw or "").strip().lower()
    mapping = {
        "sad": "Sad",
        "angry": "Stressed",
        "fear": "Anxious",
        "disgust": "Anxious",
        "happy": "Happy",
        "surprise": "Happy",
        "neutral": "Neutral",
        "tired": "Tired",
        "fatigue": "Tired",
    }
    return mapping.get(e, "Neutral")


def dedupe_products(products):
    seen = set()
    out = []
    for name, link in products:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append((name, link))
    return out


def build_recommendation(age_group: str, gender: str, race: str, mood: str, has_acne: bool):
    products = []
    tips = []
    sources = set()
    skin_condition = ""

    if has_acne:
        if age_group == "young":
            products += [
                ("CeraVe Renewing SA Cleanser", "https://www.amazon.com/dp/B00U1YCRD8"),
                ("Differin Adapalene Gel 0.1%", "https://www.amazon.com/dp/B07L1PHSY9"),
                ("Oil-free gel moisturizer", "https://www.amazon.com/dp/B07RGKMKZQ"),
                ("Non-comedogenic SPF 30+", "https://www.amazon.com/dp/B00F97FHAW"),
            ]
        elif age_group == "adult":
            products += [
                ("Niacinamide serum", "https://www.amazon.com/dp/B07PV2K9T4"),
                ("Adapalene 0.1% (PM)", "https://www.amazon.com/dp/B07L1PHSY9"),
                ("Non-comedogenic moisturizer", "https://www.amazon.com/dp/B01N7T7JKJ"),
                ("SPF 30+ broad spectrum", "https://www.amazon.com/dp/B00F97FHAW"),
            ]
        else:
            products += [
                ("Gentle low-pH cleanser", "https://www.amazon.com/dp/B01N2T9M01"),
                ("Retinal/retinoid (low-irritation, PM)", "https://www.amazon.com/dp/B07XQJGC8Y"),
                ("Ceramide/peptide moisturizer", "https://www.amazon.com/dp/B00TTD9BRC"),
                ("SPF 30+ with strong UVA coverage", "https://www.amazon.com/dp/B00F97FHAW"),
            ]

        if mood == "Tired":
            skin_condition = "Fatigue-related Acne"
            sources.add("Health.com")
            tips += ["Ensure 7–9 hours sleep", "Cut late caffeine/alcohol", "Add zinc-rich foods"]
        elif mood in ("Stressed", "Anxious"):
            skin_condition = "Stress-induced Acne & Sensitivity"
            sources.add("AAD")
            tips += ["10 min mindfulness daily", "Limit high-glycemic foods", "Hydrate for barrier support"]
        else:
            skin_condition = "Acne-prone Skin"
            tips += ["Introduce actives slowly", "Cleanse → treat → moisturize → SPF"]

    else:
        if age_group == "young":
            products += [
                ("Gentle hydrating cleanser", "https://www.amazon.com/dp/B01MSSDEPK"),
                ("Light gel-cream moisturizer", "https://www.amazon.com/dp/B00NR1YQHM"),
                ("Broad-spectrum SPF 30+", "https://www.amazon.com/dp/B00F97FHAW"),
            ]
        elif age_group == "adult":
            products += [
                ("Vitamin C (AM)", "https://www.amazon.com/dp/B01M4MCUAF"),
                ("Lightweight moisturizer", "https://www.amazon.com/dp/B00NR1YQHM"),
                ("SPF 30+; optional gentle retinoid (PM)", "https://www.amazon.com/dp/B07XQJGC8Y"),
            ]
        else:
            products += [
                ("Vitamin C/antioxidant (AM)", "https://www.amazon.com/dp/B01M4MCUAF"),
                ("Retinoid/retinal (PM)", "https://www.amazon.com/dp/B07XQJGC8Y"),
                ("Ceramide-rich moisturizer", "https://www.amazon.com/dp/B00TTD9BRC"),
                ("SPF 30+ with UVA/PA rating", "https://www.amazon.com/dp/B00F97FHAW"),
            ]

        if mood == "Happy":
            skin_condition = "Healthy & Glowing Skin"
            tips += ["Balanced diet", "Daily movement", "Hydrate well"]
            sources.add("General Advice")
        elif mood == "Neutral":
            skin_condition = "Balanced Skin"
            tips += ["Be consistent", "Daily sunscreen", "Avoid smoking/excess alcohol"]
            sources.add("General Advice")
        else:
            skin_condition = "Well-maintained Skin"
            tips += ["Stay consistent; gentle weekly exfoliation can help"]

    products = dedupe_products(products)
    return {
        "skin_condition": skin_condition,
        "products": products,
        "lifestyle_changes": tips,
        "sources": sorted(list(sources)) if sources else ["General Advice"],
    }


# -------------------- Acne predictor wrapper (returns dict for Streamlit) --------------------
def predict_image_details_local(model, image_path: str, *,
                               threshold=None, agg="topk", topk=3,
                               min_pos=2, min_cluster=3,
                               roi="cheeks", skin_only=True,
                               min_skin=0.50, center_frac=0.90,
                               tta=True, save_debug: str = "") -> dict:
    """
    Reuses YOUR predict.py helpers + constants, but returns a structured dict
    (so Streamlit can display + log metrics).
    Works even if predict.py has no predict_image_details().
    """

    # Required helpers from predict.py
    needed = ["extract_patches_rgb", "skin_mask_hsv_ycrcb", "make_center_mask",
              "roi_mask_from_face", "tta_variants", "aggregate_probs",
              "largest_cluster_size", "load_threshold"]
    missing = [n for n in needed if not callable(getattr(predict, n, None))]
    if missing:
        return {"error": f"predict.py is missing required helper(s): {missing}"}

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return {"error": "Could not load image with cv2.imread()."}

    # Match predict.py behavior: resize to RESIZE_FOR_PATCHING before patching
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, RESIZE_FOR_PATCHING)
    img_bgr_resized = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    H, W, _ = img_rgb.shape
    ph, pw = PATCH_SIZE[1], PATCH_SIZE[0]

    # Compose global mask
    masks = []

    if roi in ("cheeks", "forehead", "cheeks+forehead"):
        roi_mode = roi if roi != "cheeks+forehead" else "cheeksforehead"  # contains both substrings
        masks.append(predict.roi_mask_from_face(img_rgb, roi_mode=roi_mode, center_frac=center_frac))

    if skin_only:
        masks.append(predict.skin_mask_hsv_ycrcb(img_rgb))

    if center_frac and roi == "none":
        masks.append(predict.make_center_mask(W, H, center_frac))

    global_mask = None
    if masks:
        global_mask = masks[0].astype(np.uint8)
        for m in masks[1:]:
            global_mask = cv2.bitwise_and(global_mask, m.astype(np.uint8))

    # Extract patches
    patches, coords = predict.extract_patches_rgb(img_rgb)

    keep_idx = np.arange(len(patches))
    cov = None
    if global_mask is not None:
        cov = []
        for (x, y) in coords:
            cov.append(global_mask[y:y+ph, x:x+pw].mean())
        cov = np.array(cov, dtype=np.float32)

        # Match predict.py logic
        min_cov = float(min_skin if skin_only else 0.35)
        keep_idx = keep_idx[cov >= min_cov]

    if keep_idx.size == 0:
        thr_val = float(predict.load_threshold()) if threshold is None else float(threshold)
        return {
            "label": 0,
            "threshold": thr_val,
            "image_prob": 0.0,
            "probs_max": 0.0,
            "probs_mean": 0.0,
            "topk_prob": 0.0,
            "num_over": 0,
            "largest_cluster": 0,
            "patches_total": int(len(patches)),
            "patches_kept": 0,
            "agg": agg,
            "topk": int(topk),
            "roi": roi,
            "skin_only": bool(skin_only),
            "min_skin": float(min_skin),
            "center_frac": float(center_frac),
            "tta": bool(tta),
            "debug_path": "",
            "note": "No patches passed ROI/skin/center filters → defaulting to CLEAR",
        }

    patches = [patches[i] for i in keep_idx]
    coords = [coords[i] for i in keep_idx]

    def predict_batch(patches_list):
        arr = preprocess_input(np.array(patches_list, dtype=np.float32))
        return model.predict(arr, verbose=0).ravel()

    if tta:
        all_probs = []
        for v in predict.tta_variants(img_rgb):
            v_patches = [v[y:y+ph, x:x+pw] for (x, y) in coords]
            all_probs.append(predict_batch(v_patches))
        probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    else:
        probs = predict_batch(patches)

    thr_val = float(predict.load_threshold()) if threshold is None else float(threshold)
    img_prob = float(predict.aggregate_probs(probs, agg, topk))
    num_over = int((probs > thr_val).sum())

    grid_w = (W - pw) // STRIDE + 1
    grid_h = (H - ph) // STRIDE + 1
    cluster = int(predict.largest_cluster_size(coords, probs, thr_val, grid_w, grid_h))

    label = int(img_prob > thr_val and num_over >= int(min_pos) and cluster >= int(min_cluster))

    debug_path = ""
    if save_debug and probs.size:
        out = img_bgr_resized.copy()
        for i, p in enumerate(probs):
            if p > thr_val:
                x, y = coords[i]
                cv2.rectangle(out, (x, y), (x + pw, y + ph), (0, 0, 255), 2)
                cv2.putText(out, f"{p:.2f}", (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        try:
            cv2.imwrite(save_debug, out)
            debug_path = save_debug
        except Exception as e:
            logger.warning("Failed to save debug overlay: %s", e)

    return {
        "label": label,
        "threshold": thr_val,
        "image_prob": float(img_prob),
        "probs_max": float(np.max(probs)) if probs.size else 0.0,
        "probs_mean": float(np.mean(probs)) if probs.size else 0.0,
        "topk_prob": float(predict.aggregate_probs(probs, "topk", topk)) if probs.size else 0.0,
        "num_over": int(num_over),
        "largest_cluster": int(cluster),
        "patches_total": int(len(patches)),
        "patches_kept": int(len(probs)),
        "agg": agg,
        "topk": int(topk),
        "roi": roi,
        "skin_only": bool(skin_only),
        "min_skin": float(min_skin),
        "center_frac": float(center_frac),
        "tta": bool(tta),
        "debug_path": debug_path,
    }


def call_acne_predictor(model, image_path: str, save_debug: str):
    """
    If predict.py ever adds predict_image_details in the future, we’ll use it.
    Otherwise we use our local wrapper (current working path).
    """
    fn = getattr(predict, "predict_image_details", None)
    if callable(fn):
        # Try to be signature-flexible
        try:
            sig = inspect.signature(fn)
            params = set(sig.parameters.keys())
            kwargs = {}
            if "threshold" in params: kwargs["threshold"] = ACNE_THRESHOLD
            if "agg" in params: kwargs["agg"] = ACNE_AGG
            if "topk" in params: kwargs["topk"] = int(ACNE_TOPK)
            if "min_pos" in params: kwargs["min_pos"] = int(ACNE_MIN_POS)
            if "min_cluster" in params: kwargs["min_cluster"] = int(ACNE_MIN_CLUSTER)
            if "roi" in params: kwargs["roi"] = ACNE_ROI
            if "skin_only" in params: kwargs["skin_only"] = bool(ACNE_SKIN_ONLY)
            if "min_skin" in params: kwargs["min_skin"] = float(ACNE_MIN_SKIN)
            if "center_frac" in params: kwargs["center_frac"] = float(ACNE_CENTER_FRAC)
            if "tta" in params: kwargs["tta"] = bool(ACNE_TTA)
            if "save_debug" in params: kwargs["save_debug"] = save_debug
            return fn(model, image_path, **kwargs)
        except Exception as e:
            logger.warning("predict.predict_image_details exists but failed (%s). Falling back to local wrapper.", e)

    # Current stable path (your provided predict.py)
    return predict_image_details_local(
        model, image_path,
        threshold=ACNE_THRESHOLD,
        agg=ACNE_AGG,
        topk=int(ACNE_TOPK),
        min_pos=int(ACNE_MIN_POS),
        min_cluster=int(ACNE_MIN_CLUSTER),
        roi=ACNE_ROI,
        skin_only=bool(ACNE_SKIN_ONLY),
        min_skin=float(ACNE_MIN_SKIN),
        center_frac=float(ACNE_CENTER_FRAC),
        tta=bool(ACNE_TTA),
        save_debug=save_debug,
    )


# -------------------- Main flow --------------------
if uploaded_file:
    logger.info(
        "Uploaded file: name=%s, type=%s, size=%s bytes",
        uploaded_file.name,
        uploaded_file.type,
        uploaded_file.size,
    )

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st_image_compat(image, caption="Uploaded image")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
        image.save(tmp_path, format="JPEG", quality=95)
    logger.info("Temp image saved to %s", tmp_path)

    # 1) DeepFace analysis
    with st.spinner("Analyzing face (age, gender, race, mood)…"):
        _ = warmup_deepface()
        arr = np.array(image)
        with face_lock:
            analysis = DeepFace.analyze(arr, actions=["age", "gender", "emotion", "race"], enforce_detection=False)
        a0 = analysis[0] if isinstance(analysis, list) else analysis

        age_val = int(round(a0.get("age", 0) or 0))
        gender_raw = a0.get("dominant_gender", "") or ""
        race_raw = a0.get("dominant_race", "") or ""
        emotion_raw = a0.get("dominant_emotion", "") or ""

        age_group = age_group_from_age(age_val)
        gender = normalize_gender(gender_raw)
        race = normalize_race(race_raw)
        mood = normalize_mood(emotion_raw)

        logger.info(
            "DeepFace → age=%s (group=%s), gender=%s → %s, race=%s → %s, mood=%s → %s",
            age_val, age_group, gender_raw, gender, race_raw, race, emotion_raw, mood,
        )

    st.info(f"Age: **{age_val}**  •  Gender: **{gender}**  •  Race: **{race}**  •  Mood: **{mood}**")

    # 2) Acne detection
    debug_path = ""
    if ACNE_SAVE_DEBUG:
        debug_path = tmp_path.replace(".jpg", "_debug.jpg")

    with st.spinner("Checking for acne…"):
        result = call_acne_predictor(acne_model, tmp_path, save_debug=debug_path)

    if isinstance(result, dict) and "error" in result:
        logger.error("Acne detection error: %s", result["error"])
        st.error(result["error"])
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        st.stop()

    # Normalize return (in case future predict_image_details returns something else)
    if not isinstance(result, dict):
        result = {"error": f"Unexpected acne predictor return type: {type(result)}"}
        st.error(result["error"])
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        st.stop()

    logger.info(
        "Acne result → label=%s thr=%.4f image_prob=%.4f probs_max=%.4f probs_mean=%.4f num_over=%s cluster=%s "
        "agg=%s topk=%s roi=%s skin_only=%s min_skin=%.2f center_frac=%.2f tta=%s debug=%s",
        result.get("label"),
        float(result.get("threshold", default_thr)),
        float(result.get("image_prob", 0.0)),
        float(result.get("probs_max", 0.0)),
        float(result.get("probs_mean", 0.0)),
        result.get("num_over", 0),
        result.get("largest_cluster", 0),
        result.get("agg", ACNE_AGG),
        result.get("topk", ACNE_TOPK),
        ACNE_ROI,
        ACNE_SKIN_ONLY,
        ACNE_MIN_SKIN,
        ACNE_CENTER_FRAC,
        ACNE_TTA,
        result.get("debug_path", ""),
    )

    has_acne = (result.get("label") == 1)
    st.success(f"Skin status: **{'Acne' if has_acne else 'Clear'}**")

    if ACNE_SAVE_DEBUG and result.get("debug_path"):
        st_image_compat(result["debug_path"], caption="Debug overlay (patches over threshold)")

    if ACNE_SHOW_METRICS:
        with st.expander("Detection metrics"):
            st.json(result)

    # 3) Recommendations
    rec = build_recommendation(age_group, gender, race, mood, has_acne)

    st.markdown(f"### 💡 Recommended Products — *{rec['skin_condition']}*")
    for name, link in rec["products"]:
        st.markdown(f"- [{name}]({link})")

    st.markdown("### 🌿 Lifestyle Tips")
    for tip in rec["lifestyle_changes"]:
        st.markdown(f"- {tip}")

    st.caption(f"Sources: {', '.join(rec['sources'])}")

    try:
        os.remove(tmp_path)
        logger.info("Removed temp file %s", tmp_path)
    except Exception as e:
        logger.warning("Temp cleanup failed: %s", e)

st.markdown("---")
st.caption("© 2024 Jaroslav Sidor. All rights reserved.")
