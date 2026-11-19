"""CrackSAFE-MRC | Accessible, guided structural-crack research demo.

Stability update: preserves segmentation, calibration, thresholds, and detections;
computes optional GradCAM on a smaller input to reduce peak CPU memory.
Model weights load from the public university Hugging Face repository.
GradCAM is opt-in and runs at 256px to reduce memory pressure on free hosting.
"""
from __future__ import annotations

import hashlib
import io
import json
import gc
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "phase14a_demo_config.json").read_text(encoding="utf-8"))
MODEL_REPO = "mufrat-montasir/CrackSAFE-MRC-Model"
RESEARCH_REPO = "https://github.com/mufrat-montasir/CrackSAFE-MRC"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
EXAMPLES = {
    "Example A — fine crack": "https://raw.githubusercontent.com/mufrat-montasir/CrackSAFE-MRC/main/CrackSAFE_Web_Demo/examples/DeepCrack_11190-2.jpg",
    "Example B — branching crack": "https://raw.githubusercontent.com/mufrat-montasir/CrackSAFE-MRC/main/CrackSAFE_Web_Demo/examples/DeepCrack_11231-9.jpg",
}

st.set_page_config(
    page_title="CrackSAFE-MRC | AI Crack Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Prevent oversubscription on Streamlit Community Cloud CPU.
torch.set_num_threads(1)

st.markdown("""
<style>
/* CrackSAFE-MRC · FOREST ATELIER
   Editorial research interface: muted forest, warm ivory, sage, clay, brass.
   These rules change presentation only. Inference, CAM and session logic remain intact. */
:root {
  --forest: #2d4b38;
  --forest-deep: #203629;
  --forest-soft: #4f6c50;
  --sage: #aebda4;
  --sage-light: #e8eee4;
  --brass: #c5a46d;
  --clay: #ad795e;
  --cream: #f8f6ef;
  --paper: #fffcf6;
  --ink: #28342b;
  --muted: #626e61;
  --line: #dedfd3;
}
html, body, [data-testid="stAppViewContainer"] {
  color: var(--ink);
  font-family: "Aptos", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 38rem 23rem at 95% 7%, rgba(191,159,108,.16) 0%, transparent 78%),
    radial-gradient(ellipse 32rem 23rem at 2% 37%, rgba(156,179,145,.15) 0%, transparent 78%),
    linear-gradient(180deg,#f8f6ef 0%,#fbfaf6 46%,#f6f5ee 100%);
}
[data-testid="stHeader"] {background: rgba(248,246,239,.88);}
[data-testid="stToolbar"] {color:var(--forest-deep);}
.block-container {max-width:1190px; padding-top:1.75rem; padding-bottom:3.4rem;}
h1,h2,h3 {color:var(--ink);letter-spacing:-.035em;}
h2 {font-weight:730;}
p,li {line-height:1.66;}
a {color:#355c3e;}
a:hover {color:#7c573f;}

/* Hero: warm, editorial and visibly different from generic Streamlit dashboards. */
.hero {
  position:relative;
  isolation:isolate;
  overflow:hidden;
  padding:3.05rem 3.25rem 3.1rem;
  border:1px solid #526b50;
  border-radius:24px;
  background:
    radial-gradient(ellipse 26rem 23rem at 99% -18%,rgba(213,181,121,.29),transparent 74%),
    radial-gradient(ellipse 33rem 23rem at 94% 112%,rgba(162,182,141,.24),transparent 75%),
    linear-gradient(115deg,#20372b 0%,#30523b 56%,#647457 100%);
  box-shadow:0 18px 45px rgba(39,59,42,.14),0 2px 10px rgba(39,59,42,.06);
  margin-bottom:1.5rem;
}
.hero::before {
  content:"";
  position:absolute;
  top:0;left:0;right:0;height:4px;
  background:linear-gradient(90deg,#b78d5e,#e4c997 39%,#a3b498 77%,transparent);
  z-index:1;
}
.hero::after {
  content:"";
  position:absolute;
  width:335px;height:335px;
  right:-81px;top:-132px;
  border-radius:50%;
  border:1px solid rgba(246,228,194,.18);
  box-shadow:0 0 0 50px rgba(241,221,183,.037),0 0 0 114px rgba(241,221,183,.025);
  z-index:-1;
  pointer-events:none;
}
.eyebrow {
  display:inline-block;
  border-left:3px solid #d6b476;
  padding:.18rem 0 .18rem .85rem;
  font-weight:780;
  letter-spacing:.165em;
  text-transform:uppercase;
  font-size:.72rem;
  color:#ecd3a5;
}
.hero h1 {
  position:relative;
  max-width:900px;
  font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif;
  font-size:clamp(2.3rem,4.05vw,3.55rem);
  font-weight:700;
  line-height:1.13;
  letter-spacing:-.042em;
  color:#fff9eb!important;
  margin:.9rem 0 1rem;
}
.hero p {
  max-width:730px;
  font-size:1.055rem;
  line-height:1.75;
  color:#e6ecdf;
  margin:0;
}
.hero-meta {margin-top:1.55rem;display:flex;gap:.65rem;flex-wrap:wrap;}
.meta-chip {
  display:inline-flex;align-items:center;
  border:1px solid rgba(229,215,179,.36);
  border-radius:99px;
  padding:.39rem .83rem;
  color:#f7ecd7;
  font-size:.76rem;
  letter-spacing:.015em;
  font-weight:650;
  background:rgba(255,255,246,.09);
  backdrop-filter:blur(4px);
}
.meta-chip:first-child {border-color:rgba(232,197,140,.56);background:rgba(198,156,93,.16);}

/* Friendly, deliberate workflow cards. */
.steps {display:flex;gap:.9rem;margin:0 0 1.6rem;flex-wrap:wrap;}
.step {
  display:flex;align-items:center;gap:.68rem;
  background:linear-gradient(125deg,#fffcf6,#f4f1e8);
  border:1px solid #e4decf;
  border-top:3px solid #bfa06e;
  border-radius:13px;
  padding:.73rem 1rem;
  color:#485449;
  box-shadow:0 5px 16px rgba(61,64,46,.045);
  font-size:.86rem;
  font-weight:675;
}
.step:nth-child(2){border-top-color:#8c9e82;}
.step:nth-child(3){border-top-color:#bc866c;}
.step b {
  display:inline-grid;place-items:center;
  width:27px;height:27px;
  border-radius:9px;
  background:#e9e7dc;
  color:#2b4c36;
  font-size:.72rem;font-weight:800;
}
.step:nth-child(3) b{background:#f1e5dc;color:#865842;}
.section-label {
  position:relative;
  font-size:1.24rem;
  color:#263c2e;
  font-weight:800;
  letter-spacing:-.02em;
  margin:0 0 .25rem;
  padding-left:1.05rem;
}
.section-label::before {
  content:"";position:absolute;left:0;top:.2rem;
  width:4px;height:1.13rem;border-radius:4px;
  background:linear-gradient(#bb8c65,#d3b584);
}
.section-sub {color:#687265;font-size:.94rem;margin:0 0 1rem;}

/* The upload workspace should feel like a premium research instrument. */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color:#dddccf!important;
  border-radius:20px!important;
  background:linear-gradient(130deg,#fffefa,#fdfbf5 66%,#f5f3e9)!important;
  box-shadow:0 11px 32px rgba(55,58,37,.065);
}
[data-testid="stFileUploaderDropzone"] {
  border:1px dashed #baa47e!important;
  border-radius:15px!important;
  background:linear-gradient(125deg,#fcfaf4,#f3f1e7)!important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color:#8b9b78!important;
  background:#f3f5eb!important;
}
[data-testid="stFileUploaderDropzone"] button {
  color:#2d4a35!important;
  background:#fffdf8!important;
  border:1px solid #d8d5c8!important;
  border-radius:10px!important;
  font-weight:670!important;
}
[data-testid="stRadio"] [role="radiogroup"] {gap:.75rem;}
[data-testid="stRadio"] label p {color:#374a37;}
[data-testid="stImage"] img {border-radius:13px;box-shadow:0 5px 20px rgba(47,53,36,.09);}
[data-testid="stExpander"] {
  border:1px solid #dfded1!important;
  border-radius:12px!important;
  background:rgba(255,255,251,.83)!important;
}
[data-testid="stExpander"] summary {color:#344a36;font-weight:660;}

/* Strong but understated olive-green action; secondary actions stay warm-white. */
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"] {
  min-height:49px;
  border:1px solid #355540!important;
  border-radius:12px!important;
  background:linear-gradient(112deg,#284832 0%,#42664a 60%,#657b58 100%)!important;
  box-shadow:0 6px 16px rgba(37,67,45,.16);
  color:#fffdf5!important;
  font-weight:730!important;
  letter-spacing:.012em;
  transition:filter .2s ease,transform .2s ease,box-shadow .2s ease;
}
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover {
  filter:brightness(1.07);
  box-shadow:0 8px 19px rgba(37,67,45,.20);
}
.stButton button:focus-visible,.stDownloadButton button:focus-visible {
  outline:3px solid #b59464!important;outline-offset:3px!important;
}
.stButton button:not([kind="primary"]), .stDownloadButton button:not([kind="primary"]) {
  border-radius:10px;
  border-color:#d9d8cb;
  min-height:43px;
  color:#36533b;
  background:#fffdf8;
}
.stButton button:not([kind="primary"]):hover,.stDownloadButton button:not([kind="primary"]):hover {
  border-color:#8d9d83;
  background:#f2f4e9;
  color:#274331;
}
button:disabled {box-shadow:none!important;opacity:.72;}

/* Results: slight earthy variation brings hierarchy without looking loud. */
[data-testid="stMetric"] {
  position:relative;
  overflow:hidden;
  border:1px solid #e1dfd3!important;
  background:linear-gradient(140deg,#fffef9 30%,#f3f4ea 100%);
  border-radius:15px!important;
  padding:17px 18px!important;
  min-height:116px;
  box-shadow:0 7px 18px rgba(52,65,41,.052);
}
[data-testid="stMetric"]:before {
  content:"";position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#bca06d,#8b9f7d 65%,#be8f71);
}
[data-testid="stMetricLabel"] {color:#667466;font-size:.81rem;}
[data-testid="stMetricValue"] {color:#273d2d;font-weight:760;font-size:1.58rem;}
.result-banner {
  border:1px solid #cbd9c4;
  border-left:5px solid #688363;
  background:linear-gradient(105deg,#e9f0e4 0%,#f8f8ef 72%,#fffaf0 100%);
  color:#284632;
  padding:1.25rem 1.55rem;
  border-radius:15px;
  margin:0 0 1.25rem;
  box-shadow:0 6px 20px rgba(57,80,45,.04);
}
.result-banner.no-crack {
  border-color:#e3d8c8;
  border-left-color:#b78c69;
  background:linear-gradient(115deg,#f5eadf,#fdf8ed 80%);
  color:#674b39;
}
.result-banner strong {font-size:1.12rem;}
.result-banner p {margin:.35rem 0 0;font-size:.9rem;color:inherit;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  border-bottom-color:#deded1;
  gap:.4rem;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  color:#5b6758;
  border-radius:10px 10px 0 0;
  padding-left:.75rem;padding-right:.75rem;
  font-weight:650;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
  color:#2f5a3d!important;
  background:#eef1e6;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background-color:#8b7555!important;}
.help-strip {
  padding:1.15rem 1.3rem;
  border:1px solid #d9dfcf;
  border-left:4px solid #b99d6f;
  border-radius:14px;
  background:linear-gradient(116deg,#edf1e7 0%,#f8f5e9 100%);
  color:#485947;
  font-size:.91rem;
  line-height:1.72;
}
.small-overline {
  color:#8b6447;
  font-size:.77rem;
  font-weight:800;
  letter-spacing:.14em;
  text-transform:uppercase;
}
.footer-line {
  padding-top:1.5rem;
  margin-top:2.4rem;
  color:#697466;
  font-size:.82rem;
  line-height:1.8;
  border-top:1px solid #dcdccf;
}
.footer-line a {color:#365d3e;text-decoration:underline;text-decoration-color:#c19f68;text-underline-offset:3px;}
::-webkit-scrollbar {width:10px;height:10px;}
::-webkit-scrollbar-thumb {background:#c8c9b8;border:2px solid #f8f6ef;border-radius:9px;}
@media (max-width:760px) {
  .block-container {padding-top:.75rem;padding-left:1rem;padding-right:1rem;}
  .hero {padding:2rem 1.5rem;border-radius:17px;}
  .hero h1 {font-size:clamp(2rem,7vw,2.6rem);}
  .hero::after {width:240px;height:240px;right:-115px;top:-120px;}
  .hero p {font-size:.97rem;}
  .step {padding:.55rem .72rem;gap:.45rem;font-size:.78rem;}
  .step b {width:24px;height:24px;}
  .steps {gap:.55rem;}
  [data-testid="stMetric"] {min-height:98px;}
}
@media (prefers-reduced-motion:reduce) {
  .stButton button[kind="primary"],.stDownloadButton button[kind="primary"] {transition:none;}
}
</style>
<div class="hero">
  <div class="eyebrow">CrackSAFE-MRC · University Research Demo</div>
  <h1>See the crack. Understand the analysis.</h1>
  <p>Upload a photo of a wall or concrete surface. Explore the detected crack regions,
     segmentation mask and a visual explanation of what the AI model examined.</p>
  <div class="hero-meta">
    <span class="meta-chip">No account needed</span>
    <span class="meta-chip">JPG · PNG</span>
    <span class="meta-chip">AI research demonstration</span>
  </div>
</div>
<div class="steps">
  <div class="step"><b>01</b> Add an image</div>
  <div class="step"><b>02</b> Run analysis</div>
  <div class="step"><b>03</b> Understand and download</div>
</div>
""", unsafe_allow_html=True)

def log_process_memory(stage: str) -> None:
    """Lightweight memory telemetry for Streamlit's cloud logs (Linux only).

    This adds no third-party dependency and is useful if the hosting process
    gets killed for resource usage before Python can emit a traceback.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as stat:
            metrics = {}
            for line in stat:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value = line.split(":", 1)
                    metrics[key] = round(int(value.strip().split()[0]) / 1024)
        print(
            f"[CrackSAFE resource] {stage} | "
            f"RSS={metrics.get('VmRSS', '?')} MiB | "
            f"peak={metrics.get('VmHWM', '?')} MiB",
            flush=True,
        )
    except (OSError, ValueError):
        pass


@st.cache_resource(show_spinner="Downloading and loading the research model...")
def load_model():
    checkpoint = hf_hub_download(repo_id=MODEL_REPO, filename=CFG["checkpoint_path"])
    model = smp.UnetPlusPlus(
        encoder_name=CFG["encoder_name"],
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    # Only load weights from the repository controlled by the research author.
    # weights_only prevents execution of arbitrary pickle code during loading.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        state_dict = next(
            (ckpt[key] for key in ("state_dict", "model_state_dict", "model", "net")
             if key in ckpt and isinstance(ckpt[key], dict)),
            ckpt,
        )
    else:
        state_dict = ckpt
    cleaned = {}
    for key, value in state_dict.items():
        fixed_key = key
        for prefix in ("module.", "model."):
            if fixed_key.startswith(prefix):
                fixed_key = fixed_key[len(prefix):]
        cleaned[fixed_key] = value
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    # Drop the temporary checkpoint and state-dict references after the weights
    # have been copied into the model. Keep only one cached model in RAM.
    del ckpt, state_dict, cleaned
    gc.collect()
    log_process_memory("model loaded")
    preprocessing = smp.encoders.get_preprocessing_fn(CFG["encoder_name"], "imagenet")
    # Cache a single model with a shared lock: concurrent Streamlit sessions
    # must not change its GradCAM hooks/gradients simultaneously.
    return model, preprocessing, threading.RLock()


def preprocess_rgb(rgb, preprocessing):
    resized = cv2.resize(rgb, (int(CFG["image_size"]), int(CFG["image_size"])),
                         interpolation=cv2.INTER_LINEAR)
    array = preprocessing(resized.astype(np.float32))
    array = np.transpose(array, (2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(array).unsqueeze(0)
    return resized, tensor


def clean_binary_mask(mask):
    out = mask.astype(np.uint8).copy()
    if CFG["apply_packaging_close"] and int(CFG["packaging_close_kernel"]) > 1:
        k = np.ones((int(CFG["packaging_close_kernel"]),) * 2, np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    min_area = int(CFG["packaging_min_component_area"])
    if min_area:
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        cleaned = np.zeros_like(out)
        for index in range(1, labels_count):
            if int(stats[index, cv2.CC_STAT_AREA]) >= min_area:
                cleaned[labels == index] = 1
        out = cleaned
    return out


def target_layer(model):
    blocks = getattr(model.encoder, "_blocks", None)
    if blocks is not None and len(blocks) > 0:
        return blocks[-1]
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise RuntimeError("No convolutional layer available for CAM")


class BinarySegmentationTarget:
    def __init__(self, mask):
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __call__(self, output):
        if output.ndim == 4:
            output = output[0, 0]
        else:
            output = output[0]
        return (output * self.mask.to(output.device)).sum()


def build_components(prob, mask):
    labels_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label in range(1, labels_count):
        ys, xs = np.where(labels == label)
        if not len(xs):
            continue
        mask_region = labels == label
        coords = np.column_stack([xs, ys]).astype(np.float32)
        obb = cv2.boxPoints(cv2.minAreaRect(coords)).astype(int).tolist()
        components.append({
            "area_px": int(mask_region.sum()),
            "score_p95": float(np.percentile(prob[mask_region], 95)),
            "aabb": {"x1": int(xs.min()), "y1": int(ys.min()),
                     "x2": int(xs.max()), "y2": int(ys.max())},
            "obb": {"points": obb},
        })
    return components


def make_overlay(image, mask, components):
    green = np.zeros_like(image)
    green[..., 1] = 255
    overlay = np.where(mask[..., None] > 0, image * 0.65 + green * 0.35, image).astype(np.uint8)
    for index, component in enumerate(components, 1):
        aabb = component["aabb"]
        cv2.rectangle(overlay, (aabb["x1"], aabb["y1"]),
                      (aabb["x2"], aabb["y2"]), (255, 0, 0), 2)
        box = np.array(component["obb"]["points"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [box], True, (255, 255, 0), 2)
        cv2.putText(overlay, f"{index}:{component['score_p95']:.2f}",
                    (aabb["x1"], max(15, aabb["y1"] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def run_analysis(image, cam_name, confidence_gate):
    log_process_memory("before segmentation")
    model, preprocessing, inference_lock = load_model()
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized, tensor = preprocess_rgb(original, preprocessing)

    with inference_lock, torch.inference_mode():
        logits = model(tensor).float()[0, 0]
        prob = torch.sigmoid(logits / float(CFG["temperature"])).numpy().astype(np.float32)

    mask = clean_binary_mask(prob >= float(CFG["mask_threshold"]))
    cam_mask = mask if mask.sum() else (prob >= np.percentile(prob, 99.5)).astype(np.uint8)
    components = build_components(prob, mask)
    overlay = make_overlay(resized, mask, components)
    binary_mask_rgb = np.repeat((mask * 255)[..., None], 3, axis=2).astype(np.uint8)
    max_prob = float(prob.max())
    summary = {
        "pred_any": int(mask.sum() > 0),
        "confidence_gate_pass": int(max_prob >= confidence_gate),
        "max_calibrated_confidence": max_prob,
        "mask_threshold": float(CFG["mask_threshold"]),
        "confidence_gate": float(confidence_gate),
        "crack_area_px": int(mask.sum()),
        "crack_area_ratio": float(mask.mean()),
        "num_components": len(components),
        "cam_method": cam_name,
        "components": components,
    }
    # Keep the same mask/threshold/connected-components as the research model.
    # GradCAM is optional because backward hooks on 448x448 can exhaust free CPU RAM.
    del tensor, logits
    gc.collect()
    log_process_memory("segmentation complete")
    return overlay, binary_mask_rgb, None, summary, cam_mask


def generate_explanation(image: Image.Image, target_mask: np.ndarray, cam_name: str) -> np.ndarray:
    """Compute optional CAM on a 256x256 input to reduce free-tier RAM use.

    This changes only the visualization resolution; the 448x448 segmentation,
    mask threshold, component detection and quantitative metrics are unchanged.
    """
    log_process_memory("before optional CAM")
    model, preprocessing, inference_lock = load_model()
    cam_side = 256
    native_side = int(CFG["image_size"])
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    display_rgb = cv2.resize(original, (native_side, native_side), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.resize(original, (cam_side, cam_side), interpolation=cv2.INTER_LINEAR)
    x = preprocessing(rgb.astype(np.float32))
    tensor = torch.from_numpy(np.transpose(x, (2, 0, 1)).astype(np.float32)).unsqueeze(0)
    mask = cv2.resize(target_mask.astype(np.uint8), (cam_side, cam_side), interpolation=cv2.INTER_NEAREST)
    target = BinarySegmentationTarget(mask)
    cam_cls = {"HiResCAM": HiResCAM, "GradCAMPlusPlus": GradCAMPlusPlus,
               "GradCAM": GradCAM}[cam_name]
    # Resource caching is shared across users. Do not overlap model gradients/hooks.
    try:
        with inference_lock:
            model.zero_grad(set_to_none=True)
            with cam_cls(model=model, target_layers=[target_layer(model)]) as cam:
                gray_cam = cam(input_tensor=tensor, targets=[target])[0]
            model.zero_grad(set_to_none=True)
        # Overlay onto the same 448x448 input used in the segmentation view.
        scaled_cam = cv2.resize(gray_cam, (native_side, native_side), interpolation=cv2.INTER_LINEAR)
        return show_cam_on_image(display_rgb.astype(np.float32) / 255.0,
                                 np.clip(scaled_cam, 0, 1), use_rgb=True)
    finally:
        # Explicit cleanup helps avoid retaining GradCAM intermediate graphs.
        with inference_lock:
            model.zero_grad(set_to_none=True)
        del tensor
        gc.collect()
        log_process_memory("optional CAM finished")


@st.cache_data(show_spinner=False, ttl=3600)
def load_example(url: str) -> bytes:
    """Optional convenience only: the app still works when sample fetch fails."""
    if url not in EXAMPLES.values():
        raise ValueError("Unknown example URL")
    req = Request(url, headers={"User-Agent": "CrackSAFE-MRC-Research-Demo/1.0"})
    with urlopen(req, timeout=12) as response:
        data = response.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The example image is too large")
    return data


def prepare_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("Please select a JPG or PNG image.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Please choose an image smaller than 12 MB.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in ("JPEG", "PNG"):
                raise ValueError("Only JPG and PNG images are supported.")
            if im.width * im.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image resolution is too large. Maximum: 25 megapixels.")
            # Keep decoded 25MP images out of Streamlit session state: inference
            # is at 448px, so a preview max dimension of 1600px is sufficient.
            preview = ImageOps.exif_transpose(im)
            preview.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
            return preview.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The file could not be opened as a JPG or PNG image.") from exc


def png_bytes(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def reset_results() -> None:
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_key", None)
    gc.collect()


st.markdown('<div class="section-label">Upload and analyze</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Use your own photo, or try one of the project examples. No technical setup required.</div>', unsafe_allow_html=True)

with st.container(border=True):
    left, right = st.columns([1.10, .90], gap="large")
    with left:
        source = st.radio("Image source", ["Upload a photo", "Try a sample"], horizontal=True)
        image_bytes = None
        image_name = None
        if source == "Upload a photo":
            selected = st.file_uploader(
                "Choose a wall or concrete photo",
                type=["png", "jpg", "jpeg"],
                help="JPG or PNG, up to 12 MB; maximum 25 megapixels.",
                key="uploaded_surface_image",
            )
            if selected is not None:
                image_bytes = selected.getvalue()
                image_name = selected.name
            st.caption("Tip: Use a clear, well-lit photo with the surface filling the frame.")
        else:
            example_name = st.selectbox("Choose a project example", list(EXAMPLES.keys()))
            try:
                image_bytes = load_example(EXAMPLES[example_name])
                image_name = example_name
            except Exception:
                st.warning("The sample is temporarily unavailable. Please upload your own JPG or PNG image.")
        with st.expander("Advanced options (optional)", expanded=False):
            cam_method = st.selectbox(
                "Explanation technique",
                ("HiResCAM", "GradCAMPlusPlus", "GradCAM"),
                help="Changes the visual explanation, not the underlying segmentation mask.",
            )
            confidence = st.slider(
                "Peak-probability check", .50, .99, .90, .01,
                help="Compares the highest pixel probability with this value. Does not change the segmentation mask or establish overall inspection certainty.",
            )
            st.caption("Default choices work for most demonstrations. The mask threshold is fixed by the research configuration.")
    with right:
        preview_image = None
        if image_bytes:
            try:
                preview_image = prepare_image(image_bytes)
                st.image(preview_image, caption=f"Input preview · {preview_image.width} × {preview_image.height} px", width="stretch")
            except ValueError as err:
                st.error(str(err))
        else:
            st.markdown("""
            <div class="help-strip">
              <strong>Your photo preview will appear here.</strong><br>
              You can upload a JPG/PNG or load a sample, then select <b>Analyze image</b>.
              The model evaluates a resized 448 × 448 version of the image.
            </div>
            """, unsafe_allow_html=True)

    if preview_image is not None:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        run_col, clear_col = st.columns([2.5,1], gap="small")
        with run_col:
            run_clicked = st.button("Analyze image", type="primary", width="stretch")
        with clear_col:
            clear_clicked = st.button("Clear results", width="stretch")
        if clear_clicked:
            reset_results()
        key = hashlib.sha256(image_bytes).hexdigest() + f"|{cam_method}|{confidence:.2f}"
        if run_clicked:
            reset_results()
            try:
                with st.spinner("Running crack segmentation..."):
                    result = run_analysis(preview_image, cam_method, confidence)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_key"] = key
            except Exception as exc:
                st.error("The analysis could not be completed. Please retry, or try a smaller image.")
                with st.expander("Technical error details"):
                    st.exception(exc)
        elif st.session_state.get("analysis_result") and st.session_state.get("analysis_key") != key:
            st.info("Your image or settings changed. Click **Analyze image** to refresh the results.")
    else:
        key = None
        st.button("Analyze image", disabled=True, type="primary", width="stretch")

if key and st.session_state.get("analysis_key") == key and "analysis_result" in st.session_state:
    overlay, binary, heatmap, summary, cam_target_mask = st.session_state["analysis_result"]
    found = bool(summary["pred_any"])
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="small-overline">Analysis results</div>', unsafe_allow_html=True)
    message = ("Possible crack regions detected" if found else "No crack region detected at the configured threshold")
    explanation = ("The model identified one or more segmented regions. Review the overlay before interpreting the result."
                   if found else "No positive mask region remained after post-processing. This does not establish that the surface is crack-free.")
    cssclass = "result-banner" if found else "result-banner no-crack"
    st.markdown(f'<div class="{cssclass}"><strong>{message}</strong><p>{explanation}</p></div>',unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4, gap="small")
    m1.metric("Finding", "Crack" if found else "No mask")
    m2.metric("Detected regions", f"{int(summary['num_components']):,}")
    m3.metric("Estimated image coverage", f"{summary['crack_area_ratio'] * 100:.1f}%")
    m4.metric("Peak pixel probability", f"{summary['max_calibrated_confidence'] * 100:.1f}%")
    st.caption("Coverage refers to the 448 × 448 model mask; peak probability is not a whole-image accuracy or safety score.")

    st.markdown("### See what the model found")
    if heatmap is None:
        st.caption("For reliable free hosting, crack segmentation runs first. "
                   "The optional explanation uses additional CPU resources at a reduced 256px resolution.")
        if st.button("Generate explanation heatmap (optional, higher memory use)", key="make_cam", width="stretch"):
            try:
                with st.spinner("Preparing the explanation on the shared CPU..."):
                    heatmap = generate_explanation(preview_image, cam_target_mask, cam_method)
                st.session_state["analysis_result"] = (overlay, binary, heatmap, summary, cam_target_mask)
            except Exception as exc:
                st.warning("The explanation could not be generated on this host. The crack analysis above remains available.")
                with st.expander("Explanation error details"):
                    st.exception(exc)
    tab_overview, tab_overlay, tab_mask, tab_heat, tab_details = st.tabs(
        ["Quick comparison", "Detection overlay", "Binary mask", "AI explanation", "Technical details"])

    with tab_overview:
        a,b,c = st.columns(3, gap="medium")
        with a:
            st.image(preview_image, caption="1 · Your image", width="stretch")
        with b:
            st.image(overlay, caption="2 · Marked regions", width="stretch")
        with c:
            if heatmap is not None:
                st.image(heatmap, caption="3 · Visual explanation (256px CAM)", width="stretch")
            else:
                st.info("Optional explanation not generated. Use the button above to add it.")
        st.markdown("""
        <div class="help-strip"><strong>How to read this:</strong> The green tint marks pixels in the predicted crack mask.
        Blue rectangles and yellow outlines mark connected regions. Warm colors in the heatmap highlight
        areas the model used for the selected explanation method; they are not additional crack predictions.</div>
        """, unsafe_allow_html=True)

    with tab_overlay:
        view, info = st.columns([1.35, .85], gap="large")
        with view:
            st.image(overlay, caption="Detected region overlay", width="stretch")
            st.download_button("Download overlay (PNG)", png_bytes(overlay),
                               "cracksafe_detection_overlay.png", "image/png",width="stretch")
        with info:
            st.markdown("**Detected regions**")
            st.write(f"{int(summary['num_components'])} connected region(s) after morphological cleanup.")
            st.write("**Green** = predicted crack pixels; **blue** = axis-aligned boxes; **yellow** = oriented boxes.")
            st.caption("Region labels indicate the local 95th-percentile pixel score, not a structural risk rating.")

    with tab_mask:
        view, info = st.columns([1.35, .85], gap="large")
        with view:
            st.image(binary, caption="Binary segmentation mask", width="stretch")
            st.download_button("Download mask (PNG)", png_bytes(binary),
                               "cracksafe_binary_mask.png", "image/png",width="stretch")
        with info:
            st.markdown("**Reading the mask**")
            st.write("White pixels represent the model's predicted crack area; black pixels represent background.")
            st.write(f"Research mask threshold: **{float(CFG['mask_threshold']):.2f}**")
            st.caption("Mask pixels are computed at 448 × 448 resolution and then cleaned according to the packaged research configuration.")

    with tab_heat:
        view, info = st.columns([1.35, .85], gap="large")
        with view:
            if heatmap is not None:
                st.image(heatmap, caption=f"{summary['cam_method']} explanation (256px CAM)", width="stretch")
                st.download_button("Download heatmap (PNG)", png_bytes(heatmap),
                                   "cracksafe_explanation_heatmap.png", "image/png",width="stretch")
            else:
                st.info("Explanation not generated yet. Use the Generate AI explanation button above.")
        with info:
            st.markdown("**Understanding the heatmap**")
            st.write("Warmer colors indicate regions highlighted by the selected CAM technique.")
            st.write(f"Current technique: **{summary['cam_method']}**")
            st.caption("The optional heatmap uses a 256px CAM input for memory efficiency and is rescaled for viewing. Attribution is not a calibrated probability map or crack boundary.")

    with tab_details:
        st.markdown("**Research configuration**")
        c1,c2 = st.columns(2)
        with c1:
            st.write(f"**Encoder:** {CFG['encoder_name']}")
            st.write(f"**Selected ablation:** {CFG.get('selected_ablation','Finalist model')}")
            st.write(f"**Input resolution:** {CFG['image_size']} × {CFG['image_size']}")
        with c2:
            st.write(f"**Temperature:** {float(CFG['temperature']):.6f}")
            st.write(f"**Mask threshold:** {float(CFG['mask_threshold']):.2f}")
            st.write(f"**Peak-probability check:** {'Passed' if summary['confidence_gate_pass'] else 'Not passed'}")
        st.caption("The adjustable peak-probability check is only an additional informational gate; it does not modify mask segmentation.")
        with st.expander("Show full detections JSON", expanded=False):
            st.json(summary, expanded=False)
        st.download_button("Download full results (JSON)",
                           json.dumps(summary, indent=2).encode("utf-8"),
                           "cracksafe_analysis_results.json", "application/json")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    with st.expander("How reliable is this result?", expanded=False):
        st.write("The system demonstrates a research segmentation model on an uploaded surface image. "
                 "Results depend on image quality, lighting, surface material and domain differences. "
                 "A missed region does not demonstrate a structurally sound surface. "
                 "Please seek a qualified inspection for decisions about actual structural safety.")
else:
    st.markdown("""
    <div class="help-strip" style="margin-top:1.1rem">
      <strong>What you'll get after analysis</strong><br>
      A clear crack/no-mask finding, estimated segmented coverage, a region overlay,
      a black-and-white segmentation mask, an optional CAM explanation, and downloadable research data.
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="footer-line">
  CrackSAFE-MRC · University research demonstration · Not a structural safety certification tool<br>
  <a href="{RESEARCH_REPO}" target="_blank" rel="noopener noreferrer">Explore the research methodology and artifacts</a>
</div>
""",unsafe_allow_html=True)
