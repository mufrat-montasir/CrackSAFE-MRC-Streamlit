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
:root {--ink:#172d3b;--deep:#0e3445;--teal:#087f76;--muted:#617582;--rule:#dfe9e9;--soft:#f2f7f7;}
[data-testid="stAppViewContainer"] {background: #f7faf9;}
[data-testid="stHeader"] {background: rgba(247,250,249,.92);}
.block-container {max-width:1160px; padding-top:1.6rem; padding-bottom:3rem;}
h1,h2,h3 {color:#172d3b;letter-spacing:-.035em;}
h2 {font-weight:750;}
p,li {line-height:1.6;}
.hero {padding:2.1rem 2.35rem;border:1px solid #dbe6e7;border-radius:22px;
       background:linear-gradient(115deg,#ffffff 0%,#f3f9f8 66%,#e4f2ef 100%);
       box-shadow:0 10px 30px rgba(23,45,59,.045); margin-bottom:1.25rem;}
.eyebrow {font-weight:800;letter-spacing:.15em;text-transform:uppercase;font-size:.70rem;color:#087f76;}
.hero h1 {font-size:clamp(2rem,3.7vw,3rem);line-height:1.08; margin:.6rem 0 .65rem;font-weight:800;}
.hero p {color:#4f6773;margin:0;max-width:770px;font-size:1.04rem;line-height:1.55;}
.hero-meta {margin-top:1.2rem;display:flex;gap:.6rem;flex-wrap:wrap;}
.meta-chip {border:1px solid #cfe3df;border-radius:99px;padding:.35rem .74rem;
            color:#215a56;font-size:.78rem;font-weight:650;background:#ffffffbb;}
.steps {display:flex; gap:0.7rem; margin:0 0 1.15rem;flex-wrap:wrap;}
.step {display:flex;align-items:center;gap:.5rem;background:#fff;border:1px solid #e2ecec;
       border-radius:12px;padding:.65rem .9rem;color:#526d76;font-weight:650;font-size:.86rem;}
.step b {color:#087f76;font-size:.78rem;font-weight:850;}
.section-label {font-size:1.18rem;color:#173746;font-weight:800;margin:0 0 .2rem;}
.section-sub {color:#6a7f85;font-size:.91rem;margin:0 0 .8rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:#dee9e8;}
[data-testid="stMetric"] {border:1px solid #e2ebea;background:#fff;border-radius:14px;
                            padding:15px 17px;min-height:112px;}
[data-testid="stMetricLabel"] {color:#617581;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"] {
    border-radius:10px;background:#087f76!important;border-color:#087f76!important;
    color:#fff!important;font-weight:700;min-height:47px;}
.stButton button[kind="primary"]:hover {background:#09675f!important;border-color:#09675f!important;}
.stButton button, .stDownloadButton button {border-radius:10px;min-height:41px;}
.result-banner {padding:1.15rem 1.45rem;border-radius:15px;margin:0 0 1rem;
                border:1px solid #b5ddd3;background:#eaf8f4;color:#18544d;}
.result-banner.no-crack {border-color:#cbd9e6;background:#f0f5fa;color:#254763;}
.result-banner strong {font-size:1.12rem;}
.result-banner p {margin:.32rem 0 0;font-size:.89rem;color:inherit;}
.help-strip {padding:.95rem 1.15rem;background:#edf5f5;border:1px solid #d9e9e7;
             color:#34535a;border-radius:13px;font-size:.87rem;line-height:1.65;}
.footer-line {padding-top:1.6rem;color:#75848a;font-size:.79rem;border-top:1px solid #dce7e7;margin-top:2.2rem;}
.small-overline {color:#0c776f;font-size:.77rem;font-weight:800;text-transform:uppercase;letter-spacing:.12em;}
@media(max-width:720px){.hero{padding:1.45rem 1.25rem;border-radius:16px;}
 .hero h1{font-size:2rem;}.block-container{padding-left:1rem;padding-right:1rem;}
 .step{font-size:.78rem;padding:.52rem .6rem;}}
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
