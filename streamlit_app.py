"""CrackSAFE-MRC Streamlit research demo.

Inference and visualization are adapted from the author's Gradio demo.
Weights are downloaded from the public university Hugging Face model repository.
"""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "phase14a_demo_config.json").read_text(encoding="utf-8"))
MODEL_REPO = "mufrat-montasir/CrackSAFE-MRC-Model"

st.set_page_config(
    page_title="CrackSAFE-MRC | Structural Crack Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Limit CPU oversubscription on shared, memory-constrained hosting.
torch.set_num_threads(2)

st.markdown("""
<style>
.block-container {padding-top: 1.7rem; max-width: 1400px;}
.hero {padding: 1.25rem 1.5rem; border: 1px solid #cbd5e1; border-radius: 1rem;
       background: linear-gradient(120deg,#f8fafc,#edf2f8); color: #0f172a; margin-bottom:1.25rem;}
.hero h1 {font-size: 2.1rem; font-weight: 750; margin: 0 0 .4rem; color:#0f172a;}
.hero p {font-size:1rem; margin:0; color:#334155;}
.note {color:#64748b; font-size:.84rem;}
</style>
<div class="hero">
  <h1>CrackSAFE-MRC</h1>
  <p>Multi-model structural crack research · Pixel segmentation · Detected components · Explainability</p>
</div>
""", unsafe_allow_html=True)


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
    preprocessing = smp.encoders.get_preprocessing_fn(CFG["encoder_name"], "imagenet")
    return model, preprocessing


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
    model, preprocessing = load_model()
    original = np.asarray(image.convert("RGB"), dtype=np.uint8)
    resized, tensor = preprocess_rgb(original, preprocessing)

    with torch.inference_mode():
        logits = model(tensor).float()[0, 0]
        prob = torch.sigmoid(logits / float(CFG["temperature"])).numpy().astype(np.float32)

    mask = clean_binary_mask(prob >= float(CFG["mask_threshold"]))
    cam_mask = mask if mask.sum() else (prob >= np.percentile(prob, 99.5)).astype(np.uint8)
    target = BinarySegmentationTarget(cam_mask)
    cam_cls = {"HiResCAM": HiResCAM, "GradCAMPlusPlus": GradCAMPlusPlus,
               "GradCAM": GradCAM}[cam_name]
    with cam_cls(model=model, target_layers=[target_layer(model)]) as cam:
        heatmap = cam(input_tensor=tensor, targets=[target])[0]
    heatmap_rgb = show_cam_on_image(resized.astype(np.float32) / 255.0,
                                    heatmap, use_rgb=True)

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
    return overlay, binary_mask_rgb, heatmap_rgb, summary


with st.sidebar:
    st.header("Input & controls")
    upload = st.file_uploader("Upload a wall or concrete image", type=["jpg", "jpeg", "png"])
    cam_method = st.selectbox("Explainability method", ("HiResCAM", "GradCAMPlusPlus", "GradCAM"))
    confidence = st.slider("Confidence gate", 0.50, 0.99, 0.90, step=0.01)
    run = st.button("Run analysis", type="primary", use_container_width=True, disabled=upload is None)
    with st.expander("Model details"):
        st.write(f"**Selected ablation:** {CFG.get('selected_ablation', 'Finalist model')}")
        st.write(f"**Encoder:** {CFG['encoder_name']}")
        st.write(f"**Input size:** {CFG['image_size']} × {CFG['image_size']}")
        st.write(f"**Temperature:** {float(CFG['temperature']):.6f}")
        st.write(f"**Mask threshold:** {CFG['mask_threshold']}")
        st.write(f"**Minimum region area:** {CFG['packaging_min_component_area']} px")
    st.caption("Research demonstration only. Not a structural safety certification tool.")

if upload is None:
    st.info("Upload a wall or concrete image in the sidebar, then select **Run analysis**.")
else:
    try:
        uploaded_image = Image.open(upload).convert("RGB")
        if uploaded_image.width * uploaded_image.height > 30_000_000:
            st.error("Image is too large. Please select an image smaller than 30 megapixels.")
        else:
            col_preview, col_note = st.columns([1, 1])
            with col_preview:
                st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
            with col_note:
                st.markdown("### Analysis workflow")
                st.markdown("Segmentation → calibrated mask → connected components → CAM visualization")
                st.caption(f"Input: {uploaded_image.width} × {uploaded_image.height}. Inference: {CFG['image_size']} × {CFG['image_size']}.")
            if run:
                with st.spinner("Running segmentation and explainability on CPU. First model load can take longer..."):
                    overlay, binary, heatmap, summary = run_analysis(uploaded_image, cam_method, confidence)
                st.session_state["last_analysis"] = (overlay, binary, heatmap, summary)
                st.session_state["last_file"] = upload.name
            if "last_analysis" in st.session_state and st.session_state.get("last_file") == upload.name:
                overlay, binary, heatmap, summary = st.session_state["last_analysis"]
                st.subheader("Analysis results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", "Crack" if summary["pred_any"] else "No crack")
                c2.metric("Components", summary["num_components"])
                c3.metric("Gate passed", "Yes" if summary["confidence_gate_pass"] else "No")
                tab1, tab2, tab3 = st.tabs(["Detection overlay", "Binary mask", "Explanation heatmap"])
                with tab1:
                    st.image(overlay, use_container_width=True)
                with tab2:
                    st.image(binary, use_container_width=True)
                with tab3:
                    st.image(heatmap, use_container_width=True)
                with st.expander("Detections JSON", expanded=True):
                    st.json(summary)
                    st.download_button("Download results JSON", data=json.dumps(summary, indent=2),
                                       file_name="cracksafe_detections.json", mime="application/json")
    except Exception as exc:
        st.error("The input image could not be processed. Please try a standard JPG or PNG image.")
        st.exception(exc)

st.caption("CrackSAFE-MRC · Academic computer vision research · University demonstration")
