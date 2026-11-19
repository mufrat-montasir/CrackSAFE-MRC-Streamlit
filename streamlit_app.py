"""CrackSAFE-MRC | Accessible, guided structural-crack research demo.

UI-only redesign: preserves the project inference, segmentation, calibration,
connected-component detection, and CAM logic from the previous deployment.
Model weights load from the public university Hugging Face repository.
"""
from __future__ import annotations

import hashlib
import io
import json
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
torch.set_num_threads(2)

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
