# CrackSAFE-MRC | Streamlit Demonstration

A public, CPU-based academic research demo for structural crack segmentation and model explainability. It is adapted from the CrackSAFE-MRC Gradio demo.

## Deploying

1. Create a **new public, empty** GitHub repository named `CrackSAFE-MRC-Streamlit` under `mufrat-montasir`.
2. Extract this ZIP, and upload all four files to the repository root (do **not** upload the ZIP itself).
3. Go to https://share.streamlit.io/, choose **Create app → Deploy a public app from GitHub** and select:
   - Repository: `mufrat-montasir/CrackSAFE-MRC-Streamlit`
   - Branch: `main`
   - Main file: `streamlit_app.py`
   - Python: 3.11, if the Advanced setting provides a choice
4. Click **Deploy**. A successful deployment will load `checkpoint.pt` from the public model repository `mufrat-montasir/CrackSAFE-MRC-Model`.
5. Test with one wall/concrete image and check all three image outputs plus JSON.

## Notes

- No trained model is embedded in this GitHub repository. A trusted `checkpoint.pt` is downloaded via `hf_hub_download` and cached by the hosting platform.
- The file is loaded with `torch.load(..., weights_only=True)`; never substitute unknown checkpoints.
- Gradio app functionality was converted to Streamlit; Gradio and Streamlit layouts are not pixel-identical.
- Model inference and explainability run on CPU. Depending on traffic and Streamlit's available memory, startup or CAM generation might be slow or exceed free-tier limits.
- This research demo is not a substitute for professional structural engineering inspection.
- The original Hugging Face Space `abArman6979/Wallcrack` is not needed by this Streamlit app.

GitHub research repository: https://github.com/mufrat-montasir/CrackSAFE-MRC

Model repository: https://huggingface.co/mufrat-montasir/CrackSAFE-MRC-Model
