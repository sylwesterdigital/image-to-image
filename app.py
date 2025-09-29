import io
import os
from typing import Optional

from flask import Flask, request, send_file, send_from_directory
from PIL import Image

import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler

# -----------------------------
# Model setup (fast + minimal)
# -----------------------------
# Uses an LCM UNet distilled for SD 1.5 and plugs it into a regular img2img pipeline,
# then swaps in the LCMScheduler for 2–4 inference steps, per HF docs.

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# LCM UNet (image2image)
# "SimianLuo/LCM_Dreamshaper_v7" provides an LCM UNet distilled for v1.5
unet = UNet2DConditionModel.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    subfolder="unet",
    torch_dtype=dtype,
)

# Base img2img pipeline (v1.5 family; DreamShaper v7 pairs well with the above LCM UNet)
pipe = AutoPipelineForImage2Image.from_pretrained(
    "Lykon/dreamshaper-7",
    unet=unet,
    torch_dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# Small speed tweaks
pipe.enable_model_cpu_offload() if device != "cuda" else pipe.to(device)  # keep simple
pipe.set_progress_bar_config(disable=True)
# Safety checker off by default for speed; switch on if needed:
pipe.safety_checker = None

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__, static_url_path="", static_folder=".")

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

def _to_rgb(img: Image.Image, max_side: int = 640) -> Image.Image:
    """Convert to RGB and downscale (keeps things snappy)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

@app.post("/process")
def process():
    """
    Accepts multipart/form-data:
      - frame: JPEG/PNG bytes from the browser canvas
      - prompt: text prompt
      - strength: float (0..1)
      - guidance: float (~1..13, LCM sweet spot often 3..9)
      - steps: int (2..6; LCM works in ~2–4)
    Returns: JPEG bytes of the transformed image.
    """
    file = request.files.get("frame")
    if file is None:
        return ("No frame", 400)

    prompt: str = request.form.get("prompt", "").strip() or "high quality photo, cinematic lighting"
    try:
        strength = float(request.form.get("strength", "0.5"))
        guidance = float(request.form.get("guidance", "4.0"))
        steps = int(request.form.get("steps", "4"))
    except Exception:
        strength, guidance, steps = 0.5, 4.0, 4

    img = Image.open(file.stream)
    img = _to_rgb(img, max_side=640)

    generator: Optional[torch.Generator] = None  # could set a seed if needed
    kwargs = dict(
        prompt=prompt,
        image=img,
        num_inference_steps=max(2, min(steps, 8)),
        guidance_scale=max(1.0, min(guidance, 13.0)),
        strength=max(0.05, min(strength, 1.0)),
        generator=generator,
    )

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda"):
                out = pipe(**kwargs).images[0]
        else:
            out = pipe(**kwargs).images[0]

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    # Usage: python app.py  (then open http://localhost:5000)
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
