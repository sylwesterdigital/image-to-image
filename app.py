# app.py — fixes erratic LCM steps + safer /process + HTTPS
import io
import os
from typing import Optional

from flask import Flask, request, send_file, render_template, jsonify
from PIL import Image

import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler

# -----------------------------
# Device & dtype
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

# -----------------------------
# Model (LCM UNet + SD1.5 img2img + LCM scheduler)
# -----------------------------
unet = UNet2DConditionModel.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    subfolder="unet",
    torch_dtype=dtype,
)

pipe = AutoPipelineForImage2Image.from_pretrained(
    "Lykon/dreamshaper-7",
    unet=unet,
    torch_dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to(device)
pipe.set_progress_bar_config(disable=True)
pipe.safety_checker = None
if device == "cpu":
    pipe.enable_model_cpu_offload()

# -----------------------------
# Flask
# -----------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.ico")

def _to_rgb(img: Image.Image, max_side: int = 640) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

@app.post("/process")
def process():
    file = request.files.get("frame")
    if file is None:
        return ("No frame", 400)

    prompt: str = request.form.get("prompt", "").strip() or "high quality photo, cinematic lighting"
    try:
        strength = float(request.form.get("strength", "0.5"))
        guidance = float(request.form.get("guidance", "1.0"))  # LCM likes low CFG (≈0–2)
        steps = int(request.form.get("steps", "4"))
    except Exception:
        strength, guidance, steps = 0.5, 1.0, 4

    steps = max(2, min(steps, 8))
    guidance = max(0.0, min(guidance, 6.0))
    strength = max(0.05, min(strength, 1.0))

    img = Image.open(file.stream)
    img = _to_rgb(img, max_side=640)

    # --- IMPORTANT: ensure LCM scheduler state is reset every call ---
    # Some diffusers versions leave _step_index=None if timesteps weren't set explicitly.
    pipe.scheduler.set_timesteps(steps, device=pipe.device)
    if getattr(pipe.scheduler, "_step_index", None) is None:
        pipe.scheduler._step_index = 0  # guard for buggy sched init paths

    kwargs = dict(
        prompt=prompt,
        image=img,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=strength,
        generator=None,
    )

    try:
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast("cuda"):
                    out = pipe(**kwargs).images[0]
            else:
                out = pipe(**kwargs).images[0]
    except Exception as e:
        # return JSON error so the frontend can back off gracefully
        return jsonify({"error": str(e)}), 500

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    if os.environ.get("DISABLE_HTTPS", "0") != "1":
        try:
            import cryptography  # noqa: F401
        except Exception:
            raise SystemExit("Install 'cryptography' (e.g., `pip install cryptography`) to enable HTTPS for camera access.")
        app.run(host="0.0.0.0", port=port, threaded=True, ssl_context="adhoc")
    else:
        app.run(host="0.0.0.0", port=port, threaded=True)
