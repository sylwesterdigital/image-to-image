# app.py — auto-HTTPS with persistent self-signed cert (SAN includes LAN IP)
import io
import os
import time
import socket
import ipaddress
from pathlib import Path
from datetime import datetime, timedelta
from threading import Lock

from flask import Flask, request, send_file, render_template, jsonify, g
from PIL import Image

import torch
from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler

# -----------------------------
# Flask
# -----------------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

# --- DEV convenience: pick up template/static edits on browser refresh ---
# (No server restart required; keeps your "python app.py" workflow.)
app.config["TEMPLATES_AUTO_RELOAD"] = True          # reload Jinja templates on each request
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0         # disable static caching in dev


# --- session outputs dir ---
from datetime import datetime
from flask import send_from_directory

BASE_OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
SESSION_STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SESSION_DIR = os.path.join(BASE_OUT_DIR, SESSION_STAMP)
os.makedirs(SESSION_DIR, exist_ok=True)
print(f"[outputs] session dir: {SESSION_DIR}")

@app.get("/outputs/<path:fn>")
def serve_output(fn):
    # Serve files from outputs/ (read-only)
    safe_path = os.path.normpath(fn)
    if safe_path.startswith(("..", "/")):
        return {"error": "bad path"}, 400
    return send_from_directory(BASE_OUT_DIR, safe_path)

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

_infer_lock = Lock()  # serialize scheduler access


@app.before_request
def _stamp_t0():
    g.t0 = time.time()


@app.after_request
def _log_and_no_cache(resp):
    # Log timing
    try:
        dt = (time.time() - g.t0) * 1000
        print(f"{request.method} {request.path} -> {resp.status_code} ({dt:.0f} ms)")
    except Exception:
        pass

    # Dev: make sure the browser doesn't cache templates/static so refresh always reflects edits
    resp.headers.setdefault("Cache-Control", "no-store")
    resp.headers.setdefault("Pragma", "no-cache")
    resp.headers.setdefault("Expires", "0")
    return resp


@app.get("/health")
def health():
    return {"ok": True}, 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    try:
        return app.send_static_file("favicon.ico")
    except Exception:
        return ("", 204)


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
        return jsonify({"error": "No frame"}), 400

    prompt: str = request.form.get("prompt", "").strip() or "high quality photo, cinematic lighting"
    try:
        strength = float(request.form.get("strength", "0.5"))
        guidance = float(request.form.get("guidance", "1.0"))
        steps = int(request.form.get("steps", "4"))
    except Exception:
        strength, guidance, steps = 0.5, 1.0, 4

    steps = max(2, min(steps, 8))
    guidance = max(0.0, min(guidance, 6.0))
    strength = max(0.05, min(strength, 1.0))

    img = Image.open(file.stream)
    img = _to_rgb(img, max_side=640)

    try:
        with _infer_lock:
            pipe.scheduler.set_timesteps(steps, device=pipe.device)
            if getattr(pipe.scheduler, "_step_index", None) is None:
                pipe.scheduler._step_index = 0

            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast("cuda"):
                        out = pipe(
                            prompt=prompt,
                            image=img,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            strength=strength,
                            generator=None,
                        ).images[0]
                else:
                    out = pipe(
                        prompt=prompt,
                        image=img,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        strength=strength,
                        generator=None,
                    ).images[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # --- save to disk (PNG) ---
    ts = datetime.now().strftime("%H-%M-%S_%f")[:-3]
    filename = f"{ts}.png"
    abs_path = os.path.join(SESSION_DIR, filename)
    rel_path = os.path.relpath(abs_path, BASE_OUT_DIR).replace("\\", "/")
    try:
        out.save(abs_path, format="PNG")  # lossless on disk
    except Exception as e:
        print("[outputs] save error:", e)

    # --- also stream JPEG back to client (fast) ---
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    resp = send_file(buf, mimetype="image/jpeg")
    resp.headers["X-Output-Path"] = rel_path  # e.g. "2025-10-09_15-42-01/12-33-44_123.png"
    return resp


# -----------------------------
# Auto-create persistent self-signed cert with SAN for LAN IP
# -----------------------------
def _detect_ip(default="192.168.1.90"):
    ip = os.environ.get("HOST_IP", "").strip()
    if ip:
        return ip
    # best-effort local IP detection; fallback to default
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return default


def ensure_cert(cert_path: Path, key_path: Path, ip_str: str):
    """
    Create cert/key if missing or if existing cert does not contain the current IP in SANs.
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    cert_path.parent.mkdir(parents=True, exist_ok=True)

    need_new = True
    if cert_path.exists():
        try:
            data = cert_path.read_bytes()
            cert = x509.load_pem_x509_certificate(data)
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            all_ips = [str(v) for v in san.get_values_for_type(x509.IPAddress)]
            all_dns = [str(v) for v in san.get_values_for_type(x509.DNSName)]
            if ip_str in all_ips and "localhost" in all_dns:
                need_new = False
        except Exception:
            need_new = True

    if not need_new and key_path.exists():
        return  # good to go

    # (Re)generate
    ip_obj = ipaddress.ip_address(ip_str)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"lcm-local")])

    san = x509.SubjectAlternativeName([
        x509.IPAddress(ip_obj),
        x509.DNSName("localhost"),
        # extra conveniences
        x509.DNSName("127.0.0.1"),
    ])

    now = datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(san, critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    # write files
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    # Where to listen (and which IP to embed in the cert SANs)
    host_ip = _detect_ip()  # override via HOST_IP env if needed (e.g. 192.168.1.90)
    host = "0.0.0.0"        # listen on all interfaces

    # Cert paths (persistent)
    certs_dir = Path(os.environ.get("CERTS_DIR", "certs"))
    cert_path = certs_dir / "lan.pem"
    key_path = certs_dir / "lan-key.pem"

    if os.environ.get("DISABLE_HTTPS", "0") == "1":
        print(f"HTTP mode on http://{host_ip}:{port}")
        app.run(host=host, port=port, threaded=True)
    else:
        try:
            # Create/reuse a self-signed cert that includes your LAN IP
            ensure_cert(cert_path, key_path, host_ip)
        except Exception as e:
            raise SystemExit(
                f"Failed to create certs automatically: {e}\n"
                f"Set DISABLE_HTTPS=1 to run over HTTP (no webcam), "
                f"or install 'cryptography'."
            )
        print(f"HTTPS mode on https://{host_ip}:{port}  (self-signed)")
        app.run(host=host, port=port, threaded=True, ssl_context=(str(cert_path), str(key_path)))
