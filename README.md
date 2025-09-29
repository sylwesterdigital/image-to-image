# Webcam → Real-Time LCM (img2img) — POC



https://github.com/user-attachments/assets/dc103afa-a3dd-4bdb-ba02-65b8a643edda



Dead-simple demo: left pane shows live webcam, right pane shows image-to-image output from a Real-Time Latent Consistency Model (LCM).
Stack: **Flask + vanilla JS + lil-gui + Diffusers**. Works on macOS (Apple Silicon M1/M2 via **MPS**), NVIDIA CUDA, and CPU (slow).

---

## Features

* Two-pane responsive UI (mobile friendly).
* Webcam device chooser with **lil-gui**.
* Resolution, facing mode, horizontal/vertical flip.
* LCM img2img with **very low steps** (2–8), low CFG (sweet spot ~0–2).
* One-shot processing or adaptive live loop that paces to actual latency.
* HTTPS by default for camera APIs (self-signed dev cert).

---

## Repo layout

```
.
├─ app.py                    # Flask server + Diffusers pipeline (LCM img2img)
├─ templates/
│  └─ index.html             # UI (vanilla JS + lil-gui)
└─ static/
   └─ favicon.ico            # optional
```

---

## Requirements

* Python 3.10–3.12 (3.13 also works with recent wheels).
* macOS (Apple Silicon), Linux, or Windows.
* For HTTPS: `cryptography` wheel installed.

Python deps (installed below):
`flask pillow diffusers transformers accelerate safetensors torch torchvision torchaudio cryptography`

> First run will download models from Hugging Face:
> `SimianLuo/LCM_Dreamshaper_v7` (UNet) + `Lykon/dreamshaper-7` (SD 1.5 img2img backbone).

---

## Quickstart

### macOS (M2/M1) — Zsh

```zsh
python3 -m venv .venv
source .venv/bin/activate

# PyTorch (MPS) + deps
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install flask pillow diffusers transformers accelerate safetensors cryptography

# Optional: MPS fallbacks for missing ops
export PYTORCH_ENABLE_MPS_FALLBACK=1

python app.py
```

Open:

* Desktop: **[https://localhost:5000](https://localhost:5000)** (accept the self-signed certificate)
* Phone (same Wi-Fi): **https://<your-mac-ip>:5000** (accept the certificate)

### Linux (CUDA) / Windows (CUDA)

Install a CUDA-matching PyTorch from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) then:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install flask pillow diffusers transformers accelerate safetensors cryptography
python app.py
```

Open **[https://localhost:5000](https://localhost:5000)** and accept the cert.

> To run without HTTPS (not recommended; browsers will hide camera devices), set `DISABLE_HTTPS=1` in the environment before `python app.py`, then use `http://localhost:5000` on desktop only.

---

## Usage

1. Load the page over **HTTPS**. A prompt to trust the self-signed cert appears once.
2. Click **Camera → ↻ Rescan** to list devices (browsers reveal labels only after permission).
3. Pick **Device**, adjust **Facing** (user/environment) and **Resolution**.
4. Set model params in **Model**:

   * **Prompt**: text prompt for img2img.
   * **Strength**: deviation from the input frame (0.05–1).
   * **Guidance**: CFG; **LCM prefers low values** (0–2).
   * **Steps**: 2–8; 2–4 are typically fastest.
5. Click **Process one frame** or toggle **Live**.
6. Output image updates in the right pane.

Controls live inside a floating **lil-gui** panel (top-right by default).

---

## API

### `POST /process`

`multipart/form-data`:

* `frame`: JPEG image bytes (captured in browser)
* `prompt`: str (optional)
* `strength`: float [0.05, 1.0]
* `guidance`: float [0, 6] (LCM sweet spot ~0–2)
* `steps`: int [2, 8]

Returns: `image/jpeg` (transformed frame).
On error: JSON `{"error":"..."}` with HTTP 500.

---

## Environment variables

* `PORT` — default `5000`.
* `DISABLE_HTTPS=1` — serve HTTP only (not recommended; camera device list may be empty).

  * Without this, server requires `cryptography` and serves **HTTPS** with an ad-hoc self-signed cert.

---

## Performance tips

* **LCM** is optimized for **very few steps**. Use **2–4** steps + guidance **0–2**.
* Lower the capture resolution in GUI (**640×480**) for higher throughput.
* The live loop auto-paces to the measured round-trip latency; it adjusts after each response.
* On CPU, keep resolution very low and expect slow throughput.
* On MPS (Apple Silicon), ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops.

---

## Troubleshooting

* **No cameras listed / “needs HTTPS or localhost”**
  Use **[https://localhost:5000](https://localhost:5000)** or **https://<LAN-IP>:5000**. Accept the cert.
  Browser security blocks camera enumeration on insecure origins.

* **Camera list empty until permission**
  Click **↻ Rescan** after granting camera permission; browsers reveal device labels only after first grant.

* **500 error with LCMScheduler `_step_index`**
  The POC explicitly calls `pipe.scheduler.set_timesteps(steps, device=...)` and guards `_step_index`.
  Ensure `diffusers`, `accelerate`, and `transformers` are up to date.

* **“Install 'cryptography'… to enable HTTPS”**
  `pip install cryptography` inside the venv.

* **Safari quirks**
  Always use HTTPS. Older Safari versions may require a first `getUserMedia` call before `enumerateDevices` returns labels.

* **High VRAM usage (CUDA)**
  Reduce steps, guidance, and input max side (`_to_rgb(..., max_side=640)` in `app.py`).
  Keep `strength` modest (e.g., 0.4–0.6) to avoid excessive denoising work.

---

## Notes on models

* UNet: `SimianLuo/LCM_Dreamshaper_v7` (LCM-distilled)
* Backbone: `Lykon/dreamshaper-7` (SD 1.5 img2img)
* Scheduler: `LCMScheduler`
* Safety checker disabled in this POC for speed. Re-enable if required.

---

## Development

```bash
# optional dev reload (use a proper WSGI for production)
export FLASK_ENV=development
python app.py
```

---

## License

MIT (or choose a license). Add a proper `LICENSE` file if distributing.
