"""
HTTP server for LIBERO video stream + MiniCPM-o-4.5 analysis.

Runs from the `minicpm` conda environment (transformers 4.51.0).
VLA runs from `unifolm-vla` env, posts frames to this server.

Endpoints:
  GET  /        HTML page with video + analysis panel
  GET  /video   MJPEG video stream
  GET  /stream  SSE text stream (MiniCPM observations)
  POST /frame   Push a new frame (called by VLA eval loop)
"""

import asyncio
import base64
import io
import os
import sys
import threading
import time
from typing import List
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI(title="MiniCPM-o-4.5 Monitor")

# ── Global state ─────────────────────────────────────────────────────
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()
_texts: List[dict] = []
_texts_lock = threading.Lock()
_recent_frames: deque = deque(maxlen=4)

# ── MiniCPM client ───────────────────────────────────────────────────
_minicpm_model = None


def _load_minicpm():
    global _minicpm_model
    print("[Server] Loading MiniCPM-o-4.5 (bf16, vision-only)...", file=sys.stderr)
    from transformers import AutoModel
    import torch

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            pass

    kw = {
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda",
        "init_vision": True,
        "init_audio": False,
        "init_tts": False,
    }
    if hf_token:
        kw["token"] = hf_token

    _minicpm_model = AutoModel.from_pretrained("openbmb/MiniCPM-o-4_5", **kw)
    _minicpm_model.eval()
    print("[Server] MiniCPM loaded.", file=sys.stderr)


def _run_minicpm_inference():
    """Called periodically from eval loop via /frame. Runs in background thread."""
    global _minicpm_model, _recent_frames
    if _minicpm_model is None:
        return

    frames = list(_recent_frames)
    if len(frames) < 2:
        return

    pil_images = [Image.fromarray(f).convert("RGB") for f in frames]
    prompt = (
        "You are watching a robot arm execute a task in a simulated kitchen. "
        "These frames are in chronological order. Describe:\n"
        "1. What is the robot currently doing?\n"
        "2. Is it making progress?\n"
        "3. Any issues (stuck, wrong object, shaking)?\n"
        "Be concise — 2-4 sentences."
    )
    msgs = [{"role": "user", "content": pil_images + [prompt]}]

    try:
        import torch
        with torch.inference_mode():
            resp = _minicpm_model.chat(msgs=msgs, sampling=True, temperature=0.1, max_new_tokens=256)
        if resp:
            with _texts_lock:
                _texts.append({"time": time.time(), "text": resp.strip()})
    except Exception as e:
        print(f"[Server] MiniCPM inference error: {e}", file=sys.stderr)


# ── Fonts ────────────────────────────────────────────────────────────

def _get_font(size: int = 13):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                 "C:\\Windows\\Fonts\\consola.ttf"):
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except Exception: pass
    return ImageFont.load_default()


def _draw_overlay(img: np.ndarray, lines: List[str], color: str = "white") -> np.ndarray:
    pil = Image.fromarray(img).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    font = _get_font(13)
    colors = {"white": (255,255,255), "yellow": (255,255,100), "green": (100,255,100), "red": (255,100,100)}
    lh, pad = 16, 6
    bh = len(lines) * lh + pad * 2
    ov = Image.new("RGBA", (pil.width, bh), (0, 0, 0, 180))
    pil.paste(ov, (0, pil.height - bh), ov)
    for i, line in enumerate(lines):
        y = pil.height - bh + pad + i * lh
        draw.text((pad, y), line, fill=colors.get(color, colors["white"]), font=font)
    return np.array(pil.convert("RGB"))


# ── Pages ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("""<!DOCTYPE html><html><head><title>MiniCPM-o-4.5 Monitor</title>
<style>body{margin:0;background:#111;color:#eee;font-family:monospace;display:flex;height:100vh}
#video{flex:3;display:flex;align-items:center;justify-content:center}
#video img{max-width:100%;max-height:100vh}
#panel{flex:1;background:#1a1a1a;overflow-y:auto;padding:12px;border-left:1px solid #333;min-width:300px}
.entry{margin:8px 0;padding:6px 8px;background:#222;border-radius:4px;font-size:13px;line-height:1.5}
.entry .time{color:#888;font-size:11px}</style></head><body>
<div id="video"><img src="/video"/></div>
<div id="panel"><h3 style="margin:0 0 10px;color:#4af">MiniCPM-o-4.5</h3><div id="txt"></div></div>
<script>const s=new EventSource("/stream");s.onmessage=e=>{const d=JSON.parse(e.data);
const div=document.createElement("div");div.className="entry";
div.innerHTML='<span class="time">'+d.time+'</span><br>'+d.text.replace(/\\n/g,"<br>");
document.getElementById("txt").prepend(div)}</script></body></html>""")


@app.get("/video")
async def video():
    async def gen():
        while True:
            with _frame_lock:
                img = _latest_frame.copy() if _latest_frame is not None else None
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
                img = _draw_overlay(img, ["Waiting for LIBERO frames..."], "yellow")
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=75)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
            await asyncio.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/stream")
async def stream():
    async def gen():
        idx = 0
        while True:
            if idx < len(_texts):
                with _texts_lock:
                    new = _texts[idx:]
                    idx = len(_texts)
                for entry in new:
                    t = time.strftime("%H:%M:%S", time.localtime(entry["time"]))
                    import json
                    data = json.dumps({"time": t, "text": entry["text"]})
                    yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/frame")
async def push_frame(request: Request):
    global _latest_frame, _recent_frames
    data = await request.json()
    jpg = base64.b64decode(data["image"])
    img = np.array(Image.open(io.BytesIO(jpg)).convert("RGB"))

    # Status bar
    if data.get("status"):
        img = _draw_overlay(img, [data["status"]], "yellow")

    with _frame_lock:
        _latest_frame = img
    _recent_frames.append(img)

    # Run MiniCPM inference based on step counter
    step = data.get("step", 0)
    interval = data.get("interval", 50)
    if step > 0 and step % interval == 0:
        t = threading.Thread(target=_run_minicpm_inference, daemon=True)
        t.start()

    return {"ok": True}


@app.get("/health")
async def health():
    return {"minicpm_loaded": _minicpm_model is not None, "texts": len(_texts)}


# ── Startup ──────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup_load_minicpm():
    t = threading.Thread(target=_load_minicpm, daemon=True)
    t.start()


if __name__ == "__main__":
    _load_minicpm()
    uvicorn.run(app, host="0.0.0.0", port=8778, log_level="info")
