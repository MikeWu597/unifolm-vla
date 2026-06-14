"""
Full-duplex MiniCPM-o-4.5 monitor server.

Every incoming frame → streaming_prefill (continuous vision input).
Background thread → streaming_generate loop (continuous text output).
The model decides when to comment — no fixed interval.

Runs from minicpm conda env. VLA posts frames from unifolm-vla env.
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
import queue

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI(title="MiniCPM-o-4.5 Full-Duplex Monitor")

# ── Global state ─────────────────────────────────────────────────────
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()
_texts: List[dict] = []
_texts_lock = threading.Lock()
_frame_queue: queue.Queue = queue.Queue()  # frames → duplex model
_model_loaded = threading.Event()
_model = None


# ── MiniCPM duplex ───────────────────────────────────────────────────

def _load_minicpm():
    global _model
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

    m = AutoModel.from_pretrained("openbmb/MiniCPM-o-4_5", **kw)
    m.eval()

    # Convert to full-duplex mode
    m = m.as_duplex()
    m.prepare(
        prefix_system_prompt=(
            "You are watching a robot arm execute tasks in a simulated kitchen. "
            "Look at the live video feed and comment on what you see. "
            "Describe the robot's actions, whether it's making progress, "
            "and any problems (stuck, wrong object, shaking, dropped item). "
            "Speak only when you have a meaningful observation — don't repeat yourself."
        ),
    )
    _model = m
    print("[Server] MiniCPM duplex ready.", file=sys.stderr)
    _model_loaded.set()


def _duplex_generate_loop():
    """Continuous generation thread — model decides when to speak."""
    if not _model_loaded.wait(timeout=600):
        print("[Server] Model load timeout!", file=sys.stderr)
        return

    m = _model
    print("[Server] Starting duplex generate loop...", file=sys.stderr)

    while True:
        try:
            import torch
            with torch.inference_mode():
                result = m.streaming_generate(
                    max_new_speak_tokens_per_chunk=30,
                    decode_mode="sampling",
                )
            if result is None:
                time.sleep(0.1)
                continue

            is_listen = result.get("is_listen", True)
            text = (result.get("text") or "").strip()

            if not is_listen and text:
                with _texts_lock:
                    _texts.append({"time": time.time(), "text": text})
                print(f"[MiniCPM] {text[:120]}", file=sys.stderr)
        except Exception as e:
            print(f"[Server] Generate loop error: {e}", file=sys.stderr)
            time.sleep(1)


def _duplex_feed_loop():
    """Feed incoming frames to the duplex model."""
    if not _model_loaded.wait(timeout=600):
        return

    m = _model
    print("[Server] Starting duplex feed loop...", file=sys.stderr)

    while True:
        try:
            frame = _frame_queue.get(timeout=1)
            pil = Image.fromarray(frame).convert("RGB")
            import torch
            with torch.inference_mode():
                m.streaming_prefill(
                    frame_list=[pil],
                    max_slice_nums=1,
                    batch_vision_feed=False,
                )
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Server] Feed error: {e}", file=sys.stderr)


# ── Startup ──────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup():
    # Load model in background
    t_load = threading.Thread(target=_load_minicpm, daemon=True)
    t_load.start()

    # Wait for model, then start duplex loops
    def _start_loops():
        _model_loaded.wait()
        threading.Thread(target=_duplex_generate_loop, daemon=True).start()
        threading.Thread(target=_duplex_feed_loop, daemon=True).start()

    threading.Thread(target=_start_loops, daemon=True).start()


# ── Fonts ────────────────────────────────────────────────────────────

def _get_font(size: int = 13):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
              "C:\\Windows\\Fonts\\consola.ttf"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except Exception: pass
    return ImageFont.load_default()


def _draw_overlay(img, lines, color="white"):
    pil = Image.fromarray(img).convert("RGBA")
    d = ImageDraw.Draw(pil)
    f = _get_font(13)
    colors = {"white": (255,255,255), "yellow": (255,255,100), "green": (100,255,100), "red": (255,100,100)}
    lh, pad = 16, 6
    bh = len(lines) * lh + pad * 2
    ov = Image.new("RGBA", (pil.width, bh), (0, 0, 0, 180))
    pil.paste(ov, (0, pil.height - bh), ov)
    for i, line in enumerate(lines):
        y = pil.height - bh + pad + i * lh
        d.text((pad, y), line, fill=colors.get(color, colors["white"]), font=f)
    return np.array(pil.convert("RGB"))


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("""<!DOCTYPE html><html><head><title>MiniCPM-o-4.5 Full-Duplex</title>
<style>body{margin:0;background:#111;color:#eee;font-family:monospace;display:flex;height:100vh}
#video{flex:3;display:flex;align-items:center;justify-content:center}
#video img{max-width:100%;max-height:100vh}
#panel{flex:1;background:#1a1a1a;overflow-y:auto;padding:12px;border-left:1px solid #333;min-width:300px}
.entry{margin:8px 0;padding:6px 8px;background:#222;border-radius:4px;font-size:13px;line-height:1.5}
.entry .time{color:#888;font-size:11px}</style></head><body>
<div id="video"><img src="/video"/></div>
<div id="panel"><h3 style="margin:0 0 10px;color:#4af">MiniCPM-o-4.5 Live</h3><div id="txt"></div></div>
<script>let c=0;setInterval(async()=>{try{const r=await fetch("/texts?since="+c);
const d=await r.json();d.texts.forEach(t=>{const div=document.createElement("div");
div.className="entry";div.innerHTML='<span class="time">'+t.time+'</span><br>'
+t.text.replace(/\\n/g,"<br>");document.getElementById("txt").prepend(div)});
c=d.total}catch(e){}},1500)</script></body></html>""")


@app.get("/video")
async def video():
    async def gen():
        while True:
            with _frame_lock:
                img = _latest_frame.copy() if _latest_frame is not None else None
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
                img = _draw_overlay(img, ["Waiting for frames..."], "yellow")
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=75)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
            await asyncio.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/texts")
async def texts(since: int = 0):
    import json
    with _texts_lock:
        all_texts = [{"time": time.strftime("%H:%M:%S", time.localtime(e["time"])),
                       "text": e["text"]} for e in _texts[since:]]
    return {"texts": all_texts, "total": len(_texts)}


@app.post("/frame")
async def push_frame(request: Request):
    global _latest_frame
    data = await request.json()
    jpg = base64.b64decode(data["image"])
    img = np.array(Image.open(io.BytesIO(jpg)).convert("RGB"))

    if data.get("status"):
        img = _draw_overlay(img, [data["status"]], "yellow")

    with _frame_lock:
        _latest_frame = img

    # Feed to duplex model immediately
    _frame_queue.put(img)

    return {"ok": True}


@app.get("/health")
async def health():
    return {"minicpm_loaded": _model_loaded.is_set(), "texts": len(_texts),
            "queue_size": _frame_queue.qsize()}
