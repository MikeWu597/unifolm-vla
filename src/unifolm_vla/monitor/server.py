"""
HTTP server for real-time LIBERO video stream + MiniCPM text output.

Endpoints:
  GET  /        HTML page with video player + text panel
  GET  /video   MJPEG video stream (512×512, 30fps)
  GET  /stream  SSE text stream (MiniCPM observations)
  POST /frame   Push a new frame (called by eval loop)
  POST /text    Push MiniCPM text (called by monitor client)
"""

import asyncio
import time
import io
import threading
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI(title="UnifoLM-VLA Monitor")

# ── Global state ─────────────────────────────────────────────────────
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()

_texts: List[dict] = []  # [{"time": float, "text": str}, ...]
_texts_lock = threading.Lock()
_new_text_event = asyncio.Event()  # fires when new text arrives


def _get_font(size: int = 13):
    import os
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
                 "C:\\Windows\\Fonts\\consola.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_overlay(img: np.ndarray, lines: List[str], color: str = "white") -> np.ndarray:
    """Draw semi-transparent text bar at the bottom of an image."""
    pil = Image.fromarray(img).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    font = _get_font(13)
    colors = {"white": (255, 255, 255), "yellow": (255, 255, 100),
              "green": (100, 255, 100), "red": (255, 100, 100)}
    line_h, pad = 16, 6
    bar_h = len(lines) * line_h + pad * 2
    overlay = Image.new("RGBA", (pil.width, bar_h), (0, 0, 0, 180))
    pil.paste(overlay, (0, pil.height - bar_h), overlay)
    for i, line in enumerate(lines):
        y = pil.height - bar_h + pad + i * line_h
        draw.text((pad, y), line, fill=colors.get(color, colors["white"]), font=font)
    return np.array(pil.convert("RGB"))


# ── Pages ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head><title>UnifoLM-VLA Monitor</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family:monospace; display:flex; height:100vh; }
  #video { flex:3; display:flex; align-items:center; justify-content:center; }
  #video img { max-width:100%; max-height:100vh; }
  #panel { flex:1; background:#1a1a1a; overflow-y:auto; padding:12px; border-left:1px solid #333; min-width:300px; }
  .entry { margin:8px 0; padding:6px 8px; background:#222; border-radius:4px; font-size:13px; line-height:1.5; }
  .entry .time { color:#888; font-size:11px; }
</style></head>
<body>
<div id="video"><img src="/video" /></div>
<div id="panel"><h3 style="margin:0 0 10px;color:#4af;">MiniCPM-o-4.5</h3><div id="txt"></div></div>
<script>
const src = new EventSource("/stream");
src.onmessage = e => {
    const d = JSON.parse(e.data);
    const div = document.createElement("div");
    div.className = "entry";
    div.innerHTML = '<span class="time">'+d.time+'</span><br>'+d.text.replace(/\\n/g,"<br>");
    document.getElementById("txt").prepend(div);
};
</script>
</body></html>""")


# ── Video stream (MJPEG) ─────────────────────────────────────────────

@app.get("/video")
async def video():
    async def generate():
        while True:
            with _frame_lock:
                img = _latest_frame.copy() if _latest_frame is not None else None
            if img is None:
                # Return a black placeholder frame
                img = np.zeros((512, 512, 3), dtype=np.uint8)
                img = draw_overlay(img, ["Waiting for first frame..."], "yellow")
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=75)
            frame = buf.getvalue()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            await asyncio.sleep(0.033)  # ~30fps
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Text stream (SSE) ────────────────────────────────────────────────

@app.get("/stream")
async def stream():
    async def generate():
        idx = len(_texts)
        while True:
            if idx < len(_texts):
                with _texts_lock:
                    new = _texts[idx:]
                    idx = len(_texts)
                for entry in new:
                    t = time.strftime("%H:%M:%S", time.localtime(entry["time"]))
                    data = '{"time":"' + t + '","text":' + __import__("json").dumps(entry["text"]) + '}'
                    yield f"data: {data}\n\n"
            await asyncio.sleep(0.5)
    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Push endpoints ────────────────────────────────────────────────────

@app.post("/frame")
async def push_frame(request: Request):
    """Receive a frame (JSON with base64 JPEG) from the eval loop."""
    global _latest_frame
    data = await request.json()
    import base64
    jpg = base64.b64decode(data["image"])
    img = np.array(Image.open(io.BytesIO(jpg)).convert("RGB"))
    if data.get("status"):
        img = draw_overlay(img, [data["status"]], "yellow")
    with _frame_lock:
        _latest_frame = img
    return {"ok": True}


@app.post("/text")
async def push_text(request: Request):
    """Receive MiniCPM observation text."""
    data = await request.json()
    with _texts_lock:
        _texts.append({"time": time.time(), "text": data["text"]})
    return {"ok": True}


def start_server(host: str = "0.0.0.0", port: int = 8778):
    """Start the HTTP server in the current thread (blocking)."""
    uvicorn.run(app, host=host, port=port, log_level="info")


def start_server_thread(host: str = "0.0.0.0", port: int = 8778):
    """Start the HTTP server in a daemon thread."""
    t = threading.Thread(target=start_server, args=(host, port), daemon=True)
    t.start()
    return t
