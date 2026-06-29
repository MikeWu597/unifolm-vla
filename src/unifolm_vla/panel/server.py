"""
FastAPI server for VLA Panel — MJPEG video + command input.
"""

import asyncio
import io
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI(title="UnifoLM-VLA Panel")

# ── Global state ─────────────────────────────────────────────────────
_latest_frame: np.ndarray | None = None
_frame_lock = threading.Lock()
_status = {"running": False, "instruction": "", "step": 0, "active_since": 0}
_status_lock = threading.Lock()
_cmd_queue: list = []  # pending user commands


def set_status(**kw):
    with _status_lock:
        _status.update(kw)


def get_status() -> dict:
    with _status_lock:
        return dict(_status)


def push_frame(img: np.ndarray):
    global _latest_frame
    with _frame_lock:
        _latest_frame = img.copy()


def next_command() -> str | None:
    if _cmd_queue:
        return _cmd_queue.pop(0)
    return None


# ── HTML ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>UnifoLM-VLA Panel</title>
<style>
body{margin:0;background:#111;color:#eee;font-family:monospace;display:flex;height:100vh}
#video{flex:3;display:flex;align-items:center;justify-content:center;background:#000}
#video img{max-width:100%;max-height:100vh}
#ctrl{flex:1;background:#1a1a1a;padding:16px;display:flex;flex-direction:column;min-width:300px}
#status{padding:8px 12px;border-radius:4px;margin-bottom:12px;font-size:14px}
.status-idle{background:#333;color:#888}
.status-running{background:#153015;color:#4a4}
#instr{width:100%;padding:10px;font-size:14px;background:#222;color:#eee;border:1px solid #444;
  border-radius:4px;margin-bottom:8px;font-family:monospace;resize:vertical}
#instr:focus{outline:none;border-color:#4af}
.btn{padding:10px 16px;font-size:14px;border:none;border-radius:4px;cursor:pointer;font-family:monospace;margin-bottom:8px}
.btn-go{background:#294;color:#fff}
.btn-go:hover{background:#3a6}
.btn-stop{background:#922;color:#fff}
.btn-stop:hover{background:#b33}
#info{font-size:12px;color:#888;margin-top:auto}
</style></head><body>
<div id="video"><img src="/video"/></div>
<div id="ctrl">
<div id="status" class="status-idle">Idle — enter instruction below</div>
<input id="instr" placeholder="e.g. pick up the block and place it on the plate" autofocus>
<button class="btn btn-go" onclick="go()">Execute</button>
<button class="btn btn-stop" onclick="stop()">Stop</button>
<button class="btn btn-stop" onclick="reset_()" style="background:#444">Reset Arm</button>
<div id="info">UnifoLM-VLA-Panel | robosuite tabletop</div>
</div>
<script>
async function go(){const t=document.getElementById("instr").value;if(!t)return;
document.getElementById("status").className="status-running";document.getElementById("status").innerText="Executing: "+t;
await fetch("/cmd",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({instruction:t})})}
async function stop(){await fetch("/cmd",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({instruction:"__STOP__"})});
document.getElementById("status").className="status-idle";document.getElementById("status").innerText="Idle"}
async function reset_(){await fetch("/cmd",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({instruction:"__RESET__"})})}
setInterval(async()=>{try{const r=await fetch("/status");const s=await r.json();
if(!s.running){document.getElementById("status").className="status-idle";
document.getElementById("status").innerText="Idle — enter instruction below"}
else{document.getElementById("status").className="status-running";
document.getElementById("status").innerText="["+s.step+" steps] "+s.instruction}}catch(e){}},1500)
</script></body></html>""")


# ── Video ────────────────────────────────────────────────────────────

@app.get("/video")
async def video():
    async def gen():
        while True:
            with _frame_lock:
                img = _latest_frame.copy() if _latest_frame is not None else None
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=75)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n"
            await asyncio.sleep(0.033)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── API ──────────────────────────────────────────────────────────────

@app.post("/cmd")
async def post_cmd(request: Request):
    data = await request.json()
    inst = data.get("instruction", "").strip()
    _cmd_queue.append(inst)
    if inst == "__STOP__":
        set_status(running=False, instruction="")
    else:
        set_status(running=True, instruction=inst, step=0, active_since=time.time())
    return {"ok": True}


@app.get("/status")
async def get_status_route():
    return get_status()
