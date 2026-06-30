# UnifoLM-VLA-Panel

Browser-controlled VLA demo with LIBERO tabletop simulation.

## Architecture

```
Browser (http://localhost:8778)
    │  POST /cmd   "pick up the bowl..."
    │  GET  /video  MJPEG stream
    │  GET  /status JSON state
    ▼
FastAPI Server  ◄──  UnifoLM-VLA  ◄──  LIBERO OffScreenRenderEnv
                                             Panda arm + tabletop
```

## Requirements

- GPU: RTX 4090 (24GB) or better
- OS: Ubuntu 22.04
- Python: 3.10

## Quick Start (AutoDL)

```bash
# 1. Clone
git clone https://github.com/MikeWu597/unifolm-vla.git
cd unifolm-vla && git checkout UnifoLM-VLA-Panel

# 2. Environment (Python 3.10 required)
conda create -n vla python==3.10.18 -y
conda init bash && source ~/.bashrc && conda activate vla

# 3. PyTorch
pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. VLA inference deps
pip install transformers==4.52.3 accelerate==1.5.2 diffusers==0.35.1 qwen-vl-utils
pip install tiktoken einops omegaconf pydantic pillow "numpy<2" tyro scipy matplotlib
pip install huggingface_hub fastapi uvicorn

# 5. Simulation (robosuite + mujoco, no LIBERO needed)
pip install mujoco==3.3.5 robosuite==1.4.1 robosuite_models

# 6. Register package
pip install -e . --no-deps

# 7. Headless rendering
apt-get install -y libosmesa6-dev libegl1 libegl-dev

# 8. Download weights (~35GB, use mirror for China)
python download_weights.py --output-dir ./models --use-mirror

# 9. Run
unset OMP_NUM_THREADS
export MUJOCO_GL=egl
python examples/panel_demo.py \
    --ckpt ./models/UnifoLM-VLA-LIBERO/checkpoints/pytorch_model.pt \
    --vlm ./models/UnifoLM-VLM-Base \
    --port 8778
```

## Access

### AutoDL
1. Console → Custom Service → Add port 8778 → Get public URL

### SSH tunnel
```bash
ssh -L 8778:127.0.0.1:8778 -p <port> root@<host>
# Open http://localhost:8778
```

## Usage

1. Open browser → see robot arm + tabletop scene
2. Type instruction (e.g. "pick up the block and place it on the plate")
3. Click Execute → VLA runs → watch real-time video
4. Click Stop or type new instruction to switch tasks
