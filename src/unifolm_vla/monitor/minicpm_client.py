"""
MiniCPM-o-4.5 local inference for VLA monitoring.

Loads MiniCPM-o-4.5 (vision only, no audio/TTS) and provides
video-frame-based text generation. Runs alongside UnifoLM-VLA on A6000.
"""

import time
import threading
from typing import List
import numpy as np
import torch
from PIL import Image


class MiniCPMClient:
    """Local MiniCPM-o-4.5 (vision-only) for real-time VLA observation."""

    def __init__(self, model_id: str = "openbmb/MiniCPM-o-4_5"):
        self.model_id = model_id
        self.model = None

    def load(self, token: str | None = None):
        """Load the model (call once, blocks until loaded)."""
        import os, sys
        from transformers import AutoModel

        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()

        print(f"[MiniCPM] Loading {self.model_id} (bf16, vision-only)...", file=sys.stderr)
        t0 = time.time()

        load_kwargs = {
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": "cuda",
            "init_vision": True,
            "init_audio": False,
            "init_tts": False,
        }
        if hf_token:
            load_kwargs["token"] = hf_token

        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()

        dt = time.time() - t0
        print(f"[MiniCPM] Loaded in {dt:.1f}s", file=sys.stderr)

    @torch.inference_mode()
    def chat(self, images: List[np.ndarray], prompt: str,
             max_new_tokens: int = 512, temperature: float = 0.1) -> str:
        """
        Send images + prompt to MiniCPM, return text response.

        Args:
            images: list of numpy RGB uint8 images
            prompt: text prompt for the model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        pil_images = [Image.fromarray(img).convert("RGB") for img in images]
        msgs = [{"role": "user", "content": pil_images + [prompt]}]

        try:
            response = self.model.chat(
                msgs=msgs,
                sampling=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            return response or ""
        except Exception as e:
            import sys
            print(f"[MiniCPM] Inference error: {e}", file=sys.stderr)
            return ""


# ── Global singleton ──────────────────────────────────────────────

_client: MiniCPMClient | None = None
_model_loaded = threading.Event()


def get_client(model_id: str = "openbmb/MiniCPM-o-4_5") -> MiniCPMClient:
    global _client
    if _client is None:
        _client = MiniCPMClient(model_id=model_id)
    return _client


def load_in_background(model_id: str = "openbmb/MiniCPM-o-4_5"):
    """Load MiniCPM in a background thread (non-blocking)."""
    def _load():
        c = get_client(model_id)
        c.load()
        _model_loaded.set()
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    return t


def wait_loaded(timeout: float = 300):
    """Wait for the background load to complete."""
    if not _model_loaded.wait(timeout=timeout):
        raise TimeoutError("MiniCPM model failed to load within timeout")
