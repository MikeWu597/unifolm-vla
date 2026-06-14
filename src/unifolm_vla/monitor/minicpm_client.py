"""
MiniCPM-o-4.5 local inference for VLA monitoring.

Loads MiniCPM-o-4.5 INT4 (or bf16) and provides video-frame-based
text generation. Designed to run alongside UnifoLM-VLA on A6000 (48GB).
"""

import time
import threading
from typing import List
import numpy as np
import torch
from PIL import Image


class MiniCPMClient:
    """Local MiniCPM-o-4.5 for real-time VLA observation."""

    def __init__(self, model_id: str = "openbmb/MiniCPM-o-4_5", use_int4: bool = True):
        """
        Args:
            model_id: HuggingFace model id or local path
            use_int4: load INT4 quantized version (recommended for 48GB)
        """
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.use_int4 = use_int4

    def load(self, token: str | None = None):
        """Load the model (call once, blocks until loaded)."""
        import os, sys
        from transformers import AutoModel, AutoProcessor

        # HF token: env var or manual
        hf_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            # Try huggingface-cli login cache
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()

        print(f"[MiniCPM] Loading {self.model_id} (INT4={self.use_int4})...", file=sys.stderr)
        t0 = time.time()

        load_kwargs = {
            "attn_implementation": "sdpa",
            "trust_remote_code": True,
            "device_map": "cuda",
        }
        if hf_token:
            load_kwargs["token"] = hf_token
        if self.use_int4:
            load_kwargs["load_in_4bit"] = True
            load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        # Patch: MiniCPM-o's Resampler lacks _initialize_weights (required by transformers 4.52)
        import torch.nn as nn
        if not hasattr(nn.Module, '_initialize_weights'):
            nn.Module._initialize_weights = lambda self, m: None

        self.model = AutoModel.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

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
            max_new_tokens: generation limit
            temperature: sampling temperature
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Convert numpy arrays to PIL
        pil_images = [Image.fromarray(img).convert("RGB") for img in images]

        # MiniCPM-o uses a chat format with images + text
        msgs = [{"role": "user", "content": pil_images + [prompt]}]

        try:
            response = self.model.chat(
                tokenizer=self.processor,
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


# ── Global singleton for use by eval loop ────────────────────────────

_client: MiniCPMClient | None = None
_model_loaded = threading.Event()


def get_client(model_id: str = "openbmb/MiniCPM-o-4_5", use_int4: bool = True) -> MiniCPMClient:
    global _client
    if _client is None:
        _client = MiniCPMClient(model_id=model_id, use_int4=use_int4)
    return _client


def load_in_background(model_id: str = "openbmb/MiniCPM-o-4_5", use_int4: bool = True):
    """Load MiniCPM in a background thread (non-blocking)."""
    def _load():
        c = get_client(model_id, use_int4)
        c.load()
        _model_loaded.set()
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    return t


def wait_loaded(timeout: float = 300):
    """Wait for the background load to complete."""
    if not _model_loaded.wait(timeout=timeout):
        raise TimeoutError("MiniCPM model failed to load within timeout")
