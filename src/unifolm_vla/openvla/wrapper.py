"""
OpenVLA-OFT wrapper — same 7D action space as UnifoLM, stronger generalization.

Model: moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
- 970k demo backbone + LIBERO fine-tuning
- unnorm_key="libero_spatial_no_noops" (same as UnifoLM)
"""

import torch
import numpy as np
from PIL import Image


class OpenVLAWrapper:
    def __init__(self, model_id: str = "openvla/openvla-7b"):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda").eval()

    @torch.inference_mode()
    def predict_action(self, image: Image.Image, instruction: str) -> np.ndarray:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to("cuda", dtype=torch.bfloat16)
        return self.model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",
            do_sample=False,
        )
