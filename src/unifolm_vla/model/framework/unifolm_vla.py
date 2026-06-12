"""
UnifoLM-VLA: Vision-Language-Action model with agent API integration.

Dual-mode architecture:
  VLA Head (local):  fine-tuned Qwen2.5-VL → FlowmatchingActionHead → actions
  Agent (API):       Bailian Qwen-VL API → text/tool_calls

The VLA backbone was fine-tuned on robot data, shifting its hidden states
away from the original lm_head. Agent text generation uses an external
Qwen-VL API which has an intact lm_head.
"""

import os
import io
import base64
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from unifolm_vla.training.trainer_utils import initialize_overwatch
logger = initialize_overwatch(__name__)

from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.model.modules.vlm import get_vlm_model
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model, FlowmatchingActionHead
from unifolm_vla.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("unifolm_vla")
class Unifolm_VLA(baseframework):
    """
    VLA model with API-based agent text generation.

    VLA Head (local GPU):  Qwen2.5-VL → ActionHead → 7D actions
    Agent (Bailian API):   Qwen-VL via DashScope → text with tool_calls
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        agent_model_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # ── VLA backbone (fine-tuned, local GPU) ───────────────────
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.processor = self.qwen_vl_interface.processor
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # ── Agent: Bailian Qwen-VL API ─────────────────────────────
        # Model: qwen3-vl-plus / qwen-vl-plus / qwen2.5-vl-72b-instruct
        # Secrets: DASHSCOPE_API_KEY
        if agent_model_id is None and config is not None:
            agent_cfg = config.framework.get("agent_vlm", {})
            agent_model_id = agent_cfg.get("base_vlm", "qwen3-vl-plus")
        elif agent_model_id is None:
            agent_model_id = os.environ.get("AGENT_VLM_MODEL", "qwen3-vl-plus")

        self.agent_model_id = agent_model_id
        self._agent_client = None  # lazy init

    @property
    def agent_client(self):
        """Lazy-init OpenAI-compatible client for Bailian."""
        if self._agent_client is None:
            from openai import OpenAI
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            base_url = os.environ.get(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            self._agent_client = OpenAI(api_key=api_key, base_url=base_url)
        return self._agent_client

    # ------------------------------------------------------------------
    # VLA Head: action prediction (unchanged)
    # ------------------------------------------------------------------

    def forward(self, qwen_inputs=None, **kwargs):
        actions = qwen_inputs["action"].to(torch.bfloat16)
        state = qwen_inputs["state"].to(torch.bfloat16)
        state = state.unsqueeze(1) if state.dim() == 2 else state

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states=True, return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            rd = self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            action_loss = self.action_model(
                last_hidden.repeat(rd, 1, 1),
                actions.repeat(rd, 1, 1),
                state.repeat(rd, 1, 1) if state is not None else None,
            )
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, qwen_inputs, **kwargs):
        state = qwen_inputs["state"]
        state = state.unsqueeze(1) if state.dim() == 2 else state

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states=True, return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state = state.to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    # Agent: Bailian Qwen-VL API
    # ------------------------------------------------------------------

    @staticmethod
    def _image_to_base64(img: np.ndarray) -> str:
        """Convert numpy RGB image to base64 data URI."""
        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img).convert("RGB")
        else:
            pil = img.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def predict_text(
        self,
        images: List[np.ndarray],
        prompt_text: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        history: Optional[List[dict]] = None,
    ) -> Tuple[str, List[dict]]:
        """
        Call Bailian Qwen-VL API with multi-turn conversation history.

        Args:
            images: list of numpy RGB images (H, W, 3) uint8
            prompt_text: agent monitoring prompt
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            history: previous messages (text-only). Modified in-place.

        Returns:
            (response_text, updated_history) tuple.
        """
        # Build current user message with images
        content = []
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_base64(img)},
            })
        content.append({"type": "text", "text": prompt_text})

        user_msg = {"role": "user", "content": content}

        # Build full message list: history (text-only) + current (with images)
        messages = (history or []) + [user_msg]

        try:
            response = self.agent_client.chat.completions.create(
                model=self.agent_model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            reply = response.choices[0].message.content or ""

            # Update history: add user text summary + assistant reply (text-only, no images)
            new_history = (history or []) + [
                {"role": "user", "content": [{"type": "text", "text": prompt_text[-500:]}]},
                {"role": "assistant", "content": reply},
            ]
            return reply, new_history
        except Exception as e:
            import sys
            print(f"[Agent API] Error: {e}", file=sys.stderr)
            return "", (history or [])
