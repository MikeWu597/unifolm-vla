"""
UnifoLM-VLA: dual-head Vision-Language-Action model.

Dual-model architecture (QwenSupervised):
  VLA Model (fine-tuned):  Qwen2.5-VL → FlowmatchingActionHead → actions
  Agent Model (vanilla):   Qwen2.5-VL-Instruct → lm_head → text/tool_calls

Two separate VLM instances — the VLA backbone was fine-tuned on robot data,
shifting its hidden states away from the original lm_head distribution.
Agent text generation uses a fresh Qwen2.5-VL-Instruct with intact lm_head.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from unifolm_vla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.model.modules.vlm import get_vlm_model
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model, FlowmatchingActionHead
from unifolm_vla.model.tools import FRAMEWORK_REGISTRY


def _load_agent_vlm(agent_model_id: str = None):
    """Load a VANILLA Qwen2.5-VL for agent text generation (no robot fine-tuning).

    Model selection order:
      1. $AGENT_VLM_PATH  environment variable (local path or HF model ID)
      2. agent_model_id    function argument
      3. "Qwen/Qwen2.5-VL-3B-Instruct" (auto-download from HuggingFace)
    """
    import os, sys
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    if agent_model_id is None:
        agent_model_id = os.environ.get(
            "AGENT_VLM_PATH", "Qwen/Qwen2.5-VL-3B-Instruct"
        )

    print(f"[AgentVLM] Loading vanilla {agent_model_id} (sdpa, bf16)...", file=sys.stderr)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        agent_model_id,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(agent_model_id)
    processor.tokenizer.padding_side = "left"
    print(f"[AgentVLM] Loaded OK", file=sys.stderr)
    return model, processor


@FRAMEWORK_REGISTRY.register("unifolm_vla")
class Unifolm_VLA(baseframework):
    """
    Multimodal vision-language-action model with separate agent VLM.

    Two independent Qwen2.5-VL backbones:
      self.qwen_vl_interface  — fine-tuned, drives predict_action() (VLA Head)
      self.agent_model        — vanilla,  drives predict_text()   (LLM Head)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        agent_model_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # ── VLA backbone (fine-tuned) ──────────────────────────────
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )
        self.processor = self.qwen_vl_interface.processor
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # ── Agent backbone (vanilla, no robot fine-tuning) ─────────
        # Default: same model ID as VLA base, but loaded fresh from original weights.
        # Override via config.framework.agent_vlm.base_vlm or agent_model_id arg.
        if agent_model_id is None and config is not None:
            agent_cfg = config.framework.get("agent_vlm", {})
            agent_model_id = agent_cfg.get("base_vlm", "Qwen/Qwen2.5-VL-3B-Instruct")
        elif agent_model_id is None:
            agent_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

        self.agent_model, self.agent_processor = _load_agent_vlm(agent_model_id)

    # ------------------------------------------------------------------
    # VLA Head: action prediction (unchanged)
    # ------------------------------------------------------------------

    def forward(
        self,
        qwen_inputs: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        actions = qwen_inputs["action"].to(torch.bfloat16)
        state = qwen_inputs["state"].to(torch.bfloat16)
        state = state.unsqueeze(1) if state.dim() == 2 else state

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = (
                state.repeat(repeated_diffusion_steps, 1, 1)
                if state is not None else None
            )
            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated,
            )
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, qwen_inputs, **kwargs: str) -> np.ndarray:
        state = qwen_inputs["state"]
        state = state.unsqueeze(1) if state.dim() == 2 else state

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state = state.to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    # ------------------------------------------------------------------
    # LLM Head: agent text generation (uses vanilla agent VLM)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_text(
        self,
        qwen_inputs,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs,
    ) -> str:
        """
        Agent text generation using the VANILLA Qwen2.5-VL-Instruct.

        The fine-tuned VLA backbone has drifted hidden states that break the
        original lm_head. We use a separate, fresh Qwen2.5-VL whose lm_head
        is intact, ensuring coherent text/tool_call output.
        """
        model = self.agent_model
        tokenizer = self.agent_processor.tokenizer
        device = qwen_inputs["input_ids"].device
        eos_id = tokenizer.eos_token_id

        input_ids = qwen_inputs["input_ids"]
        attn_mask = qwen_inputs["attention_mask"]
        pix_vals = qwen_inputs["pixel_values"]
        grid_thw = qwen_inputs["image_grid_thw"]

        generated_ids = []
        past_kv = None

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for step in range(max_new_tokens):
                is_first = (step == 0)

                outputs = model(
                    input_ids=input_ids if is_first else input_ids[:, -1:],
                    attention_mask=attn_mask,
                    pixel_values=pix_vals if is_first else None,
                    image_grid_thw=grid_thw if is_first else None,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                )

                past_kv = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                if do_sample and temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                token_id = next_id.item()
                generated_ids.append(token_id)

                if token_id == eos_id:
                    break

                attn_mask = torch.cat(
                    [attn_mask, torch.ones(1, 1, device=device, dtype=attn_mask.dtype)], dim=1
                )

        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text
