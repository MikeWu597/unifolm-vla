from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from unifolm_vla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.model.modules.vlm import get_vlm_model
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model, FlowmatchingActionHead
from unifolm_vla.model.tools import FRAMEWORK_REGISTRY

@FRAMEWORK_REGISTRY.register("unifolm_vla")
class Unifolm_VLA(baseframework):
    """
    Multimodal vision-language-action model.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        self.processor = self.qwen_vl_interface.processor
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  
    
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
                output_hidden_states = True,       
                return_dict=True,  
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            
        with torch.autocast("cuda", dtype=torch.float32):
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
            action_loss = self.action_model(
                last_hidden_repeated, 
                actions_target_repeated, 
                state_repeated,
            )
            
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        qwen_inputs,
        **kwargs: str,
    ) -> np.ndarray:

        state = qwen_inputs["state"]
        state = state.unsqueeze(1) if state.dim() == 2 else state

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states = True,       
                return_dict=True,  
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            
        state = state.to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, 
                state,
            )
        normalized_actions = pred_actions.detach().cpu().numpy()
        
        return {"normalized_actions": normalized_actions}

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
        LLM Head: autoregressive text generation for agent reasoning and tool calls.

        Reuses the Qwen2.5-VL backbone's built-in lm_head (nn.Linear(3584, vocab_size))
        via model.generate(). This is the second head — independent from predict_action().

        Args:
            qwen_inputs: dict with input_ids, attention_mask, pixel_values, image_grid_thw
            max_new_tokens: max tokens to generate (default 256)
            do_sample: whether to sample (False = greedy)
            temperature: sampling temperature
            **kwargs: passed to model.generate()

        Returns:
            str: decoded text output (may contain <tool_call>...</tool_call> blocks)
        """
        # Route to the Qwen model's native generate (lm_head is built-in)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated_ids = self.qwen_vl_interface.model.generate(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs.get("pixel_values", None),
                image_grid_thw=qwen_inputs.get("image_grid_thw", None),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **kwargs,
            )

        # Decode only the newly generated tokens (skip input prompt)
        input_len = qwen_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        text = self.processor.decode(new_tokens, skip_special_tokens=False)
        return text

