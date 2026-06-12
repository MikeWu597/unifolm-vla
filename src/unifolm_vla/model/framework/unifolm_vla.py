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

        Manual greedy decode with KV cache. Uses the SAME forward path as
        predict_action() to ensure images are properly embedded via the
        Qwen2.5-VL vision encoder.
        """
        import torch.nn.functional as F

        model = self.qwen_vl_interface.model
        tokenizer = self.processor.tokenizer
        device = qwen_inputs["input_ids"].device
        eos_id = tokenizer.eos_token_id

        input_ids = qwen_inputs["input_ids"]          # full prompt with image tokens
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
                logits = outputs.logits[:, -1, :]  # [1, vocab]

                if do_sample and temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

                token_id = next_id.item()
                generated_ids.append(token_id)

                if token_id == eos_id:
                    break

                # Extend attention mask for the new token
                attn_mask = torch.cat(
                    [attn_mask, torch.ones(1, 1, device=device, dtype=attn_mask.dtype)], dim=1
                )

        text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        return text

