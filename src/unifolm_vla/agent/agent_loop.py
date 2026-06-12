"""
Agent loop: orchestrates VLA Head (action) and LLM Head (reasoning).

Workflow:
  1. VLA Head executes `vla_interval` steps of action inference
  2. LLM Head checks progress: continue / new_instruction / terminate
  3. Repeat until episode ends or agent terminates

This is a lightweight agent — no training required. The LLM Head uses the
Qwen2.5-VL's built-in lm_head for zero-shot reasoning and tool calling.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from unifolm_vla.agent.tool_call_parser import ToolCallParser
from unifolm_vla.agent.tools import TOOL_REGISTRY, ToolSpec

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    """Result of an agent check."""
    action: str  # "continue", "new_instruction", or "terminate"
    reason: str = ""
    instruction: Optional[str] = None
    success: bool = False
    max_vla_steps: int = 50  # vla steps until next agent check


class ToolCallAgent:
    """
    Lightweight agent that alternates between VLA execution and LLM reasoning.

    The agent does NOT modify model weights. It orchestrates two inference
    paths on the same Qwen2.5-VL backbone:
      - predict_action()  → VLA Head  (FlowmatchingActionHead)
      - predict_text()    → LLM Head  (lm_head)

    Parameters
    ----------
    model:
        Unifolm_VLA instance with both predict_action() and predict_text().
    vla_interval:
        Number of VLA action steps between agent checks. Default 10.
    max_agent_rounds:
        Maximum number of agent check rounds per episode.
    """

    def __init__(
        self,
        model: Any,
        vla_interval: int = 10,
        max_agent_rounds: int = 30,
    ):
        self.model = model
        self.vla_interval = vla_interval
        self.max_agent_rounds = max_agent_rounds

        # Episode-level state
        self.agent_round: int = 0
        self.obs_history: deque = deque(maxlen=vla_interval * 3)
        self.action_history: deque = deque(maxlen=vla_interval * 3)

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.agent_round = 0
        self.obs_history.clear()
        self.action_history.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_progress(
        self,
        qwen_inputs: Dict[str, Any],
        task_description: str,
        step_counter: int,
        total_steps: int,
    ) -> AgentDecision:
        """
        Run the LLM Head to check task progress and decide next action.

        Args:
            qwen_inputs: Preprocessed multimodal inputs for Qwen2.5-VL.
            task_description: Current task instruction.
            step_counter: VLA steps executed so far in this episode.
            total_steps: Maximum allowed steps for the task suite.

        Returns:
            AgentDecision with the agent's verdict.
        """
        self.agent_round += 1

        # Safety: if too many rounds, force continue
        if self.agent_round > self.max_agent_rounds:
            return AgentDecision(action="continue", reason="max agent rounds reached")

        # Build agent prompt
        prompt_text = self._build_agent_prompt(task_description, step_counter, total_steps)

        # Inject the agent prompt into the inputs
        # We replace the last user message text with the agent prompt
        agent_inputs = self._inject_agent_prompt(qwen_inputs, prompt_text)

        # Run LLM Head
        try:
            text = self.model.predict_text(agent_inputs, max_new_tokens=256)
        except Exception as e:
            logger.warning(f"LLM Head inference failed: {e}. Defaulting to continue.")
            return AgentDecision(action="continue", reason=f"llm inference error: {e}")

        logger.info(f"[Agent Round {self.agent_round}] LLM output: {text[:200]}...")

        # Parse decision
        return self._parse_decision(text)

    def record_obs(self, obs: Dict[str, Any]) -> None:
        """Record an observation for the agent's history buffer."""
        self.obs_history.append(obs)

    def record_action(self, action: np.ndarray) -> None:
        """Record an executed action for the agent's history buffer."""
        self.action_history.append(action)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_agent_prompt(
        self,
        task_description: str,
        step_counter: int,
        total_steps: int,
    ) -> str:
        """Build the agent monitoring prompt."""

        # Tool descriptions in natural language
        tool_descriptions = self._get_tool_descriptions()

        prompt = (
            f"You are a robot task supervisor monitoring a VLA (Vision-Language-Action) model.\n\n"
            f"Current task: \"{task_description}\"\n"
            f"Progress: {step_counter} / {total_steps} steps executed.\n\n"
            f"Your role:\n"
            f"1. Assess whether the VLA is making progress toward the task goal.\n"
            f"2. If progress is normal, output your observation and reasoning. Do NOT call any tool.\n"
            f"3. If the task is stuck, failing, or needs correction, call the appropriate tool.\n"
            f"4. If the task appears completed, call terminate with success=true.\n\n"
            f"{tool_descriptions}\n\n"
            f"Analyze the current observation and decide:"
        )
        return prompt

    @staticmethod
    def _get_tool_descriptions() -> str:
        """Format tool specs in Qwen function-calling style."""
        lines = ["Available tools:"]
        for tool in TOOL_REGISTRY.values():
            params_str = str(tool.parameters) if tool.parameters else "none"
            lines.append(f"- {tool.name}: {tool.description} (parameters: {params_str})")
        lines.append(
            "\nTo call a tool, output:\n"
            "<tool_call>\n"
            '{"name": "<tool_name>", "arguments": {...}}\n'
            "</tool_call>"
        )
        return "\n".join(lines)

    def _inject_agent_prompt(
        self,
        qwen_inputs: Dict[str, Any],
        prompt_text: str,
    ) -> Dict[str, Any]:
        """
        Inject the agent prompt into qwen_inputs by re-tokenizing.

        This builds a fresh input with the original images + the agent prompt.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        # Reconstruct messages with agent prompt
        # We take the images from the original observation and add the agent prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        # Try to include images if available from the original inputs
        # The images are already in pixel_values; we keep them as-is
        # and only update the text by re-tokenizing

        text = self.model.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Re-process with the same images
        image_inputs, video_inputs = process_vision_info(messages)
        new_inputs = self.model.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move to same device as original
        device = qwen_inputs["input_ids"].device if "input_ids" in qwen_inputs else "cuda"
        for k, v in new_inputs.items():
            if isinstance(v, torch.Tensor):
                new_inputs[k] = v.to(device)

        return new_inputs

    def _parse_decision(self, text: str) -> AgentDecision:
        """Parse LLM output into an AgentDecision."""
        if not ToolCallParser.has_tool_call(text):
            reasoning = ToolCallParser.extract_reasoning(text)
            return AgentDecision(
                action="continue",
                reason=reasoning or "no tool call — task appears to be progressing",
            )

        tool_calls = ToolCallParser.parse(text)
        if not tool_calls:
            return AgentDecision(action="continue", reason="failed to parse tool call")

        tc = tool_calls[0]
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        if name == "terminate":
            return AgentDecision(
                action="terminate",
                reason=args.get("reason", ""),
                success=args.get("success", False),
            )
        elif name == "call_vla":
            return AgentDecision(
                action="new_instruction",
                reason=args.get("reason", ""),
                instruction=args.get("instruction", ""),
                max_vla_steps=args.get("max_vla_steps", 50),
            )
        else:
            return AgentDecision(
                action="continue",
                reason=f"unknown tool: {name}",
            )
