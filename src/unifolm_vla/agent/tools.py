"""
Tool definitions and registry for the VLA Agent.

Tools are exposed to the LLM Head as callable functions. The model can invoke:
  - call_vla:   hand control back to the VLA Head with a (possibly updated) instruction
  - terminate:  end the current episode
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolSpec:
    """Specification of a tool that the LLM Head can call."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable] = None  # Set at registration time


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

TOOL_CALL_VLA = ToolSpec(
    name="call_vla",
    description=(
        "Call the VLA model to execute a sub-task with a new instruction. "
        "Use this ONLY when the current instruction cannot be completed or needs correction. "
        "Do NOT call this if the task is progressing normally."
    ),
    parameters={
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "New sub-task instruction in English, e.g. 'place the black bowl on the center of the plate'",
            },
            "reason": {
                "type": "string",
                "description": "Why a new instruction is needed (in Chinese or English)",
            },
            "max_vla_steps": {
                "type": "integer",
                "default": 50,
                "description": "Maximum number of VLA execution steps before the next agent check",
            },
        },
        "required": ["instruction", "reason"],
    },
)

TOOL_TERMINATE = ToolSpec(
    name="terminate",
    description=(
        "Terminate the current episode. Call this when the task is clearly completed "
        "or when the task has irrecoverably failed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "success": {
                "type": "boolean",
                "description": "True if the task succeeded, False if failed",
            },
            "reason": {
                "type": "string",
                "description": "Brief reason for termination",
            },
        },
        "required": ["success", "reason"],
    },
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# All available tools, keyed by name
TOOL_REGISTRY: Dict[str, ToolSpec] = {
    TOOL_CALL_VLA.name: TOOL_CALL_VLA,
    TOOL_TERMINATE.name: TOOL_TERMINATE,
}


def get_tool_descriptions() -> str:
    """
    Build a human-readable tool description string for the agent prompt.

    Uses the Qwen tool-calling convention with function definitions.
    """
    lines = ["You have access to the following tools:"]
    for tool in TOOL_REGISTRY.values():
        lines.append(f"\n## {tool.name}")
        lines.append(f"{tool.description}")
        if tool.parameters:
            lines.append(f"Parameters: {tool.parameters}")
    lines.append(
        "\nTo call a tool, output exactly:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {...}}\n'
        "</tool_call>"
    )
    lines.append(
        "If no tool call is needed, just output your observation and reasoning in plain text."
    )
    return "\n".join(lines)
