"""
Tool call parser: extracts <tool_call>...</tool_call> blocks from LLM text output.

Qwen2.5 系列模型使用 XML 风格的 tool call 格式:
    <tool_call>
    {"name": "call_vla", "arguments": {"instruction": "..."}}
    </tool_call>

本模块提供解析器从模型生成文本中提取这些块。
"""

import json
import re
from typing import Any, Dict, List, Optional


class ToolCallParser:
    """Parse <tool_call> JSON blocks from Qwen-style model output."""

    # Match <tool_call>...</tool_call>, supporting multi-line JSON bodies
    TOOL_CALL_PATTERN: re.Pattern = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
    )

    @staticmethod
    def parse(text: str) -> List[Dict[str, Any]]:
        """
        Extract all tool calls from generated text.

        Args:
            text: Raw model output (may contain text before/after tool_call blocks).

        Returns:
            List of parsed tool call dicts, each with {"name": ..., "arguments": {...}}.
            Returns empty list if no valid tool calls found.
        """
        matches = ToolCallParser.TOOL_CALL_PATTERN.findall(text)
        results = []
        for match in matches:
            try:
                parsed = json.loads(match.strip())
                if isinstance(parsed, dict) and "name" in parsed:
                    results.append(parsed)
            except json.JSONDecodeError:
                # Skip malformed tool calls gracefully
                continue
        return results

    @staticmethod
    def has_tool_call(text: str) -> bool:
        """Check if text contains at least one <tool_call> block."""
        return "<tool_call>" in text

    @staticmethod
    def extract_reasoning(text: str) -> str:
        """
        Extract the free-text reasoning portion before/around tool call blocks.

        Strips out <tool_call>...</tool_call> blocks and returns the remaining text.
        """
        cleaned = ToolCallParser.TOOL_CALL_PATTERN.sub("", text)
        return cleaned.strip()

    @staticmethod
    def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
        """Format a tool call into Qwen-style XML for prompt construction."""
        args_json = json.dumps(arguments, ensure_ascii=False)
        return f"<tool_call>\n{args_json}\n</tool_call>"

    @staticmethod
    def format_tool_response(name: str, result: Any) -> str:
        """Format a tool execution result for injection into conversation."""
        return f"<tool_response>\n{json.dumps({'name': name, 'result': result}, ensure_ascii=False)}\n</tool_response>"
