"""UnifoLM-VLA-ToolCall: Agent module with dual-head architecture.

Qwen2.5-VL backbone → two heads:
  VLA Head  (FlowmatchingActionHead) → 7D actions, high-frequency
  LLM Head  (lm_head → vocab)         → text/tool_calls, low-frequency
"""
