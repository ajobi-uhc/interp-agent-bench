"""
Whitebox Target Agent - Local model with steering support.

Same as blackbox TargetAgent but:
  - Uses local HuggingFace model instead of API
  - Supports activation steering during generation
"""

import json
import re
import inspect
from typing import Optional
import torch
import uuid

# These are copied to /root/ by main.py
from extract_activations import extract_activation
from steering_hook import steering_hook


class WhiteboxTargetAgent:
    """Target model with steering support."""

    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        self.system_message: Optional[str] = None
        self.messages: list[dict] = []
        self.tools: list[dict] = []
        self.pending_tool_calls: list[dict] = []

        # Steering state: (layer_idx, vector, strength)
        self.steering = None

    def load_model(self):
        """Load model and tokenizer."""
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Try get_model_path (injected by sandbox RPC server for cached models)
        model_path = self.model_name
        try:
            model_path = get_model_path(self.model_name)
            print(f"Using cached model: {model_path}")
        except NameError:
            print(f"Loading from HuggingFace: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded on {self.model.device}")

    # ========================================================================
    # Tool Management (same as blackbox)
    # ========================================================================

    def add_tool(self, name: str, code: str, environment: str):
        """Register a synthetic tool."""
        local_scope = {}
        exec(code, {}, local_scope)

        func = next((v for v in local_scope.values() if callable(v) and not isinstance(v, type)), None)
        if func is None:
            raise ValueError(f"No function found in code for tool '{name}'")

        self.tools.append(self._build_tool_schema(func, name))

    def _build_tool_schema(self, func, name: str) -> dict:
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or f"Function {name}"

        parameters = {"type": "object", "properties": {}, "required": []}
        type_map = {int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}

        for pname, param in sig.parameters.items():
            ptype = type_map.get(param.annotation, "string") if param.annotation != inspect.Parameter.empty else "string"
            parameters["properties"][pname] = {"type": ptype, "description": f"Parameter {pname}"}
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(pname)

        return {"type": "function", "function": {"name": name, "description": docstring, "parameters": parameters}}

    # ========================================================================
    # Conversation
    # ========================================================================

    def initialize(self, system_message: str, tools: list[dict]):
        """Initialize with system prompt."""
        self.system_message = system_message
        self.messages = []
        self.tools = tools
        self.pending_tool_calls = []
        self.load_model()

    def send_message(self, message: str, prefill: str = "") -> dict:
        """Send user message and get response."""
        self.messages.append({"role": "user", "content": message})

        if prefill and prefill.strip():
            self.messages.append({"role": "assistant", "content": prefill.strip(), "prefill": True})

        return self._generate()

    def resume_with_tool_result(self, tool_call_id: str, result: str, status: str = "success", prefill: str = "") -> dict:
        """Resume after tool result."""
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})
        self.pending_tool_calls = [tc for tc in self.pending_tool_calls if tc["id"] != tool_call_id]

        if self.pending_tool_calls:
            return {"type": "waiting", "pending": len(self.pending_tool_calls)}

        if prefill and prefill.strip():
            self.messages.append({"role": "assistant", "content": prefill.strip(), "prefill": True})

        return self._generate()

    def _generate(self) -> dict:
        """Generate response with optional steering."""
        import sys

        try:
            messages = self._format_messages()

            # Build chat template args
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }

            # Add tools if available (Qwen3 supports Hermes-style tool calling)
            if self.tools:
                template_kwargs["tools"] = self.tools

            prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Log generation start (visible in RPC logs)
            steering_info = f"steering={self.steering is not None}" if self.steering else "no steering"
            print(f"[generate] Starting generation ({steering_info}, input_len={inputs.input_ids.shape[1]})", file=sys.stderr, flush=True)

            if self.steering:
                layer_idx, vector, strength = self.steering
                print(f"[generate] Applying steering: layer={layer_idx}, strength={strength}", flush=True)
                with steering_hook(self.model.model.layers, layer_idx, vector, strength):
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=512, do_sample=True,
                        temperature=0.7, pad_token_id=self.tokenizer.eos_token_id
                    )
            else:
                outputs = self.model.generate(
                    **inputs, max_new_tokens=512, do_sample=True,
                    temperature=0.7, pad_token_id=self.tokenizer.eos_token_id
                )

            print(f"[generate] Completed, output_len={outputs.shape[1]}", file=sys.stderr, flush=True)

            generated = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)

            # Check for Hermes-style tool calls
            tool_calls = self._parse_tool_calls(response)
            if tool_calls:
                self.messages.append({"role": "assistant", "content": response, "tool_calls": tool_calls})
                self.pending_tool_calls = tool_calls
                return {"type": "tool_calls", "tool_calls": tool_calls}

            self.messages.append({"role": "assistant", "content": response})
            return {"type": "text", "content": response}

        except Exception as e:
            import traceback
            return {"type": "text", "content": f"[Error: {e}]\n{traceback.format_exc()}"}

    def _format_messages(self) -> list[dict]:
        """Format messages for chat template.

        Qwen3 supports system role. Tool messages are converted to user
        messages for compatibility. Consecutive same-role messages are merged.
        """
        msgs = []

        if self.system_message:
            msgs.append({"role": "system", "content": self.system_message})

        for m in self.messages:
            if m.get("prefill"):
                continue

            role = m["role"]
            content = m["content"]

            # Convert tool messages to user (HF templates don't support tool role)
            if role == "tool":
                role = "user"
                content = f"[Tool Result]\n{content}"
            elif role not in ["user", "assistant"]:
                continue

            # Merge consecutive same-role messages to ensure alternation
            if msgs and msgs[-1]["role"] == role:
                msgs[-1]["content"] += f"\n\n{content}"
            else:
                msgs.append({"role": role, "content": content})

        return msgs

    def _parse_tool_calls(self, response: str) -> list[dict]:
        """Parse Hermes-style tool calls from response.

        Hermes format:
            <tool_call>
            {"name": "func_name", "arguments": {"arg": "value"}}
            </tool_call>
        """
        tool_calls = []

        # Find all <tool_call>...</tool_call> blocks
        pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match.strip())
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": data.get("name", ""),
                    "arguments": data.get("arguments", {}),
                })
            except json.JSONDecodeError:
                continue

        return tool_calls

    # ========================================================================
    # Steering
    # ========================================================================

    def get_contrastive_vector(self, prompt_a: str, prompt_b: str, layer_idx: int) -> torch.Tensor:
        """
        Get steering vector from contrastive prompts.
        Returns: activation(prompt_b) - activation(prompt_a)
        """
        print(f"[steering] Creating contrastive vector at layer {layer_idx}", flush=True)
        print(f"[steering]   prompt_a: {prompt_a[:50]}...", flush=True)
        print(f"[steering]   prompt_b: {prompt_b[:50]}...", flush=True)

        act_a = extract_activation(self.model, self.tokenizer, prompt_a, layer_idx, position=-1)
        print(f"[steering]   act_a shape: {act_a.shape}, norm: {act_a.norm():.2f}", flush=True)

        act_b = extract_activation(self.model, self.tokenizer, prompt_b, layer_idx, position=-1)
        print(f"[steering]   act_b shape: {act_b.shape}, norm: {act_b.norm():.2f}", flush=True)

        vector = act_b - act_a
        print(f"[steering]   vector norm: {vector.norm():.2f}", flush=True)
        return vector

    def set_steering(self, vector: torch.Tensor, layer_idx: int, strength: float = 1.0):
        """Set steering vector for future generations."""
        print(f"[steering] Setting steering: layer={layer_idx}, strength={strength}, vector_norm={vector.norm():.2f}", flush=True)
        self.steering = (layer_idx, vector, strength)

    def clear_steering(self):
        """Remove steering."""
        print("[steering] Clearing steering", flush=True)
        self.steering = None

    # ========================================================================
    # State
    # ========================================================================

    def get_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "num_messages": len(self.messages),
            "num_tools": len(self.tools),
            "pending_tool_calls": len(self.pending_tool_calls),
            "steering_active": self.steering is not None,
        }
