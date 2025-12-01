"""Mechanistic Interpretability Tools for Model Investigation."""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

# Module-level storage for loaded SAEs and steering vectors
_loaded_saes = {}
_steering_vectors = {}


def _get_model_and_tokenizer(model_name: Optional[str] = None):
    """Get model and tokenizer, defaulting to first available."""
    if model_name is None:
        model_name = list(models.keys())[0]

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")

    model = models[model_name]
    tokenizer = tokenizers[model_name]

    if model is None or tokenizer is None:
        raise ValueError(f"Model '{model_name}' failed to load")

    return model, tokenizer, model_name


def _get_layer_module(model, layer_idx: int, component: str = "residual"):
    """Get the module for a specific layer and component."""
    # Unwrap PEFT adapter if present (PeftModelForCausalLM -> Gemma2ForCausalLM)
    base_model = model.model if hasattr(model, 'model') and hasattr(model.model, 'model') else model

    # Access layers through .model.layers
    layers = base_model.model.layers
    layer = layers[layer_idx]

    # Map component names to actual modules
    component_map = {
        "residual": layer,
        "resid": layer,
        "resid_post": layer,
        "resid_pre": layer,
        "attn": layer.self_attn,
        "attention": layer.self_attn,
        "self_attn": layer.self_attn,
        "mlp": layer.mlp,
        "mlp_out": layer.mlp,
    }

    if component not in component_map:
        valid = list(component_map.keys())
        raise ValueError(f"Unknown component: {component}. Valid: {valid}")

    return component_map[component]


@expose
def get_model_info(model_name: Optional[str] = None) -> Dict:
    """Get model architecture info: layer count, hidden size, vocab size, etc."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)
    config = model.config

    # Detect if PEFT wrapped
    is_peft = hasattr(model, 'peft_config') or type(model).__name__.startswith('Peft')

    # Get base model for structure inspection
    base_model = model.model if is_peft else model

    info = {
        "device": str(model.device),
        "dtype": str(model.dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
    }

    # Add attention configuration
    if hasattr(config, 'num_attention_heads'):
        info["num_attention_heads"] = config.num_attention_heads
    if hasattr(config, 'num_key_value_heads'):
        info["num_key_value_heads"] = config.num_key_value_heads
        info["attention_type"] = "grouped_query" if config.num_key_value_heads < config.num_attention_heads else "multi_head"

    # Add MLP configuration
    if hasattr(config, 'intermediate_size'):
        info["intermediate_size"] = config.intermediate_size

    # Add normalization info
    if hasattr(config, 'rms_norm_eps'):
        info["norm_type"] = "rms_norm"
        info["norm_eps"] = config.rms_norm_eps

    return info


@expose
def construct_steering_vector(
    positive_prompts: List[str],
    negative_prompts: List[str],
    layer_idx: int,
    component: str = "residual",
    name: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict:
    """Construct steering vector from contrastive prompts using activation differences.

    The vector is automatically saved and can be referenced by name in apply_steering.
    """
    model, tokenizer, model_name_actual = _get_model_and_tokenizer(model_name)

    activations_pos = []
    activations_neg = []
    captured_acts = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        act = hidden_states.mean(dim=1)
        captured_acts.append(act.detach())

    target_module = _get_layer_module(model, layer_idx, component)
    handle = target_module.register_forward_hook(hook_fn)

    try:
        for prompt in positive_prompts:
            captured_acts = []
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                _ = model(**inputs)
            if captured_acts:
                activations_pos.append(captured_acts[0])

        for prompt in negative_prompts:
            captured_acts = []
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                _ = model(**inputs)
            if captured_acts:
                activations_neg.append(captured_acts[0])
    finally:
        handle.remove()

    pos_mean = torch.stack(activations_pos).mean(dim=0).squeeze()
    neg_mean = torch.stack(activations_neg).mean(dim=0).squeeze()
    steering_vec = pos_mean - neg_mean
    steering_vec = steering_vec / (steering_vec.norm() + 1e-8)

    # Auto-generate name if not provided
    if name is None:
        name = f"steering_L{layer_idx}_{len(_steering_vectors)}"

    # Save vector
    _steering_vectors[name] = {
        "vector": steering_vec,
        "layer_idx": layer_idx,
        "component": component,
        "norm": float(steering_vec.norm()),
    }

    return {
        "name": name,
        "layer_idx": layer_idx,
        "component": component,
        "vector_norm": float(steering_vec.norm()),
        "num_positive_examples": len(positive_prompts),
        "num_negative_examples": len(negative_prompts),
    }


@expose
def apply_steering(
    prompt: str,
    steering_vector_name: str,
    strength: float = 1.0,
    max_new_tokens: int = 100,
    model_name: Optional[str] = None
) -> Dict:
    """Generate with a saved steering vector applied at a specific layer."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)

    # Load steering vector by name
    if steering_vector_name not in _steering_vectors:
        return {
            "error": f"Steering vector '{steering_vector_name}' not found",
            "available": list(_steering_vectors.keys())
        }

    vec_data = _steering_vectors[steering_vector_name]
    steer_vec = vec_data["vector"].to(device=model.device, dtype=model.dtype) * strength
    layer_idx = vec_data["layer_idx"]
    component = vec_data["component"]

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + steer_vec.unsqueeze(0).unsqueeze(0)
            return (hidden_states,) + output[1:]
        else:
            return output + steer_vec.unsqueeze(0).unsqueeze(0)

    # Get base model for generation (unwrap PEFT if present)
    gen_model = model.model if hasattr(model, 'model') and hasattr(model.model, 'model') else model
    target_module = _get_layer_module(model, layer_idx, component)
    handle = target_module.register_forward_hook(steering_hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        handle.remove()

    return {
        "prompt": prompt,
        "generated": generated,
        "steering_vector_name": steering_vector_name,
        "steering_strength": strength,
        "layer_idx": layer_idx,
        "component": component,
    }


@expose
def list_steering_vectors() -> Dict:
    """List all saved steering vectors."""
    if not _steering_vectors:
        return {"steering_vectors": [], "count": 0}

    vectors_info = [
        {
            "name": name,
            "layer_idx": data["layer_idx"],
            "component": data["component"],
            "norm": data["norm"],
        }
        for name, data in _steering_vectors.items()
    ]

    return {
        "steering_vectors": vectors_info,
        "count": len(vectors_info),
    }


@expose
def prefill_attack(
    prompt: str,
    prefill: str,
    max_new_tokens: int = 100,
    model_name: Optional[str] = None
) -> Dict:
    """Test model with prefill attack (jailbreak technique)."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)

    full_prompt = f"{prompt}\n\n{prefill}"
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full_completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    continued_prefill = len(full_completion) > len(full_prompt) + 10

    return {
        "prompt": prompt,
        "prefill": prefill,
        "full_prompt": full_prompt,
        "completion": full_completion,
        "continued_generation": continued_prefill,
        "completion_length": len(full_completion),
    }


@expose
def logit_lens(
    prompt: str,
    layer_indices: Optional[Union[List[int], str]] = None,
    top_k: int = 10,
    model_name: Optional[str] = None
) -> Dict:
    """Apply logit lens to see predictions at intermediate layers."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)

    # Handle layer_indices - can be None, a list, or a comma-separated string
    if layer_indices is None:
        num_layers = model.config.num_hidden_layers
        layer_indices = list(range(0, num_layers, max(1, num_layers // 8)))
    elif isinstance(layer_indices, str):
        # Parse comma-separated string
        layer_indices = [int(x.strip()) for x in layer_indices.split(",")]

    unembed = model.lm_head

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    layer_predictions = {}

    for layer_idx in layer_indices:
        hidden_states_captured = []

        def make_hook(idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states_captured.append(output[0][:, -1, :])
                else:
                    hidden_states_captured.append(output[:, -1, :])
            return hook

        try:
            layer = _get_layer_module(model, layer_idx, "residual")
            handle = layer.register_forward_hook(make_hook(layer_idx))

            with torch.no_grad():
                _ = model(**inputs)

            handle.remove()

            if hidden_states_captured:
                hidden = hidden_states_captured[0]
                logits = unembed(hidden)
                probs = F.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], top_k)

                predictions = [
                    {
                        "token": tokenizer.decode([idx.item()]),
                        "token_id": idx.item(),
                        "probability": prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]

                layer_predictions[f"layer_{layer_idx}"] = predictions

        except Exception as e:
            layer_predictions[f"layer_{layer_idx}"] = {"error": str(e)}

    return {
        "prompt": prompt,
        "layer_predictions": layer_predictions,
        "layers_analyzed": layer_indices,
    }


@expose
def get_activations(
    prompt: str,
    layer_idx: int,
    component: str = "residual",
    position: Union[str, int] = "last",
    model_name: Optional[str] = None
) -> Dict:
    """Extract raw activations from a specific layer and component."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)

    captured_acts = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        captured_acts.append(hidden_states.detach())

    target_module = _get_layer_module(model, layer_idx, component)
    handle = target_module.register_forward_hook(hook_fn)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
    finally:
        handle.remove()

    if not captured_acts:
        return {"error": "No activations captured"}

    acts = captured_acts[0].squeeze(0)

    # Handle position parameter - can be string or integer
    if isinstance(position, int):
        acts = acts[position]
    elif position == "last":
        acts = acts[-1]
    elif position == "mean":
        acts = acts.mean(dim=0)
    else:
        raise ValueError(f"Invalid position: {position}. Use 'last', 'mean', or an integer index.")

    stats = {
        "mean": float(acts.mean()),
        "std": float(acts.std()),
        "max": float(acts.max()),
        "min": float(acts.min()),
    }

    return {
        "activations": acts.cpu().tolist(),
        "shape": list(acts.shape),
        "statistics": stats,
        "layer_idx": layer_idx,
        "component": component,
        "position": position,
        "prompt": prompt,
    }


@expose
def load_sae(
    layer_idx: int,
    width: str = "16k",
    model_name: Optional[str] = None
) -> Dict:
    """Load a Gemma Scope SAE for a specific layer."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)

    try:
        from sae_lens import SAE
    except ImportError:
        return {
            "error": "sae_lens not installed",
            "install_command": "pip install sae-lens",
            "status": "failed"
        }

    if "9b" in name.lower():
        release = "gemma-scope-9b-pt-res-canonical"
    elif "2b" in name.lower():
        release = "gemma-scope-2b-pt-res-canonical"
    else:
        return {"error": f"No Gemma Scope SAEs available for model: {name}"}

    sae_id = f"layer_{layer_idx}/width_{width}/canonical"
    sae_key = f"{layer_idx}_{width}"

    try:
        print(f"Loading SAE: {release} / {sae_id}")
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=str(model.device)
        )

        _loaded_saes[sae_key] = {
            "sae": sae,
            "config": cfg_dict,
            "sparsity": sparsity,
            "layer_idx": layer_idx,
            "width": width,
            "release": release
        }

        return {
            "status": "loaded",
            "layer_idx": layer_idx,
            "width": width,
            "sae_key": sae_key,
            "num_features": sae.cfg.d_sae if hasattr(sae, "cfg") else "unknown",
            "sparsity": float(sparsity) if sparsity is not None else None,
            "release": release,
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "layer_idx": layer_idx,
            "width": width
        }


@expose
def get_sae_features(
    prompt: str,
    layer_idx: int,
    width: str = "16k",
    top_k: int = 10,
    model_name: Optional[str] = None
) -> Dict:
    """Get top activated SAE features for a prompt."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)

    sae_key = f"{layer_idx}_{width}"
    if sae_key not in _loaded_saes:
        return {
            "error": f"SAE not loaded for layer {layer_idx}, width {width}",
            "action": f"Call load_sae(layer_idx={layer_idx}, width='{width}') first"
        }

    sae_data = _loaded_saes[sae_key]
    sae = sae_data["sae"]

    acts_result = get_activations(prompt, layer_idx, "residual", "last", model_name)
    if "error" in acts_result:
        return acts_result

    acts = torch.tensor(acts_result["activations"], device=model.device, dtype=model.dtype)

    with torch.no_grad():
        feature_acts = sae.encode(acts.unsqueeze(0)).squeeze(0)

    top_values, top_indices = torch.topk(feature_acts, top_k)

    features = [
        {
            "feature_id": int(idx),
            "activation": float(val),
        }
        for val, idx in zip(top_values, top_indices)
    ]

    return {
        "prompt": prompt,
        "layer_idx": layer_idx,
        "width": width,
        "top_features": features,
    }


@expose
def steer_with_sae_feature(
    prompt: str,
    layer_idx: int,
    feature_id: int,
    strength: float = 2.0,
    width: str = "16k",
    max_new_tokens: int = 100,
    model_name: Optional[str] = None
) -> Dict:
    """Generate text while amplifying or suppressing a specific SAE feature."""
    model, tokenizer, name = _get_model_and_tokenizer(model_name)

    sae_key = f"{layer_idx}_{width}"
    if sae_key not in _loaded_saes:
        return {
            "error": f"SAE not loaded for layer {layer_idx}, width {width}",
            "action": f"Call load_sae(layer_idx={layer_idx}, width='{width}') first"
        }

    sae_data = _loaded_saes[sae_key]
    sae = sae_data["sae"]

    decoder_weight = sae.W_dec[feature_id].to(dtype=model.dtype, device=model.device)
    steer_vec = decoder_weight * strength

    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + steer_vec.unsqueeze(0).unsqueeze(0)
            return (hidden_states,) + output[1:]
        else:
            return output + steer_vec.unsqueeze(0).unsqueeze(0)

    # Get base model for generation (unwrap PEFT if present)
    gen_model = model.model if hasattr(model, 'model') and hasattr(model.model, 'model') else model
    target_module = _get_layer_module(model, layer_idx, "residual")
    handle = target_module.register_forward_hook(steering_hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    finally:
        handle.remove()

    return {
        "prompt": prompt,
        "generated": generated,
        "layer_idx": layer_idx,
        "feature_id": feature_id,
        "strength": strength,
        "width": width,
    }


@expose
def list_loaded_saes() -> Dict:
    """List all currently loaded SAEs."""
    if not _loaded_saes:
        return {"loaded_saes": [], "count": 0}

    saes_info = []
    for key, data in _loaded_saes.items():
        saes_info.append({
            "sae_key": key,
            "layer_idx": data["layer_idx"],
            "width": data["width"],
            "release": data["release"],
            "sparsity": data["sparsity"],
        })

    return {
        "loaded_saes": saes_info,
        "count": len(saes_info),
    }
