"""Auto-generated RPC client for model_tools

Server: https://ta-01kbmbxthyzcqppnyprs17g7jg-8080.wo-yoqdxxo82lg9nx4bidnhwtu0d.w.modal.host
"""

import requests
from typing import Any, Optional


RPC_URL = "https://ta-01kbmbxthyzcqppnyprs17g7jg-8080.wo-yoqdxxo82lg9nx4bidnhwtu0d.w.modal.host"
RPC_TIMEOUT = 600


def _call_rpc(fn_name: str, **kwargs) -> Any:
    """Internal RPC caller."""
    resp = requests.post(
        RPC_URL,
        json={"fn": fn_name, "kwargs": kwargs},
        timeout=RPC_TIMEOUT,
    )
    resp.raise_for_status()

    data = resp.json()
    if not data.get("ok"):
        error_msg = data.get("error", "Unknown error")
        traceback = data.get("traceback", "")
        if traceback:
            error_msg += f"\n\nRemote traceback:\n{traceback}"
        raise RuntimeError(error_msg)

    return data["result"]



def get_model_info():
    """Get basic model information."""
    return _call_rpc("get_model_info", )


def get_embedding(text: str):
    """Get text embedding from model."""
    return _call_rpc("get_embedding", text=text)


def compare_embeddings(text1: str, text2: str):
    """Compare embeddings of two texts."""
    return _call_rpc("compare_embeddings", text1=text1, text2=text2)

