"""Basic RPC interface template for ScopedSandbox.

This template shows the minimal structure for an interface file.

Usage:
    scoped = ScopedSandbox(config)
    scoped.start()
    interface = scoped.serve("basic_interface.py", expose_as="library", name="tools")
"""

from transformers import AutoModel, AutoTokenizer
import torch

# get_model_path() is injected by the RPC server
model_path = get_model_path("google/gemma-2-9b")
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


@expose  # @expose decorator is injected by RPC server
def get_model_info() -> dict:
    """Get basic model information.

    Returns:
        dict: Model configuration details
    """
    config = model.config
    return {
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "device": str(model.device),
    }


@expose
def get_embedding(text: str) -> dict:
    """Get text embedding from model.

    Args:
        text: Input text to embed

    Returns:
        dict: Embedding info with full embedding and preview
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()

    return {
        "text": text,
        "embedding": embedding.tolist(),
        "preview": embedding[:10].tolist(),
        "shape": list(embedding.shape),
    }


@expose
def compare_embeddings(text1: str, text2: str) -> dict:
    """Compare embeddings of two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        dict: Similarity score and metadata
    """
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    e1 = torch.tensor(emb1["embedding"])
    e2 = torch.tensor(emb2["embedding"])
    similarity = float(torch.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)))

    return {
        "text1": text1,
        "text2": text2,
        "similarity": similarity,
    }
