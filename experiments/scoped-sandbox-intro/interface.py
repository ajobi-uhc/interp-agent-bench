"""Model analysis tools - runs in GPU sandbox via RPC."""

from transformers import AutoModel, AutoTokenizer
import torch

# get_model_path and @expose are injected by RPC server
model_path = get_model_path("google/gemma-2-9b")
model = AutoModel.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)


@expose
def get_model_info() -> dict:
    """Get basic model information."""
    config = model.config
    return {
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "device": str(model.device),
    }


@expose
def get_embedding(text: str) -> dict:
    """Get text embedding from model."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    embedding = outputs.hidden_states[-1].mean(dim=1).squeeze()

    return {
        "text": text,
        "embedding": embedding.tolist(),
        "preview": embedding[:10].tolist(),
    }


@expose
def compare_embeddings(text1: str, text2: str) -> dict:
    """Compare embeddings of two texts."""
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
