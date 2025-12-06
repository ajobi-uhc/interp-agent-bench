# Scoped Simple (RPC Pattern)

- Scoped sandbox for GPU isolation
- RPC communication between local agent and GPU
- Local vs remote library execution
- Model inference via exposed functions

## Running the Experiment

```bash
cd experiments/scoped-simple
python main.py
```

## Architecture

```
┌─────────────┐         RPC          ┌──────────────┐
│ Local Agent │ ◄──────────────────► │ GPU Sandbox  │
│             │                      │              │
│ helpers.py  │                      │ interface.py │
│ (local)     │                      │ model (GPU)  │
└─────────────┘                      └──────────────┘
```

## Code Structure

### `interface.py` (GPU)

```python
model = AutoModel.from_pretrained(model_path, device_map="auto")

@expose
def get_embedding(text: str) -> dict:
    ...
```

`@expose` decorator makes functions available via RPC.

### `helpers.py` (Local)

```python
def format_result(data: dict) -> str:
    ...
```
