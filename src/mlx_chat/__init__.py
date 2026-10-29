"""MLX Chat Terminal."""

__version__ = "0.2.0"
DEFAULT_MODEL = "mlx-community/Qwen3-4B-4bit"

MODEL_PRESETS = {
    "tiny": DEFAULT_MODEL,
    "small": "mlx-community/Qwen3-8B-4bit",
    "medium": "mlx-community/Qwen3-30B-A3B-4bit",
    "large": "mlx-community/Llama-3.3-70B-Instruct-4bit",
}


def resolve_model_preset(model: str) -> str:
    """Expand a friendly model-size preset without changing normal model paths."""
    return MODEL_PRESETS.get(model.strip().lower(), model)
