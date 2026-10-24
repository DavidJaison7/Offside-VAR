import os

# Must be set before torch is imported: lets unsupported ops fall back to CPU on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def get_device():
    """Pick the fastest available device: Apple GPU (MPS) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"
