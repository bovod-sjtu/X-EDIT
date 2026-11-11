"""Device and checkpoint preconditions for the isolated ClearVoice backend."""
from pathlib import Path
from typing import Any


def select_visible_cuda_device(cuda: Any) -> int | None:
    """Return a logical index within the service's visible CUDA device set."""
    return max(range(cuda.device_count()), key=lambda index: cuda.mem_get_info(index)[0], default=None)


def require_model_checkpoints(model: Any) -> None:
    """Reject ClearVoice's silent return after an unsuccessful model download."""
    for backend in model.models:
        root = Path(backend.args.checkpoint_dir)
        marker = root / "last_best_checkpoint"
        if not marker.is_file():
            raise RuntimeError(f"ClearVoice model checkpoint manifest missing: {marker}")
        names = [line.strip() for line in marker.read_text().splitlines() if line.strip()]
        if not names or any(not (root / name).is_file() or (root / name).stat().st_size == 0 for name in names):
            raise RuntimeError(f"ClearVoice model checkpoint files incomplete: {root}")
