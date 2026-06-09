"""
General utilities for GPPyramid Transformer.
"""

from pathlib import Path


def compute_scale_lengths(target_len=365, n_scales=4):
    """
    Compute temporal lengths of the pyramidal sequence.

    The default configuration gives 365, 183, 92, and 46 steps.
    """
    lengths = [target_len]
    current = target_len
    for _ in range(n_scales - 1):
        current = (current + 2 * 1 - 3) // 2 + 1
        lengths.append(current)
    return tuple(lengths)


def ensure_dir(path):
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(path):
    """Load a YAML configuration file."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
