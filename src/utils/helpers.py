"""
helpers.py

Misc utilities that every module needs:
- Config loading from YAML
- Logging setup
- Metrics formatting
- Timer utility

Nothing fancy, just stuff I kept copy-pasting everywhere so I moved it here.
"""

import time
import functools
from pathlib import Path
from typing import Any, Dict

import yaml
from loguru import logger


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML config file. Returns flat-ish dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return config


def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log"):
    """Configure loguru — console + file output."""
    import sys
    Path("logs").mkdir(exist_ok=True)

    logger.remove()  # remove default handler
    logger.add(sys.stderr, level=log_level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}</cyan> - {message}")
    logger.add(log_file, level="DEBUG", rotation="10 MB", retention="7 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} - {message}")


def timer(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


def format_metrics(metrics: Dict[str, Any], title: str = "Results") -> str:
    """Pretty-print a metrics dict."""
    lines = [f"\n{'='*40}", f"  {title}", f"{'='*40}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<25}: {v:.4f}")
        elif isinstance(v, dict):
            lines.append(f"  {k}:")
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (int, float)):
                    lines.append(f"    {sub_k:<23}: {sub_v:.4f}")
        else:
            lines.append(f"  {k:<25}: {v}")
    lines.append("=" * 40)
    return "\n".join(lines)


def ensure_dirs(*paths: str):
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
