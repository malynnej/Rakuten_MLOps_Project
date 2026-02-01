from pathlib import Path

import yaml


def get_project_root() -> Path:
    """Find project root (where config/ folder is)"""
    current = Path(__file__).resolve()

    # Look for config/ folder going up the tree
    for parent in [current] + list(current.parents):
        if (parent / "config").exists():
            return parent

    # Fallback to /app (Docker)
    return Path("/app")


PROJECT_ROOT = get_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(config_name: str) -> dict:
    """Load YAML config"""
    with open(CONFIG_DIR / f"{config_name}.yaml") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """
    Convert config path to absolute path
    Handles both /app/... (Docker) and relative (local)
    Returns str for URLs, Path for file paths
    """
    # If it's a URL, return as-is
    if relative_path.startswith(("http://", "https://", "s3://")):
        return relative_path

    path = Path(relative_path)

    # If already absolute, use as-is
    if path.is_absolute():
        return path

    # Otherwise, resolve relative to project root
    return PROJECT_ROOT / path


# Helper function (optional but useful)
def get_path(path_key: str):
    """
    Get path from dotted key notation
    Example: get_path("data.raw") -> Path("data/raw")
    """
    paths = load_config("paths")
    keys = path_key.split(".")

    value = paths
    for key in keys:
        value = value[key]

    return resolve_path(value)
