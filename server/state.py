"""
Mutable server-state shared across modules.

These values start at their defaults but are updated during the FastAPI
lifespan once JumpNet has (optionally) announced a USB-drive storage root.
Storing them here breaks the circular import that would arise if routes
or the trainer tried to import from main.py directly.
"""
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent

# Absolute path to the models directory.  Updated by main.py lifespan.
models_dir: Path = _DEFAULT_ROOT / "models"
