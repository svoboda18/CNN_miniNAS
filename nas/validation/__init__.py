from .base import ValidationStrategy
from .full_training import FullTraining

__all__ = ['ValidationStrategy', 'FullTraining', 'get_validator']


def get_validator(config: dict) -> ValidationStrategy:
    """
    Factory. Returns a ValidationStrategy instance based on config type.

    Example:
        validator = get_validator(config)
        result    = validator.validate(architecture)
    """
    vs_type = config.get('ValidationStrategy', {}).get('type', 'FullTraining')
    registry = {
        'FullTraining': FullTraining,
    }
    if vs_type not in registry:
        raise ValueError(f"[Validation] Unknown type: '{vs_type}'. "
                         f"Available: {list(registry.keys())}")
    return registry[vs_type](config)
