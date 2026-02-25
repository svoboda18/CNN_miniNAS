from .base import SearchSpace
from .layers_based import LayersBased

__all__ = ['SearchSpace', 'LayersBased', 'get_search_space']


def get_search_space(config: dict) -> SearchSpace:
    """
    Factory. Returns a SearchSpace instance based on config type.
    Call .define_space(config) on the result to get list[dict].

    Example:
        space_obj     = get_search_space(config)        # LayersBased instance
        architectures = space_obj.define_space(config)  # list[dict]
    """
    ss_type = config.get('SearchSpace', {}).get('type', 'LayersBased')
    registry = {
        'LayersBased': LayersBased,
    }
    if ss_type not in registry:
        raise ValueError(f"[SearchSpace] Unknown type: '{ss_type}'. "
                         f"Available: {list(registry.keys())}")
    return registry[ss_type]()