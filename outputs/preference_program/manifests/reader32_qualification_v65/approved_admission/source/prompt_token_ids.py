"""Dependency-free normalization of one tokenized prompt; never iterate mapping keys."""
from collections.abc import Mapping

def normalize_prompt_ids(value, max_tokens=512):
    if isinstance(value, Mapping):
        if 'input_ids' not in value:
            raise ValueError('mapping lacks input_ids')
        value = value['input_ids']
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError('expected list or tensor input_ids')
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError('expected exactly one prompt, not a batch')
        value = value[0]
    if not value or len(value) > max_tokens:
        raise ValueError('empty or oversized prompt')
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError('token IDs must be nonnegative Python integers')
    return list(value)
