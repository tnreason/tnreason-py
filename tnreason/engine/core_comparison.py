"""Core comparison utilities."""

import numpy as np


def _same_named_shape(core0, core1):
    if set(core0.colors) != set(core1.colors):
        return False
    return all(
        core0.shape[core0.colors.index(color)] == core1.shape[core1.colors.index(color)]
        for color in core0.colors
    )


def _values_in_color_order(core, target_colors):
    values = getattr(core, "values", None)
    if not isinstance(values, np.ndarray):
        return None
    if list(core.colors) == list(target_colors):
        return values
    permutation = [core.colors.index(color) for color in target_colors]
    return np.transpose(values, permutation)


def cores_equal(core0, core1):
    """Exact tensor equality, matching colors by name rather than by axis position."""
    if not _same_named_shape(core0, core1):
        return False

    values0 = _values_in_color_order(core0, core0.colors)
    values1 = _values_in_color_order(core1, core0.colors)
    if values0 is not None and values1 is not None:
        return bool(np.array_equal(values0, values1))

    for index in np.ndindex(*core0.shape):
        color_pos_dict = {color: index[i] for i, color in enumerate(core0.colors)}
        if core0[color_pos_dict] != core1[color_pos_dict]:
            return False
    return True


def cores_close(core0, core1, rtol=1e-9, atol=1e-9):
    """Approximate tensor equality for floating-point results of contractions."""
    if not _same_named_shape(core0, core1):
        return False

    values0 = _values_in_color_order(core0, core0.colors)
    values1 = _values_in_color_order(core1, core0.colors)
    if values0 is not None and values1 is not None:
        return bool(np.allclose(values0, values1, rtol=rtol, atol=atol))

    for index in np.ndindex(*core0.shape):
        color_pos_dict = {color: index[i] for i, color in enumerate(core0.colors)}
        if not np.isclose(core0[color_pos_dict], core1[color_pos_dict], rtol=rtol, atol=atol):
            return False
    return True
