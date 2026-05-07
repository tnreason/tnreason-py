"""Core comparison utilities."""
import numpy as np


def cores_equal(core1, core2) -> bool:
    """Exact equality of two cores (value and color set)."""
    if set(core1.colors) != set(core2.colors):
        return False
    # Reorder core2 to match core1's color order
    perm = [core2.colors.index(c) for c in core1.colors]
    v2 = np.transpose(core2.values, perm)
    return np.array_equal(core1.values, v2)


def cores_close(core1, core2, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Approximate equality of two cores up to color reordering."""
    if set(core1.colors) != set(core2.colors):
        return False
    perm = [core2.colors.index(c) for c in core1.colors]
    v2 = np.transpose(core2.values, perm)
    return bool(np.allclose(core1.values, v2, rtol=rtol, atol=atol))
