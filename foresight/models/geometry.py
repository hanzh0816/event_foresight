"""Spatial dimensions for RGB preprocessing and patch feature grids.

An integer denotes a square; pairs always use (height, width) order.
"""

from numbers import Integral


def normalize_hw(value, name="size"):
    """Return positive integer (height, width) from an integer or a pair."""
    if isinstance(value, Integral) and not isinstance(value, bool):
        dimensions = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        dimensions = tuple(value)
    else:
        raise ValueError(f"{name} must be an integer or [height, width], got {value!r}")
    if any(not isinstance(item, Integral) or isinstance(item, bool) or item <= 0
           for item in dimensions):
        raise ValueError(f"{name} must contain positive integers, got {value!r}")
    return tuple(int(item) for item in dimensions)


def patch_grid(image_size, patch_size=14):
    """Return the patch grid, rejecting dimensions that would drop border pixels."""
    height, width = normalize_hw(image_size, "image_size")
    patch_height, patch_width = normalize_hw(patch_size, "patch_size")
    if height % patch_height or width % patch_width:
        raise ValueError(f"image_size {(height, width)} must be divisible by patch_size "
                         f"{(patch_height, patch_width)}")
    return height // patch_height, width // patch_width
