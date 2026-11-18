"""Minimum-image convention for a periodic parallelepiped.

Exports ``Cell``, ``reduce_image_pairs``, and ``__version__``.
Coordinates, lattice vectors, origins, and displacements share one
caller-chosen Cartesian length unit; squared distances use that unit
squared. Array arguments accept any CPU DLPack producer (such as NumPy,
PyTorch, JAX, or host CuPy) and return NumPy arrays when NumPy is
importable, or Python lists otherwise. Invalid boxes, shapes, or
dtypes raise ``ValueError``, ``TypeError``, or ``BufferError``.
"""

from minimage._lib import Cell, __version__, reduce_image_pairs

__all__ = ["Cell", "__version__", "reduce_image_pairs"]
