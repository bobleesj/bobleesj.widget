"""
bobleesj.widget: Interactive Jupyter widgets using anywidget + React.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("bobleesj-widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

# Core widgets - always available (no external dependencies)
from bobleesj.widget.show2d import Show2D, Colormap
from bobleesj.widget.show3d import Show3D

# Synthetic data generators for demos/testing
from bobleesj.widget import synthetic

# Lazy imports for widgets with external dependencies (bobleesj.detector, etc.)
def __getattr__(name):
    if name == "Show4DSTEM" or name == "Show4D":
        try:
            from bobleesj.widget.show4dstem import Show4DSTEM
            return Show4DSTEM
        except ImportError as e:
            raise ImportError(
                "Show4DSTEM requires 'bobleesj.detector'. "
                "Install it with: pip install bobleesj-detector"
            ) from e
    if name == "Show5DSTEM":
        try:
            from bobleesj.widget.show5dstem import Show5DSTEM
            return Show5DSTEM
        except ImportError as e:
            raise ImportError(
                "Show5DSTEM requires additional dependencies. "
                "See documentation for installation instructions."
            ) from e
    if name == "Reconstruct":
        try:
            from bobleesj.widget.reconstruct import Reconstruct
            return Reconstruct
        except ImportError as e:
            raise ImportError(
                "Reconstruct requires additional dependencies. "
                "See documentation for installation instructions."
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Show2D", "Colormap", "Show3D", "Show4D", "Show4DSTEM", "Show5DSTEM", "Reconstruct", "synthetic"]
