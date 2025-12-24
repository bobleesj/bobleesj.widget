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

# Lazy imports for widgets with external dependencies (bobleesj.detector, etc.)
def __getattr__(name):
    if name == "Show4DSTEM":
        from bobleesj.widget.show4dstem import Show4DSTEM
        return Show4DSTEM
    if name == "Show4D":
        from bobleesj.widget.show4dstem import Show4DSTEM
        return Show4DSTEM
    if name == "Show5DSTEM":
        from bobleesj.widget.show5dstem import Show5DSTEM
        return Show5DSTEM
    if name == "Reconstruct":
        from bobleesj.widget.reconstruct import Reconstruct
        return Reconstruct
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Show2D", "Colormap", "Show3D", "Show4D", "Show4DSTEM", "Show5DSTEM", "Reconstruct"]
