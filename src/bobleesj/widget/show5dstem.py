"""
show5dstem: Interactive 5D-STEM viewer for tilt/time series data.

Features:
- Tilt/time series navigation (dimension 0)
- Scan position navigation (dimensions 1, 2)
- Single DP transfer per change (memory efficient)
- Zoom/pan on diffraction pattern
- Colormap selection
"""

import pathlib

import anywidget
import numpy as np
import traitlets

from bobleesj.detector import (
    normalize_to_uint8,
    circular_mask,
    annular_mask,
    square_mask,
    get_probe_size,
)


class Show5DSTEM(anywidget.AnyWidget):
    """
    Interactive 5D-STEM viewer for tilt/time series.

    Designed for joint-ptycho tomography and in-situ time-series 4D-STEM.
    Only transfers one diffraction pattern at a time for memory efficiency.

    Parameters
    ----------
    data : array_like
        5D CuPy/NumPy array of shape (N, Sx, Sy, Kx, Ky) where:
        - N = number of tilts or time points
        - Sx, Sy = scan dimensions
        - Kx, Ky = detector dimensions
    title : str, optional
        Title to display above the widget.
    log_scale : bool, default True
        Use log scale for better dynamic range visualization.co

    Examples
    --------
    >>> import cupy as cp
    >>> from bobleesj.widget import Show5DSTEM
    >>> data_5d = cp.random.rand(71, 64, 64, 96, 96)
    >>> Show5DSTEM(data_5d, title="Tilt Series")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show5dstem.js"

    # Title
    title = traitlets.Unicode("").tag(sync=True)

    # 5D shape info
    n_tilts = traitlets.Int(1).tag(sync=True)
    shape_x = traitlets.Int(1).tag(sync=True)
    shape_y = traitlets.Int(1).tag(sync=True)
    det_x = traitlets.Int(1).tag(sync=True)
    det_y = traitlets.Int(1).tag(sync=True)

    # Current position
    tilt_idx = traitlets.Int(0).tag(sync=True)
    scan_x = traitlets.Int(0).tag(sync=True)
    scan_y = traitlets.Int(0).tag(sync=True)

    # Display options
    log_scale = traitlets.Bool(True).tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    
    # =========================================================================
    # Detector Integration (BF/ADF overlays)
    # =========================================================================
    has_detector = traitlets.Bool(False).tag(sync=True)
    center_x = traitlets.Float(0.0).tag(sync=True)
    center_y = traitlets.Float(0.0).tag(sync=True)
    bf_radius = traitlets.Float(0.0).tag(sync=True)
    show_bf_overlay = traitlets.Bool(True).tag(sync=True)
    show_adf_overlay = traitlets.Bool(False).tag(sync=True)
    adf_inner_radius = traitlets.Float(0.0).tag(sync=True)
    adf_outer_radius = traitlets.Float(0.0).tag(sync=True)
    
    # =========================================================================
    # Virtual Image (ROI-based)
    # =========================================================================
    roi_active = traitlets.Bool(False).tag(sync=True)
    roi_mode = traitlets.Unicode("point").tag(sync=True)  # 'point', 'circle', 'square', 'rect', 'annular'
    roi_center_x = traitlets.Float(0.0).tag(sync=True)
    roi_center_y = traitlets.Float(0.0).tag(sync=True)
    roi_radius = traitlets.Float(10.0).tag(sync=True)
    roi_radius_inner = traitlets.Float(5.0).tag(sync=True)
    roi_width = traitlets.Float(20.0).tag(sync=True)
    roi_height = traitlets.Float(10.0).tag(sync=True)
    
    # Scale Bar
    pixel_size = traitlets.Float(1.0).tag(sync=True)
    det_pixel_size = traitlets.Float(1.0).tag(sync=True)

    # Current DP bytes (single frame, not all 5D!)
    dp_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Scan image bytes (mean DP intensity at each scan position for current tilt)
    scan_image_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Stats
    stats_mean = traitlets.Float(0.0).tag(sync=True)
    stats_max = traitlets.Float(0.0).tag(sync=True)
    stats_min = traitlets.Float(0.0).tag(sync=True)

    def __init__(
        self,
        data,
        title: str = "",
        log_scale: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # CuPy required
        import cupy as cp
        self._xp = cp

        # Validate 5D shape
        if data.ndim != 5:
            raise ValueError(
                f"Expected 5D array (N, Sx, Sy, Kx, Ky), got {data.ndim}D. "
                f"Shape: {data.shape}"
            )

        self._data = data
        self.title = title
        self.log_scale = log_scale

        # Extract shape info
        self.n_tilts = data.shape[0]
        self.shape_x = data.shape[1]
        self.shape_y = data.shape[2]
        self.det_x = data.shape[3]
        self.det_y = data.shape[4]

        # Initial position at center
        self.tilt_idx = 0
        self.scan_x = self.shape_x // 2
        self.scan_y = self.shape_y // 2

        # Compute global range for consistent scaling
        self._compute_global_range()

        # Compute initial scan image for tilt 0
        if "detector" in kwargs:
             self._setup_detector(kwargs["detector"])
        else:
             self._auto_detect_probe()

        self._compute_virtual_image()

        # Send initial DP
        self._update_dp()

        # Observe position changes
        self.observe(self._update_dp, names=["tilt_idx", "scan_x", "scan_y", "log_scale"])
        self.observe(self._on_tilt_change, names=["tilt_idx"])
        self.observe(self._on_roi_change, names=[
            "roi_center_x", "roi_center_y", "roi_radius", "roi_radius_inner", "roi_width", "roi_height", "roi_mode"
        ])
    
    def _setup_detector(self, detector):
        """Configure detector overlays."""
        self.has_detector = True
        center = detector.center
        self.center_x = float(center[0])
        self.center_y = float(center[1])
        self.bf_radius = float(detector.bf_radius)
        self.adf_inner_radius = self.bf_radius * 1.5
        self.adf_outer_radius = self.bf_radius * 3.0
        
    def _auto_detect_probe(self):
        """Auto-detect probe from mean DP (collapsed over 5D)."""
        # Take mean over first tilt, center scan position
        mean_dp = self._data[0].mean(axis=(0, 1))
        radius, row, col = get_probe_size(mean_dp)
        self.center_x = float(col)
        self.center_y = float(row)
        self.bf_radius = float(radius)
        self.adf_inner_radius = self.bf_radius * 1.5
        self.adf_outer_radius = self.bf_radius * 3.0
        
        # Init ROI at BF
        self.roi_center_x = self.center_x
        self.roi_center_y = self.center_y
        self.roi_radius = self.bf_radius
        self.roi_mode = "circle"

    def _on_roi_change(self, change=None):
        """Recompute virtual image when ROI changes."""
        self._compute_virtual_image()

    def _compute_global_range(self):
        """Compute global min/max from sampled frames."""
        xp = self._xp

        # Sample a few frames for range estimation
        samples = [
            (0, 0, 0),
            (0, self.shape_x // 2, self.shape_y // 2),
            (self.n_tilts // 2, self.shape_x // 2, self.shape_y // 2),
            (self.n_tilts - 1, self.shape_x // 2, self.shape_y // 2),
        ]

        all_min, all_max = float("inf"), float("-inf")
        for t, x, y in samples:
            frame = self._data[t, x, y]
            fmin = float(frame.min())
            fmax = float(frame.max())
            all_min = min(all_min, fmin)
            all_max = max(all_max, fmax)

        self._global_min = max(all_min, 1e-10)
        self._global_max = all_max

        # Log range
        self._log_min = np.log1p(self._global_min)
        self._log_max = np.log1p(self._global_max)

    def _update_dp(self, change=None):
        """Send current DP to frontend."""
        xp = self._xp

        # Get current DP
        frame = self._data[self.tilt_idx, self.scan_x, self.scan_y]

        # Update stats
        self.stats_mean = float(frame.mean().get())
        self.stats_max = float(frame.max().get())
        self.stats_min = float(frame.min().get())

        # Apply log scale if enabled
        if self.log_scale:
            frame = xp.log1p(frame.astype(xp.float32))
            vmin, vmax = self._log_min, self._log_max
        else:
            frame = frame.astype(xp.float32)
            vmin, vmax = self._global_min, self._global_max

        # Normalize to uint8
        if vmax > vmin:
            normalized = xp.clip((frame - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.astype(xp.uint8)
        else:
            normalized = xp.zeros(frame.shape, dtype=xp.uint8)

        # Transfer to CPU and send
        self.dp_bytes = normalized.get().tobytes()

    def _on_tilt_change(self, change=None):
        """Recompute scan image when tilt changes."""
        self._compute_virtual_image()

    def _compute_scan_image(self):
        """Compute mean DP intensity at each scan position for current tilt."""
        xp = self._xp

        # Get current tilt's scan slab: (Sx, Sy, Kx, Ky)
        tilt_data = self._data[self.tilt_idx]

        # Sum over detector to get (Sx, Sy) intensity
        scan_image = tilt_data.sum(axis=(2, 3))

        # Normalize to uint8
        scan_image = scan_image.astype(xp.float32)
        vmin, vmax = float(scan_image.min()), float(scan_image.max())
        if vmax > vmin:
            normalized = xp.clip((scan_image - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.astype(xp.uint8)
        else:
            normalized = xp.zeros(scan_image.shape, dtype=xp.uint8)

        self.scan_image_bytes = normalized.get().tobytes()

    def _compute_virtual_image(self):
        """Compute virtual image for current tilt using ROI."""
        xp = self._xp
        tilt_data = self._data[self.tilt_idx] # (Sx, Sy, Kx, Ky)
        
        if self.roi_mode == "point":
             # Point detector: slice at (ky, kx)
             kx = int(max(0, min(self.det_x - 1, round(self.roi_center_y))))
             ky = int(max(0, min(self.det_y - 1, round(self.roi_center_x))))
             scan_image = tilt_data[:, :, kx, ky]
        else:
             # Masked sum
             if self.roi_mode == "circle":
                 mask = circular_mask((self.det_x, self.det_y), (self.roi_center_y, self.roi_center_x), self.roi_radius)
             elif self.roi_mode == "square":
                 mask = square_mask((self.det_x, self.det_y), (self.roi_center_y, self.roi_center_x), self.roi_radius)
             elif self.roi_mode == "rect":
                 # Custom rect mask logic (simplified as square for now or import rect)
                 # Re-using square mask logic for now but ideally should be rect
                 mask = square_mask((self.det_x, self.det_y), (self.roi_center_y, self.roi_center_x), max(self.roi_width, self.roi_height)/2)
             elif self.roi_mode == "annular":
                 mask = annular_mask((self.det_x, self.det_y), (self.roi_center_y, self.roi_center_x), self.roi_radius_inner, self.roi_radius)
             else:
                 return

             # (Sx, Sy, Kx, Ky) * (Kx, Ky) -> sum over last 2
             scan_image = (tilt_data * mask).sum(axis=(2, 3))

        # Normalize
        scan_image = scan_image.astype(xp.float32)
        vmin, vmax = float(scan_image.min()), float(scan_image.max())
        if vmax > vmin:
            normalized = xp.clip((scan_image - vmin) / (vmax - vmin) * 255, 0, 255)
            normalized = normalized.astype(xp.uint8)
        else:
            normalized = xp.zeros(scan_image.shape, dtype=xp.uint8)

        self.scan_image_bytes = normalized.get().tobytes()

    def goto(self, tilt: int | None = None, x: int | None = None, y: int | None = None):
        """
        Jump to a specific position.

        Parameters
        ----------
        tilt : int, optional
            Tilt/time index.
        x : int, optional
            Scan X position.
        y : int, optional
            Scan Y position.
        """
        if tilt is not None:
            self.tilt_idx = max(0, min(self.n_tilts - 1, tilt))
        if x is not None:
            self.scan_x = max(0, min(self.shape_x - 1, x))
        if y is not None:
            self.scan_y = max(0, min(self.shape_y - 1, y))
        return self
