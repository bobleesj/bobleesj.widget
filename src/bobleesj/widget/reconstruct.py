"""
Reconstruct: Interactive phase reconstruction widget for 4D-STEM data.

Displays DPC (FFT-based) and iDPC (iterative) side by side.
Automatically determines optimal rotation angle.

Uses bobleesj.reconstruct for all computation - no duplication!
"""

import pathlib
import time

import anywidget
import cupy as cp
import traitlets

from bobleesj.reconstruct import (
    compute_center_of_mass,
    find_optimal_rotation,
    reconstruct_phase_from_gradient,
    iDPC as iDPCReconstructor,
)


class Reconstruct(anywidget.AnyWidget):
    """
    Interactive phase reconstruction widget.

    Shows DPC (FFT-based direct integration) and iDPC (iterative) 
    reconstructions side by side. Automatically determines optimal
    rotation angle by minimizing curl.

    Parameters
    ----------
    data : array_like
        4D-STEM data. Shape can be:
        - (N, det_x, det_y) - flattened scan, will infer square scan_shape
        - (scan_x, scan_y, det_x, det_y) - full 4D
    scan_shape : tuple[int, int], optional
        Scan dimensions (Rx, Ry). If not provided and data is 3D,
        assumes square scan (sqrt(N) x sqrt(N)).

    Examples
    --------
    >>> from bobleesj.widget import Reconstruct
    >>> from bobleesj.load4dstem import load
    >>> 
    >>> data = load('data.h5')  # Shape: (65536, 192, 192)
    >>> widget = Reconstruct(data)  # Infers 256x256 scan
    >>> widget
    """

    _esm = pathlib.Path(__file__).parent / "static" / "reconstruct.js"
    _css = pathlib.Path(__file__).parent / "static" / "reconstruct.css"

    # =========================================================================
    # Shape information
    # =========================================================================
    shape_x = traitlets.Int(1).tag(sync=True)  # Scan X
    shape_y = traitlets.Int(1).tag(sync=True)  # Scan Y
    det_x = traitlets.Int(1).tag(sync=True)    # Detector X
    det_y = traitlets.Int(1).tag(sync=True)    # Detector Y

    # =========================================================================
    # Sign toggle (only user control needed for phase)
    # =========================================================================
    phase_sign = traitlets.Int(1).tag(sync=True)  # +1 or -1

    # =========================================================================
    # Display options
    # =========================================================================
    colormap = traitlets.Unicode("inferno").tag(sync=True)
    percentile_low = traitlets.Float(1.0).tag(sync=True)
    percentile_high = traitlets.Float(99.0).tag(sync=True)

    # =========================================================================
    # Image bytes (sent to frontend)
    # =========================================================================
    dpc_bytes = traitlets.Bytes(b"").tag(sync=True)    # DPC (FFT-based)
    idpc_bytes = traitlets.Bytes(b"").tag(sync=True)   # iDPC (iterative)
    com_x_bytes = traitlets.Bytes(b"").tag(sync=True)  # CoM X (aligned)
    com_y_bytes = traitlets.Bytes(b"").tag(sync=True)  # CoM Y (aligned)

    # =========================================================================
    # Auto-determined rotation angle (display only, degrees)
    # =========================================================================
    rotation_angle_deg = traitlets.Float(0.0).tag(sync=True)

    # =========================================================================
    # Reconstruction stats
    # =========================================================================
    nmse = traitlets.Float(0.0).tag(sync=True)
    iterations_done = traitlets.Int(0).tag(sync=True)
    reconstruction_time_ms = traitlets.Float(0.0).tag(sync=True)
    dpc_time_ms = traitlets.Float(0.0).tag(sync=True)

    def __init__(
        self,
        data,
        scan_shape: tuple[int, int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
       

        # Convert to CuPy if needed
        if not hasattr(data, "__cuda_array_interface__"):
            data = cp.asarray(data)

        self._data = data

        # Infer shapes
        if data.ndim == 3:
            if scan_shape is not None:
                self._scan_shape = scan_shape
            else:
                n = data.shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(
                        f"Cannot infer square scan_shape from N={n}. "
                        f"Provide scan_shape explicitly."
                    )
                self._scan_shape = (side, side)
            self._det_shape = (data.shape[1], data.shape[2])
        elif data.ndim == 4:
            self._scan_shape = (data.shape[0], data.shape[1])
            self._det_shape = (data.shape[2], data.shape[3])
        else:
            raise ValueError(f"Expected 3D or 4D array, got {data.ndim}D")

        # Set shape traits
        self.shape_x = self._scan_shape[0]
        self.shape_y = self._scan_shape[1]
        self.det_x = self._det_shape[0]
        self.det_y = self._det_shape[1]

        # Compute everything
        self._run_reconstruction()

        # Observe changes (contrast handled in JS for instant response)
        self.observe(self._on_sign_change, names=["phase_sign"])

    def _run_reconstruction(self) -> None:
        """Run full reconstruction using bobleesj.reconstruct functions."""
        # Flatten if 4D
        if self._data.ndim == 4:
            Rx, Ry = self._scan_shape
            data_flat = self._data.reshape(Rx * Ry, *self._det_shape)
        else:
            data_flat = self._data

        # =====================================================================
        # Step 1: Compute CoM using bobleesj.reconstruct
        # =====================================================================
        com_qx, com_qy = compute_center_of_mass(data_flat, normalize_zero_mean=True)

        # =====================================================================
        # Step 2: Find optimal rotation using bobleesj.reconstruct
        # =====================================================================
        com_x_aligned, com_y_aligned, rotation_deg, _ = find_optimal_rotation(
            com_qx, com_qy, self._scan_shape, rotation_steps=180
        )
        self.rotation_angle_deg = rotation_deg
        self._com_x_aligned = com_x_aligned
        self._com_y_aligned = com_y_aligned

        # =====================================================================
        # Step 3: DPC (FFT-based) using bobleesj.reconstruct
        # =====================================================================
        t0 = time.perf_counter()
        dpc_phase = reconstruct_phase_from_gradient(com_x_aligned, com_y_aligned)
        dpc_phase = dpc_phase - dpc_phase.mean()
        self.dpc_time_ms = (time.perf_counter() - t0) * 1000
        self._dpc_phase = dpc_phase

        # =====================================================================
        # Step 4: iDPC (iterative) using bobleesj.reconstruct
        # =====================================================================
        t0 = time.perf_counter()
        
        # Create iDPC reconstructor and run (using raw CoM, not rotated)
        idpc = iDPCReconstructor(data_flat, self._scan_shape)
        idpc.preprocess(normalize_com=True, verbose=False)
        idpc.reconstruct(
            max_iter=30,
            step_size=0.5,
            show_results=False,
            verbose=False
        )
        
        self.reconstruction_time_ms = (time.perf_counter() - t0) * 1000
        self._idpc_phase = idpc.phase
        self.nmse = idpc.errors[-1] if idpc.errors else 0.0
        self.iterations_done = len(idpc.errors)
        self._errors = idpc.errors

        # Update display
        self._update_display()

    def _normalize_to_bytes(self, arr) -> bytes:
        """Normalize array to uint8 bytes using full range (contrast applied in JS)."""
        import cupy as cp

        arr = arr.astype(cp.float32)
        vmin = float(arr.min())
        vmax = float(arr.max())

        if vmax - vmin < 1e-10:
            normalized = cp.zeros_like(arr, dtype=cp.uint8)
        else:
            normalized = ((arr - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(cp.uint8)

        return bytes(cp.asnumpy(normalized))

    def _update_display(self) -> None:
        """Send raw images to frontend (filtering done client-side via WebGPU)."""
        # Apply sign only - filtering is done dynamically in the frontend
        self.dpc_bytes = self._normalize_to_bytes(self._dpc_phase * self.phase_sign)
        self.idpc_bytes = self._normalize_to_bytes(self._idpc_phase * self.phase_sign)
        self.com_x_bytes = self._normalize_to_bytes(self._com_x_aligned)
        self.com_y_bytes = self._normalize_to_bytes(self._com_y_aligned)

    def _on_sign_change(self, change) -> None:
        """Handle sign toggle."""
        self._update_display()

    def _on_display_change(self, change) -> None:
        """Handle display option changes."""
        self._update_display()

    @property
    def dpc_phase(self):
        """Get DPC phase as CuPy array."""
        return self._dpc_phase * self.phase_sign

    @property
    def idpc_phase(self):
        """Get iDPC phase as CuPy array."""
        return self._idpc_phase * self.phase_sign

    def __repr__(self) -> str:
        Rx, Ry = self._scan_shape
        lines = [f"Reconstruct(scan: {Rx}×{Ry}, det: {self.det_x}×{self.det_y})"]
        lines.append(f"  Rotation: {self.rotation_angle_deg:.1f}°")
        lines.append(f"  DPC: {self.dpc_time_ms:.1f}ms")
        lines.append(f"  iDPC: {self.iterations_done} iters, NMSE={self.nmse:.2e}, {self.reconstruction_time_ms:.1f}ms")
        return "\n".join(lines)
