"""
Synthetic data generators for testing and demonstrations.

Provides realistic-looking scientific images for notebooks and testing.
"""

import numpy as np


def diffraction_pattern(
    size: int = 256,
    center_intensity: float = 100.0,
    n_spots: int = 6,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a realistic-looking electron diffraction pattern.

    Parameters
    ----------
    size : int
        Image size (square). Default 256.
    center_intensity : float
        Intensity of the central beam. Default 100.0.
    n_spots : int
        Number of spots in the first diffraction ring. Default 6.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D float32 array of shape (size, size).

    Examples
    --------
    >>> from bobleesj.widget.synthetic import diffraction_pattern
    >>> dp = diffraction_pattern(256)
    >>> dp.shape
    (256, 256)
    """
    if seed is not None:
        np.random.seed(seed)

    dp = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Central beam (Gaussian)
    sigma_center = size * 0.05
    dp += center_intensity * np.exp(-(r**2) / (2 * sigma_center**2))

    # First ring of diffraction spots
    ring_radius = size * 0.25
    spot_intensity = center_intensity * 0.3
    sigma_spot = size * 0.03

    for i in range(n_spots):
        angle = 2 * np.pi * i / n_spots
        sx = cx + ring_radius * np.cos(angle)
        sy = cy + ring_radius * np.sin(angle)
        r_spot = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        dp += spot_intensity * np.exp(-(r_spot**2) / (2 * sigma_spot**2))

    # Second ring (weaker, more spots)
    ring_radius2 = size * 0.4
    spot_intensity2 = center_intensity * 0.1
    for i in range(n_spots * 2):
        angle = 2 * np.pi * i / (n_spots * 2) + np.pi / n_spots
        sx = cx + ring_radius2 * np.cos(angle)
        sy = cy + ring_radius2 * np.sin(angle)
        r_spot = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
        dp += spot_intensity2 * np.exp(-(r_spot**2) / (2 * sigma_spot**2))

    # Add Poisson noise
    dp += np.random.poisson(1, (size, size)).astype(np.float32)
    return dp


def lattice_image(
    size: int = 256,
    spacing: float = 0.5,
    noise: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic HRTEM-like lattice image.

    Parameters
    ----------
    size : int
        Image size (square). Default 256.
    spacing : float
        Lattice spacing parameter. Default 0.5.
    noise : float
        Noise level (std of Gaussian noise). Default 0.1.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        2D float32 array of shape (size, size).

    Examples
    --------
    >>> from bobleesj.widget.synthetic import lattice_image
    >>> img = lattice_image(256, spacing=0.5)
    >>> img.shape
    (256, 256)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)

    # Gaussian envelope
    envelope = np.exp(-(X**2 + Y**2) / 8)

    # Lattice pattern
    lattice = np.cos(2 * np.pi * X / spacing) * np.cos(2 * np.pi * Y / spacing)

    img = envelope * lattice + noise * np.random.randn(size, size)
    return img.astype(np.float32)


def defocus_series(
    size: int = 256,
    n_frames: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic defocus series (3D stack).

    Parameters
    ----------
    size : int
        Image size (square). Default 256.
    n_frames : int
        Number of frames in the series. Default 10.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        3D float32 array of shape (n_frames, size, size).

    Examples
    --------
    >>> from bobleesj.widget.synthetic import defocus_series
    >>> stack = defocus_series(128, n_frames=15)
    >>> stack.shape
    (15, 128, 128)
    """
    if seed is not None:
        np.random.seed(seed)

    stack = np.zeros((n_frames, size, size), dtype=np.float32)

    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)

    for i in range(n_frames):
        freq = 1 + i * 0.3
        envelope = np.exp(-((X**2 + Y**2) / (3 + i * 0.3)))
        pattern = np.sin(freq * X) * np.cos(freq * Y)
        stack[i] = (pattern * envelope + 1) * 0.5

    return stack


def stem_4d(
    scan_size: int = 64,
    det_size: int = 128,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate synthetic 4D-STEM data with position-dependent beam shift.

    Parameters
    ----------
    scan_size : int
        Scan dimensions (square). Default 64.
    det_size : int
        Detector dimensions (square). Default 128.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        4D float32 array of shape (scan_size, scan_size, det_size, det_size).

    Examples
    --------
    >>> from bobleesj.widget.synthetic import stem_4d
    >>> data = stem_4d(32, 64)
    >>> data.shape
    (32, 32, 64, 64)
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.zeros((scan_size, scan_size, det_size, det_size), dtype=np.float32)

    # Pre-generate base diffraction pattern
    base_dp = diffraction_pattern(det_size, center_intensity=100, seed=seed)

    for sx in range(scan_size):
        for sy in range(scan_size):
            # Position-dependent beam shift
            shift_x = int(3 * np.sin(2 * np.pi * sx / scan_size))
            shift_y = int(3 * np.cos(2 * np.pi * sy / scan_size))
            dp = np.roll(np.roll(base_dp, shift_x, axis=1), shift_y, axis=0)

            # Position-dependent intensity variation
            intensity_factor = (
                0.8
                + 0.4
                * np.sin(np.pi * sx / scan_size)
                * np.cos(np.pi * sy / scan_size)
            )
            data[sx, sy] = dp * intensity_factor

    return data
