"""
Tests for bobleesj.widget visualization widgets.

Tests cover:
- Array backends (NumPy, CuPy, PyTorch)
- Various array sizes
- Widget properties and data transfer
"""

import numpy as np
import pytest

from bobleesj.widget import Show2D, Show3D, Show4DSTEM
from bobleesj.widget.array_utils import to_numpy, get_array_backend


# =============================================================================
# Array Utils Tests
# =============================================================================

class TestArrayUtils:
    """Tests for array_utils module."""
    
    def test_get_array_backend_numpy(self):
        arr = np.array([1, 2, 3])
        assert get_array_backend(arr) == "numpy"
    
    def test_get_array_backend_list(self):
        arr = [1, 2, 3]
        assert get_array_backend(arr) == "unknown"
    
    def test_to_numpy_passthrough(self):
        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)
    
    def test_to_numpy_list(self):
        arr = [1, 2, 3]
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])


# =============================================================================
# Show2D Tests
# =============================================================================

class TestShow2D:
    """Tests for Show2D widget."""
    
    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_sizes_numpy(self, size):
        """Test Show2D with various sizes."""
        data = np.random.rand(size, size).astype(np.float32)
        widget = Show2D(data)
        
        assert widget.n_images == 1
        assert widget.height == size
        assert widget.width == size
        assert len(widget.frame_bytes) == size * size
    
    def test_3d_input(self):
        """Test Show2D with 3D array (gallery mode)."""
        data = np.random.rand(5, 64, 64).astype(np.float32)
        widget = Show2D(data)
        
        assert widget.n_images == 5
        assert widget.height == 64
        assert widget.width == 64
    
    def test_list_input(self):
        """Test Show2D with list of 2D arrays."""
        images = [np.random.rand(64, 64) for _ in range(3)]
        widget = Show2D(images)
        
        assert widget.n_images == 3
        assert widget.height == 64
        assert widget.width == 64
    
    def test_labels(self):
        """Test Show2D with labels."""
        data = np.random.rand(3, 64, 64).astype(np.float32)
        labels = ["A", "B", "C"]
        widget = Show2D(data, labels=labels)
        
        assert widget.labels == labels

    def test_custom_params(self):
        """Test Show2D customization parameters."""
        data = np.random.rand(64, 64).astype(np.float32)
        widget = Show2D(data, 
                       scale_bar_length_px=100,
                       scale_bar_thickness_px=10,
                       fft_panel_size_px=200,
                       image_width_px=500)
        
        assert widget.scale_bar_length_px == 100
        assert widget.scale_bar_thickness_px == 10
        assert widget.fft_panel_size_px == 200
        assert widget.image_width_px == 500

    def test_full_parameter_suite(self):
        """Test initialization with ALL parameters set to non-default values."""
        data = np.random.rand(10, 64, 64).astype(np.float32)
        labels = [f"L{i}" for i in range(10)]
        
        from bobleesj.widget import Colormap
        
        widget = Show2D(
            data,
            labels=labels,
            title="Full Test",
            cmap=Colormap.VIRIDIS,  # Enum usage
            pixel_size_angstrom=0.5,
            scale_bar_visible=False,
            show_fft=True,
            show_histogram=True,
            show_stats=False,
            log_scale=True,
            auto_contrast=True,
            ncols=5,
            # Customization
            scale_bar_length_px=75,
            scale_bar_thickness_px=6,
            scale_bar_font_size_px=20,
            fft_panel_size_px=250,
            image_width_px=600
        )
        
        # Verify all traits
        assert widget.n_images == 10
        assert widget.labels == labels
        assert widget.title == "Full Test"
        assert widget.cmap == "viridis"
        assert widget.pixel_size_angstrom == 0.5
        assert widget.scale_bar_visible == False
        assert widget.show_fft == True
        assert widget.show_histogram == True
        assert widget.show_stats == False
        assert widget.log_scale == True
        assert widget.auto_contrast == True
        assert widget.ncols == 5
        assert widget.scale_bar_length_px == 75
        assert widget.scale_bar_thickness_px == 6
        assert widget.scale_bar_font_size_px == 20
        assert widget.fft_panel_size_px == 250
        assert widget.image_width_px == 600


# =============================================================================
# Show3D Tests
# =============================================================================

class TestShow3D:
    """Tests for Show3D widget."""
    
    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_sizes_numpy(self, size):
        """Test Show3D with various sizes."""
        data = np.random.rand(10, size, size).astype(np.float32)
        widget = Show3D(data)
        
        assert widget.n_slices == 10
        assert widget.height == size
        assert widget.width == size
        assert len(widget.frame_bytes) == size * size
    
    @pytest.mark.parametrize("n_slices", [5, 10, 20])
    def test_slice_counts(self, n_slices):
        """Test Show3D with different numbers of slices."""
        data = np.random.rand(n_slices, 64, 64).astype(np.float32)
        widget = Show3D(data)
        
        assert widget.n_slices == n_slices
    
    def test_labels(self):
        """Test Show3D with labels."""
        data = np.random.rand(5, 64, 64).astype(np.float32)
        labels = [f"Frame {i}" for i in range(5)]
        widget = Show3D(data, labels=labels)
        
        assert widget.labels == labels
    
    def test_invalid_ndim(self):
        """Test Show3D raises error for non-3D input."""
        data = np.random.rand(64, 64).astype(np.float32)
        with pytest.raises(ValueError, match="Expected 3D array"):
            Show3D(data)


# =============================================================================
# Show4DSTEM Tests
# =============================================================================

class TestShow4DSTEM:
    """Tests for Show4DSTEM widget."""
    
    @pytest.mark.parametrize("scan_size,det_size", [
        (16, 32),
        (32, 64),
        (32, 128),
    ])
    def test_sizes_4d(self, scan_size, det_size):
        """Test Show4DSTEM with various 4D sizes."""
        data = np.random.rand(scan_size, scan_size, det_size, det_size).astype(np.float32)
        widget = Show4DSTEM(data)
        
        assert widget.shape_x == scan_size
        assert widget.shape_y == scan_size
        assert widget.det_x == det_size
        assert widget.det_y == det_size
        assert len(widget.frame_bytes) == det_size * det_size
        assert len(widget.virtual_image_bytes) == scan_size * scan_size
    
    def test_3d_input_square_scan(self):
        """Test Show4DSTEM with 3D input (flattened scan)."""
        # 16x16 scan = 256 positions, 32x32 detector
        data = np.random.rand(256, 32, 32).astype(np.float32)
        widget = Show4DSTEM(data, scan_shape=(16, 16))
        
        assert widget.shape_x == 16
        assert widget.shape_y == 16
        assert widget.det_x == 32
        assert widget.det_y == 32
    
    def test_3d_input_infer_square(self):
        """Test Show4DSTEM infers square scan shape from 3D input."""
        # 100 positions -> 10x10 inferred
        data = np.random.rand(100, 32, 32).astype(np.float32)
        widget = Show4DSTEM(data)
        
        assert widget.shape_x == 10
        assert widget.shape_y == 10
    
    def test_position_bounds(self):
        """Test that position is within scan bounds."""
        data = np.random.rand(16, 16, 32, 32).astype(np.float32)
        widget = Show4DSTEM(data)
        
        # Position should be initialized to center
        assert 0 <= widget.pos_x < 16
        assert 0 <= widget.pos_y < 16
    
    def test_roi_properties(self):
        """Test ROI properties are initialized."""
        data = np.random.rand(16, 16, 32, 32).astype(np.float32)
        widget = Show4DSTEM(data)
        
        assert hasattr(widget, 'roi_center_x')
        assert hasattr(widget, 'roi_center_y')
        assert hasattr(widget, 'roi_radius')
        assert widget.roi_active == True
    
    @pytest.mark.parametrize("scan_size,det_size", [
        (64, 128),   # Realistic: 64x64 scan, 128x128 detector
        (48, 192),   # Medium scan, larger detector
        (32, 64),    # Smaller test
    ])
    def test_realistic_sizes(self, scan_size, det_size):
        """Test Show4DSTEM with realistic 4D-STEM sizes."""
        data = np.random.rand(scan_size, scan_size, det_size, det_size).astype(np.float32)
        widget = Show4DSTEM(data)
        
        assert widget.shape_x == scan_size
        assert widget.shape_y == scan_size
        assert widget.det_x == det_size
        assert widget.det_y == det_size
        # Verify data transfer sizes
        assert len(widget.frame_bytes) == det_size * det_size
        assert len(widget.virtual_image_bytes) == scan_size * scan_size


# =============================================================================
# Backend Tests (Optional - skip if not available)
# =============================================================================

class TestCuPyBackend:
    """Tests for CuPy backend support."""
    
    @pytest.fixture(autouse=True)
    def check_cupy_available(self):
        """Skip tests if CuPy is not available or not working."""
        cupy_works = False
        try:
            import cupy as cp
            # Test that CuPy actually works
            test_arr = cp.array([1.0])
            result = test_arr + test_arr
            cp.cuda.Device().synchronize()
            _ = float(result.get())
            cupy_works = True
        except Exception:
            pass
        
        if not cupy_works:
            pytest.skip("CuPy not available or CUDA not working")
    
    def test_show2d_cupy(self):
        import cupy as cp
        data = cp.random.rand(64, 64).astype(cp.float32)
        widget = Show2D(data)
        
        assert widget.height == 64
        assert widget.width == 64
    
    def test_show3d_cupy(self):
        import cupy as cp
        data = cp.random.rand(10, 64, 64).astype(cp.float32)
        widget = Show3D(data)
        
        assert widget.n_slices == 10
    
    def test_show4dstem_cupy(self):
        import cupy as cp
        data = cp.random.rand(16, 16, 32, 32).astype(cp.float32)
        widget = Show4DSTEM(data)
        
        assert widget.shape_x == 16
        assert widget.det_x == 32


class TestPyTorchBackend:
    """Tests for PyTorch backend support."""
    
    @pytest.fixture(autouse=True)
    def skip_if_no_torch(self):
        """Skip tests if PyTorch is not available."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_show2d_torch(self):
        import torch
        data = torch.rand(64, 64)
        widget = Show2D(data)
        
        assert widget.height == 64
        assert widget.width == 64
    
    def test_show3d_torch(self):
        import torch
        data = torch.rand(10, 64, 64)
        widget = Show3D(data)
        
        assert widget.n_slices == 10
    
    def test_show4dstem_torch(self):
        import torch
        data = torch.rand(16, 16, 32, 32)
        widget = Show4DSTEM(data)
        
        assert widget.shape_x == 16
        assert widget.det_x == 32
    
    def test_array_backend_detection(self):
        import torch
        tensor = torch.rand(10, 10)
        assert get_array_backend(tensor) == "torch"
    
    def test_to_numpy_torch(self):
        import torch
        tensor = torch.rand(10, 10)
        result = to_numpy(tensor)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10)
