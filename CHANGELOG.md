# Changelog

All notable changes to `bobleesj.widget` will be documented in this file.

## Dec 25, 2025

### Added
- Show3D: ROI shapes (circle, square, rectangle) with adjustable width/height
- Show3D: Reset button and zoom indicator (appears when zoomed)
- Show3D: 1000-frame stress test in notebook
- Show3D demo screenshot in README

### Changed
- Show3D: Default fps changed from 5.0 to 1.0 (1000ms per frame)
- Show3D: Reorganized controls layout (playback row + display row)
- Show3D: Crosshair only shows while dragging ROI
- Show3D: Stats box limited to content width

## Dec 24, 2025

### Added
- `show_fft` parameter - shows FFT and histogram panels together
- `show_controls` parameter - toggle visibility of control row (Log, Auto, FFT toggles, colormap)
- `panel_size_px` parameter - control size of FFT and histogram panels
- Support for different-sized images in gallery mode (auto-resize to largest)
- Image size testing notebook (`show2d_single_sizes.ipynb`)
- Test for different-sized images in gallery mode

### Changed
- Default single image size increased from 200px to 300px
- Simplified API: merged `show_fft` and `show_histogram` into single `show_fft` parameter
- Simplified API: merged `fft_panel_size_px` and `histogram_panel_size_px` into single `panel_size_px`

### Fixed
- Zoom-to-cursor not zooming toward mouse position correctly
- White background flash in Jupyter/VS Code output cells (now dark)
- Histogram not rendering on initial load (added effect dependencies)
- Stats box overflow beyond cell width (added `width: fit-content`)
- Show4DSTEM import error when `bobleesj.detector` not installed (made optional)
- CI failures due to missing numpy/traitlets dependencies

## Dec 23, 2025

### Added
- Initial release of `bobleesj.widget`
- `Show2D` - Static 2D image viewer with FFT and histogram
- `Show3D` - Interactive 3D stack viewer with playback
- `Show4DSTEM` - 4D-STEM data viewer (requires `bobleesj.detector`)
- Scale bar with customizable length, thickness, and font size
- Multiple colormap options (inferno, viridis, magma, plasma, gray)
- Zoom/pan with scroll and drag
- Gallery mode for multiple images
- WebGPU-accelerated FFT
