# bobleesj.widget

[![CI](https://github.com/bobleesj/bobleesj.widget/actions/workflows/ci.yml/badge.svg)](https://github.com/bobleesj/bobleesj.widget/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

Interactive Jupyter widgets for scientific image visualization. Works with NumPy, CuPy, and PyTorch arrays.

![Show2D Demo](assets/show2d_demo.png)

## Installation

```bash
git clone https://github.com/bobleesj/bobleesj.widget.git
cd bobleesj.widget
pip install -e .
```

No npm or Node.js needed – the JavaScript is pre-compiled.

## Usage

```python
import numpy as np
from bobleesj.widget import Show2D

# Just pass your array
image = np.random.rand(256, 256)
Show2D(image)
```

![Show2D Single](assets/show2d_single.png)

### With Options

```python
Show2D(
    image,
    title="My Image",
    pixel_size_angstrom=0.5,  # Enables scale bar
    show_fft=True,
    show_histogram=True,
)
```

### Gallery Mode

```python
# Display multiple images side by side
images = [img1, img2, img3]
Show2D(images)  # Labels auto-generated as "Image 1", "Image 2", etc.
```

![Show2D Gallery](assets/show2d_gallery.png)

## Interactive Controls

| Action | Control |
|--------|---------|
| **Zoom** | Scroll wheel |
| **Pan** | Click and drag |
| **Reset** | Double-click |
| **Resize** | Drag corner handle |

## Features

- 🖱️ **Interactive**: Zoom, pan, resize in the browser
- ⚡ **GPU-accelerated**: Real-time FFT via WebGPU
- 🔌 **Universal arrays**: NumPy, CuPy, PyTorch
- 📦 **Anywidget-based**: JupyterLab, VS Code, Colab

## API Reference

```python
Show2D(
    data,                       # 2D array or list of 2D arrays
    title="",                   # Title above the image
    labels=None,                # Labels for gallery mode
    cmap=Colormap.INFERNO,      # Colormap: inferno, viridis, magma, plasma, gray
    pixel_size_angstrom=0.0,    # Pixel size for scale bar (0 = no scale bar)
    scale_bar_visible=True,     # Show/hide scale bar
    show_fft=False,             # Show FFT panel
    show_histogram=False,       # Show histogram panel
    show_stats=True,            # Show statistics (mean, min, max, std)
    log_scale=False,            # Logarithmic intensity scaling
    auto_contrast=False,        # Percentile-based contrast
    ncols=3,                    # Columns in gallery mode
    image_width_px=0,           # Fixed width (0 = auto)
)
```

#### Which parameters should I use?

- **Just viewing an image?** → `Show2D(data)` is enough!
- **Need a scale bar?** → Add `pixel_size_angstrom=0.5`
- **Analyzing frequencies?** → Add `show_fft=True`
- **Comparing images?** → Pass a list: `Show2D([img1, img2, img3])`

📓 **See the example notebooks for complete tutorials:**
- [show2d_single.ipynb](notebooks/show2d_single.ipynb) – Single image with all options
- [show2d_multiple.ipynb](notebooks/show2d_multiple.ipynb) – Gallery mode and comparisons

## Requirements

- Python 3.11+
- JupyterLab, VS Code with Jupyter, or Google Colab

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for developer setup and guidelines.

## License

MIT
