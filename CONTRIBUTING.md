# Contributing to bobleesj.widget

Thank you for your interest in contributing! This guide covers the technical setup and coding standards.

## Developer Setup

For modifying TypeScript/React code, you'll need Node.js:

```bash
# Create environment with Node.js
mamba create -n widget-env python=3.12 numpy jupyterlab ipywidgets nodejs -y
mamba activate widget-env

# Clone and install
git clone https://github.com/bobleesj/bobleesj.widget.git
cd bobleesj.widget
pip install -e .

# Install npm dependencies (only needed for TypeScript development)
npm install
```

## Building TypeScript

After modifying any `.tsx` files in `js/`:

```bash
npm run build
```

## Project Structure

```
bobleesj.widget/
├── js/                     # TypeScript/React source
│   ├── show2d.tsx          # Show2D widget
│   ├── show3d.tsx          # Show3D widget
│   ├── core/               # Shared utilities (colors, canvas, etc.)
│   └── components.tsx      # Shared UI components
├── src/bobleesj/widget/    # Python source
│   ├── show2d.py           # Show2D Python class
│   ├── show3d.py           # Show3D Python class
│   └── static/             # Compiled JS (auto-generated)
├── notebooks/              # Example notebooks
└── tests/                  # pytest tests
```

## Guidelines

We prioritize **robustness** and **beautiful API design**.

### 1. Beautiful, Human-Centric API Design

- **Naming Matters**: Use clear, memorable names (e.g., `image_width_px` instead of `initial_image_width_px`).
- **Explicit over Implicit**: Prefer explicit arguments with units (e.g., `pixel_size_angstrom`) over generic configs.
- **Full Configuration**: Allow users to toggle behaviors easily (e.g., `scale_bar_visible=False`).

### 2. UI/UX Standards

- **Dropdowns open upward**: All `<Select>` components must use `upwardMenuProps` from `components.tsx` so dropdowns don't get cut off at the bottom of the widget.
- **Consistent styling**: Use shared constants from `core/` (colors, typography).
- **Zoom from center**: All zoom operations should zoom from the center of the image, not the top-left corner.

### 3. TypeScript Code Style

- Move types to top-level (e.g., `type ZoomState = { zoom: number; panX: number; panY: number }`)
- Use named constants instead of magic numbers (e.g., `DEFAULT_ZOOM_STATE`)
- Remove debug `console.log` statements before committing
- Use `React.useCallback` for handlers passed to child components

### 4. Python Code Style

- Use `StrEnum` for string enumerations (e.g., `Colormap`)
- All traits must have `.tag(sync=True)` for widget communication
- Support NumPy, CuPy, and PyTorch arrays via `array_utils.to_numpy()`

### 5. Testing

**Unit tests (`pytest`):**
- Every new argument must have a corresponding test case
- Test limits (e.g., 0 pixel size) and different input types

```bash
pytest tests/
```

**Visual verification:**
- Run example notebooks and visually verify the output
- Check: Does the scale bar make physical sense? Are fonts legible?

```bash
jupyter lab notebooks/show2d_single.ipynb
jupyter lab notebooks/show2d_multiple.ipynb
```

## Adding a New Widget

1. Create `js/mywidget.tsx` with React component
2. Create `src/bobleesj/widget/mywidget.py` with Python class
3. Add to `__init__.py` (use lazy import if it has external dependencies)
4. Add CSS file if needed (`js/mywidget.css`)
5. Run `npm run build`
6. Add tests in `tests/test_widgets.py`
7. Create example notebook in `notebooks/`

## Pull Request Checklist

- [ ] `npm run build` succeeds
- [ ] `pytest` passes
- [ ] No debug `console.log` statements
- [ ] Example notebook updated (if adding features)
- [ ] README updated (if adding user-facing features)
