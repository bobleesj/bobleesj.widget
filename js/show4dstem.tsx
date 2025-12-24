import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import Button from "@mui/material/Button";
import Switch from "@mui/material/Switch";
import JSZip from "jszip";
import { getWebGPUFFT, WebGPUFFT, getGPUInfo } from "./webgpu-fft";
import { COLORMAPS, fft1d, fft2d, fftshift, applyBandPassFilter, MIN_ZOOM, MAX_ZOOM } from "./shared";
import { colors, typography, controlPanel, container } from "./CONFIG";
import "./show4dstem.css";

// ============================================================================
// Constants
// ============================================================================
const RESIZE_HANDLE_RADIUS = 6;      // Size of the resize handle circle
const RESIZE_HIT_AREA_PX = 8;        // Click tolerance for resize handle
// Crosshair sizes: uses larger of (fraction of detector) or (minimum pixels)
const CROSSHAIR_FRACTION = 0.15;     // Fraction of detector size for point mode crosshair
const CROSSHAIR_FRACTION_SMALL = 0.10; // Fraction for ROI center crosshair
const CROSSHAIR_MIN_PX = 8;          // Minimum crosshair size in pixels
const CROSSHAIR_MIN_PX_SMALL = 5;    // Minimum small crosshair size in pixels
const CIRCLE_HANDLE_ANGLE = 0.707;   // cos(45°) for circle handle position
const SCALE_BAR_WIDTH_PX = 50;       // Scale bar width in screen pixels

// ============================================================================
// Scale Bar
// ============================================================================
function drawScaleBar(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  canvasHeight: number,
  zoom: number,
  pixelSize: number,
  unit: string = "nm"
) {
  const scaleBarPixels = 50;
  const realPixels = scaleBarPixels / zoom;
  const realSize = realPixels * pixelSize;

  let scaleLabel: string;
  if (unit === "nm") {
    if (realSize >= 1000) {
      scaleLabel = `${(realSize / 1000).toFixed(1)} µm`;
    } else if (realSize >= 1) {
      scaleLabel = `${Math.round(realSize)} nm`;
    } else {
      scaleLabel = `${realSize.toFixed(2)} nm`;
    }
  } else {
    scaleLabel = `${realSize.toFixed(1)} ${unit}`;
  }

  const barY = canvasHeight - 15;
  const barX = canvasWidth - scaleBarPixels - 10;

  ctx.fillStyle = "white";
  ctx.fillRect(barX, barY, scaleBarPixels, 4);

  ctx.font = "11px sans-serif";
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.fillText(scaleLabel, barX + scaleBarPixels / 2, barY - 5);

  ctx.textAlign = "left";
  ctx.fillText(`${zoom.toFixed(1)}×`, 10, canvasHeight - 10);
}

// ============================================================================
// Main Component
// ============================================================================
function Show4DSTEM() {
  // ─────────────────────────────────────────────────────────────────────────
  // Model State (synced with Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [shapeX] = useModelState<number>("shape_x");
  const [shapeY] = useModelState<number>("shape_y");
  const [detX] = useModelState<number>("det_x");
  const [detY] = useModelState<number>("det_y");

  const [posX, setPosX] = useModelState<number>("pos_x");
  const [posY, setPosY] = useModelState<number>("pos_y");
  const [roiCenterX, setRoiCenterX] = useModelState<number>("roi_center_x");
  const [roiCenterY, setRoiCenterY] = useModelState<number>("roi_center_y");
  const [, setRoiActive] = useModelState<boolean>("roi_active");

  const [pixelSize] = useModelState<number>("pixel_size");
  const [detPixelSize] = useModelState<number>("det_pixel_size");

  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [virtualImageBytes] = useModelState<DataView>("virtual_image_bytes");

  // ROI state
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiRadiusInner, setRoiRadiusInner] = useModelState<number>("roi_radius_inner");
  const [roiMode, setRoiMode] = useModelState<string>("roi_mode");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoRange, setAutoRange] = useModelState<boolean>("auto_range");
  const [percentileLow, setPercentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh, setPercentileHigh] = useModelState<number>("percentile_high");

  // Detector calibration (for presets)
  const [bfRadius] = useModelState<number>("bf_radius");
  const [centerX] = useModelState<number>("center_x");
  const [centerY] = useModelState<number>("center_y");

  // Path animation state
  const [pathPlaying, setPathPlaying] = useModelState<boolean>("path_playing");
  const [pathIndex, setPathIndex] = useModelState<number>("path_index");
  const [pathLength] = useModelState<number>("path_length");
  const [pathIntervalMs] = useModelState<number>("path_interval_ms");
  const [pathLoop] = useModelState<boolean>("path_loop");

  // ─────────────────────────────────────────────────────────────────────────
  // Local State (UI-only, not synced to Python)
  // ─────────────────────────────────────────────────────────────────────────
  const [localKx, setLocalKx] = React.useState(roiCenterX);
  const [localKy, setLocalKy] = React.useState(roiCenterY);
  const [localPosX, setLocalPosX] = React.useState(posX);
  const [localPosY, setLocalPosY] = React.useState(posY);
  const [isDraggingDP, setIsDraggingDP] = React.useState(false);
  const [isDraggingVI, setIsDraggingVI] = React.useState(false);
  const [isDraggingFFT, setIsDraggingFFT] = React.useState(false);
  const [fftDragStart, setFftDragStart] = React.useState<{ x: number, y: number, panX: number, panY: number } | null>(null);
  const [isDraggingResize, setIsDraggingResize] = React.useState(false);
  const [isDraggingResizeInner, setIsDraggingResizeInner] = React.useState(false); // For annular inner handle
  const [isHoveringResize, setIsHoveringResize] = React.useState(false);
  const [isHoveringResizeInner, setIsHoveringResizeInner] = React.useState(false);
  const [colormap, setColormap] = React.useState("inferno");

  // Band-pass filter range [innerCutoff, outerCutoff] in pixels - [0, 0] means disabled
  const [bandpass, setBandpass] = React.useState<number[]>([0, 0]);
  const bpInner = bandpass[0];
  const bpOuter = bandpass[1];

  // GPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Path animation timer
  React.useEffect(() => {
    if (!pathPlaying || pathLength === 0) return;

    const timer = setInterval(() => {
      setPathIndex((prev: number) => {
        const next = prev + 1;
        if (next >= pathLength) {
          if (pathLoop) {
            return 0;  // Loop back to start
          } else {
            setPathPlaying(false);  // Stop at end
            return prev;
          }
        }
        return next;
      });
    }, pathIntervalMs);

    return () => clearInterval(timer);
  }, [pathPlaying, pathLength, pathIntervalMs, pathLoop, setPathIndex, setPathPlaying]);

  // Keyboard shortcuts
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      const step = e.shiftKey ? 10 : 1;

      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault();
          setPosX(Math.max(0, posX - step));
          break;
        case 'ArrowDown':
          e.preventDefault();
          setPosX(Math.min(shapeX - 1, posX + step));
          break;
        case 'ArrowLeft':
          e.preventDefault();
          setPosY(Math.max(0, posY - step));
          break;
        case 'ArrowRight':
          e.preventDefault();
          setPosY(Math.min(shapeY - 1, posY + step));
          break;
        case ' ':  // Space bar
          e.preventDefault();
          if (pathLength > 0) {
            setPathPlaying(!pathPlaying);
          }
          break;
        case 'r':  // Reset view
        case 'R':
          setDpZoom(1); setDpPanX(0); setDpPanY(0);
          setViZoom(1); setViPanX(0); setViPanY(0);
          setFftZoom(1); setFftPanX(0); setFftPanY(0);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [posX, posY, shapeX, shapeY, pathPlaying, pathLength, setPosX, setPosY, setPathPlaying]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
        console.log("WebGPU FFT ready - using GPU acceleration!");
      } else {
        console.log("⚠️ WebGPU not available - using CPU FFT");
      }
    });
  }, []);

  // Zoom state
  const [dpZoom, setDpZoom] = React.useState(1);
  const [dpPanX, setDpPanX] = React.useState(0);
  const [dpPanY, setDpPanY] = React.useState(0);
  const [viZoom, setViZoom] = React.useState(1);
  const [viPanX, setViPanX] = React.useState(0);
  const [viPanY, setViPanY] = React.useState(0);
  const [fftZoom, setFftZoom] = React.useState(1);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);

  // Sync local state
  React.useEffect(() => {
    if (!isDraggingDP && !isDraggingResize) { setLocalKx(roiCenterX); setLocalKy(roiCenterY); }
  }, [roiCenterX, roiCenterY, isDraggingDP, isDraggingResize]);

  React.useEffect(() => {
    if (!isDraggingVI) { setLocalPosX(posX); setLocalPosY(posY); }
  }, [posX, posY, isDraggingVI]);

  // Canvas refs
  const dpCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const dpOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const virtualCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const virtualOverlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const fftOverlayRef = React.useRef<HTMLCanvasElement>(null);

  // ─────────────────────────────────────────────────────────────────────────
  // Effects: Canvas Rendering & Animation
  // ─────────────────────────────────────────────────────────────────────────

  // Prevent page scroll when scrolling on canvases
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const overlays = [dpOverlayRef.current, virtualOverlayRef.current, fftOverlayRef.current];
    overlays.forEach(el => el?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => overlays.forEach(el => el?.removeEventListener("wheel", preventDefault));
  }, []);

  // Store raw virtual image data for filtering
  const rawVirtualImageRef = React.useRef<Float32Array | null>(null);

  // Parse virtual image bytes into Float32Array
  React.useEffect(() => {
    if (!virtualImageBytes) return;
    const bytes = new Uint8Array(virtualImageBytes.buffer, virtualImageBytes.byteOffset, virtualImageBytes.byteLength);
    const floatData = new Float32Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) {
      floatData[i] = bytes[i];
    }
    rawVirtualImageRef.current = floatData;
  }, [virtualImageBytes]);

  // Render DP with zoom
  React.useEffect(() => {
    if (!frameBytes || !dpCanvasRef.current) return;
    const canvas = dpCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bytes = new Uint8Array(frameBytes.buffer, frameBytes.byteOffset, frameBytes.byteLength);
    const lut = COLORMAPS[colormap] || COLORMAPS.inferno;

    const offscreen = document.createElement("canvas");
    offscreen.width = detY;
    offscreen.height = detX;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    const imgData = offCtx.createImageData(detY, detX);
    const rgba = imgData.data;

    for (let i = 0; i < bytes.length; i++) {
      const v = bytes[i];
      const j = i * 4;
      const lutIdx = v * 3;
      rgba[j] = lut[lutIdx];
      rgba[j + 1] = lut[lutIdx + 1];
      rgba[j + 2] = lut[lutIdx + 2];
      rgba[j + 3] = 255;
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(dpPanX, dpPanY);
    ctx.scale(dpZoom, dpZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [frameBytes, detX, detY, colormap, dpZoom, dpPanX, dpPanY]);

  // Render DP overlay
  React.useEffect(() => {
    if (!dpOverlayRef.current) return;
    const canvas = dpOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const screenKx = localKx * dpZoom + dpPanX;
    const screenKy = localKy * dpZoom + dpPanY;

    // Calculate crosshair sizes proportional to detector size, with minimum values
    const minDetSize = Math.min(detX, detY);
    const crosshairSize = Math.max(CROSSHAIR_MIN_PX, minDetSize * CROSSHAIR_FRACTION * dpZoom);
    const crosshairSizeSmall = Math.max(CROSSHAIR_MIN_PX_SMALL, minDetSize * CROSSHAIR_FRACTION_SMALL * dpZoom);
    const centerDotRadius = Math.max(3, minDetSize * 0.03 * dpZoom);

    ctx.strokeStyle = isDraggingDP ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
    ctx.lineWidth = 2;

    // Helper to draw resize handle
    const drawResizeHandle = (handleX: number, handleY: number) => {
      let handleFill: string;
      let handleStroke: string;
      if (isDraggingResize) {
        handleFill = "rgba(0, 200, 255, 1)";   // Cyan when dragging
        handleStroke = "rgba(255, 255, 255, 1)";
      } else if (isHoveringResize) {
        handleFill = "rgba(255, 100, 100, 1)"; // Red when hovering
        handleStroke = "rgba(255, 255, 255, 1)";
      } else {
        handleFill = "rgba(0, 255, 0, 0.8)";   // Green default
        handleStroke = "rgba(255, 255, 255, 0.8)";
      }
      ctx.beginPath();
      ctx.arc(handleX, handleY, RESIZE_HANDLE_RADIUS, 0, 2 * Math.PI);
      ctx.fillStyle = handleFill;
      ctx.fill();
      ctx.strokeStyle = handleStroke;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    };

    if (roiMode === "circle" && roiRadius > 0) {
      // Circle mode: draw a filled circular ROI
      const screenRadius = roiRadius * dpZoom;
      ctx.beginPath();
      ctx.arc(screenKx, screenKy, screenRadius, 0, 2 * Math.PI);
      ctx.stroke();
      // Semi-transparent fill
      ctx.fillStyle = isDraggingDP ? "rgba(255, 255, 0, 0.15)" : "rgba(0, 255, 0, 0.15)";
      ctx.fill();
      // Draw center crosshair (smaller)
      ctx.strokeStyle = isDraggingDP ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(screenKx - crosshairSizeSmall, screenKy); ctx.lineTo(screenKx + crosshairSizeSmall, screenKy);
      ctx.moveTo(screenKx, screenKy - crosshairSizeSmall); ctx.lineTo(screenKx, screenKy + crosshairSizeSmall);
      ctx.stroke();

      // Draw resize handle at bottom-right of circle (45° position)
      const handleOffset = screenRadius * CIRCLE_HANDLE_ANGLE;
      drawResizeHandle(screenKx + handleOffset, screenKy + handleOffset);

    } else if (roiMode === "square" && roiRadius > 0) {
      // Square mode: draw a filled square ROI
      const screenHalfSize = roiRadius * dpZoom;
      const left = screenKx - screenHalfSize;
      const top = screenKy - screenHalfSize;
      const size = screenHalfSize * 2;

      ctx.beginPath();
      ctx.rect(left, top, size, size);
      ctx.stroke();
      // Semi-transparent fill
      ctx.fillStyle = isDraggingDP ? "rgba(255, 255, 0, 0.15)" : "rgba(0, 255, 0, 0.15)";
      ctx.fill();
      // Draw center crosshair (smaller)
      ctx.strokeStyle = isDraggingDP ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(screenKx - crosshairSizeSmall, screenKy); ctx.lineTo(screenKx + crosshairSizeSmall, screenKy);
      ctx.moveTo(screenKx, screenKy - crosshairSizeSmall); ctx.lineTo(screenKx, screenKy + crosshairSizeSmall);
      ctx.stroke();

      // Draw resize handle at bottom-right corner of square
      drawResizeHandle(screenKx + screenHalfSize, screenKy + screenHalfSize);

    } else if (roiMode === "rect" && roiWidth > 0 && roiHeight > 0) {
      // Rectangular mode: draw a filled rectangular ROI with independent width/height
      const screenHalfW = (roiWidth / 2) * dpZoom;
      const screenHalfH = (roiHeight / 2) * dpZoom;
      const left = screenKx - screenHalfW;
      const top = screenKy - screenHalfH;

      ctx.beginPath();
      ctx.rect(left, top, screenHalfW * 2, screenHalfH * 2);
      ctx.stroke();
      // Semi-transparent fill
      ctx.fillStyle = isDraggingDP ? "rgba(255, 255, 0, 0.15)" : "rgba(0, 255, 0, 0.15)";
      ctx.fill();
      // Draw center crosshair (smaller)
      ctx.strokeStyle = isDraggingDP ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(screenKx - crosshairSizeSmall, screenKy); ctx.lineTo(screenKx + crosshairSizeSmall, screenKy);
      ctx.moveTo(screenKx, screenKy - crosshairSizeSmall); ctx.lineTo(screenKx, screenKy + crosshairSizeSmall);
      ctx.stroke();

      // Draw resize handle at bottom-right corner of rectangle
      drawResizeHandle(screenKx + screenHalfW, screenKy + screenHalfH);

    } else if (roiMode === "annular" && roiRadius > 0) {
      // Annular mode: draw donut-shaped ROI (ADF/HAADF)
      const screenRadiusOuter = roiRadius * dpZoom;
      const screenRadiusInner = (roiRadiusInner || 0) * dpZoom;

      // Draw outer circle
      ctx.beginPath();
      ctx.arc(screenKx, screenKy, screenRadiusOuter, 0, 2 * Math.PI);
      ctx.stroke();

      // Draw inner circle (different color for distinction)
      ctx.save();
      ctx.strokeStyle = isDraggingDP ? "rgba(255, 200, 0, 0.9)" : "rgba(0, 220, 255, 0.9)"; // Cyan/orange for inner
      ctx.beginPath();
      ctx.arc(screenKx, screenKy, screenRadiusInner, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.restore();

      // Fill the annular region (donut) using composite operation
      ctx.save();
      ctx.fillStyle = isDraggingDP ? "rgba(255, 255, 0, 0.15)" : "rgba(0, 255, 0, 0.15)";
      ctx.beginPath();
      ctx.arc(screenKx, screenKy, screenRadiusOuter, 0, 2 * Math.PI);
      ctx.arc(screenKx, screenKy, screenRadiusInner, 0, 2 * Math.PI, true); // counter-clockwise to cut hole
      ctx.fill();
      ctx.restore();

      // Draw center crosshair (smaller)
      ctx.strokeStyle = isDraggingDP ? "rgba(255, 255, 0, 0.9)" : "rgba(0, 255, 0, 0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(screenKx - crosshairSizeSmall, screenKy); ctx.lineTo(screenKx + crosshairSizeSmall, screenKy);
      ctx.moveTo(screenKx, screenKy - crosshairSizeSmall); ctx.lineTo(screenKx, screenKy + crosshairSizeSmall);
      ctx.stroke();

      // Draw resize handle at outer circle's 45° position (green)
      const handleOffsetOuter = screenRadiusOuter * CIRCLE_HANDLE_ANGLE;
      drawResizeHandle(screenKx + handleOffsetOuter, screenKy + handleOffsetOuter);

      // Draw resize handle at inner circle's 45° position (cyan)
      ctx.save();
      ctx.strokeStyle = isHoveringResizeInner ? "rgba(0, 220, 255, 1)" : "rgba(0, 220, 255, 0.8)";
      ctx.fillStyle = "rgba(0, 40, 50, 0.8)";
      const handleOffsetInner = screenRadiusInner * CIRCLE_HANDLE_ANGLE;
      ctx.beginPath();
      ctx.arc(screenKx + handleOffsetInner, screenKy + handleOffsetInner, RESIZE_HANDLE_RADIUS, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

    } else {
      // Point mode: draw crosshair
      ctx.beginPath();
      ctx.moveTo(screenKx - crosshairSize, screenKy); ctx.lineTo(screenKx + crosshairSize, screenKy);
      ctx.moveTo(screenKx, screenKy - crosshairSize); ctx.lineTo(screenKx, screenKy + crosshairSize);
      ctx.stroke();
      ctx.beginPath(); ctx.arc(screenKx, screenKy, centerDotRadius, 0, 2 * Math.PI); ctx.stroke();
    }

    drawScaleBar(ctx, canvas.width, canvas.height, dpZoom, detPixelSize || 1, "mrad");
  }, [localKx, localKy, isDraggingDP, isDraggingResize, isDraggingResizeInner, isHoveringResize, isHoveringResizeInner, dpZoom, dpPanX, dpPanY, detPixelSize, roiMode, roiRadius, roiRadiusInner, roiWidth, roiHeight, detX, detY]);

  // Render filtered virtual image
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !virtualCanvasRef.current) return;
    const canvas = virtualCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = shapeY;
    const height = shapeX;

    const renderData = (filtered: Float32Array) => {
      // Normalize and render (with optional percentile contrast)
      let min = Infinity, max = -Infinity;

      if (autoRange) {
        // Percentile-based contrast: sort a sample and pick percentile values
        const sorted = Float32Array.from(filtered).sort((a, b) => a - b);
        const lowIdx = Math.floor((percentileLow / 100) * sorted.length);
        const highIdx = Math.floor((percentileHigh / 100) * sorted.length) - 1;
        min = sorted[Math.max(0, lowIdx)];
        max = sorted[Math.min(sorted.length - 1, highIdx)];
      } else {
        // Full range
        for (let i = 0; i < filtered.length; i++) {
          if (filtered[i] < min) min = filtered[i];
          if (filtered[i] > max) max = filtered[i];
        }
      }

      const lut = COLORMAPS[colormap] || COLORMAPS.inferno;
      const offscreen = document.createElement("canvas");
      offscreen.width = width;
      offscreen.height = height;
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      const imageData = offCtx.createImageData(width, height);
      for (let i = 0; i < filtered.length; i++) {
        const val = Math.floor(((filtered[i] - min) / (max - min || 1)) * 255);
        imageData.data[i * 4] = lut[val * 3];
        imageData.data[i * 4 + 1] = lut[val * 3 + 1];
        imageData.data[i * 4 + 2] = lut[val * 3 + 2];
        imageData.data[i * 4 + 3] = 255;
      }
      offCtx.putImageData(imageData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(viPanX, viPanY);
      ctx.scale(viZoom, viZoom);
      ctx.drawImage(offscreen, 0, 0);
      ctx.restore();
    };

    if (bpInner > 0 || bpOuter > 0) {
      if (gpuFFTRef.current && gpuReady) {
        // GPU filtering (Async)
        const real = rawVirtualImageRef.current.slice();
        const imag = new Float32Array(real.length);

        // We use a local flag to prevent state updates if the effect has already re-run
        let isCancelled = false;

        const runGpuFilter = async () => {
          // WebGPU version of: Forward -> Filter -> Inverse
          // Note: The provided WebGPUFFT doesn't have shift/unshift built-in yet, 
          // but we can apply the filter in shifted coordinates or modify it.
          // For now, let's keep it simple: Forward -> Filter -> Inverse.
          const { real: fReal, imag: fImag } = await gpuFFTRef.current!.fft2D(real, imag, width, height, false);

          if (isCancelled) return;

          // Shift in CPU for now (future: do this in WGSL)
          fftshift(fReal, width, height);
          fftshift(fImag, width, height);
          applyBandPassFilter(fReal, fImag, width, height, bpInner, bpOuter);
          fftshift(fReal, width, height);
          fftshift(fImag, width, height);

          const { real: invReal } = await gpuFFTRef.current!.fft2D(fReal, fImag, width, height, true);

          if (!isCancelled) renderData(invReal);
        };

        runGpuFilter();
        return () => { isCancelled = true; };
      } else {
        // CPU Fallback (Sync)
        const real = rawVirtualImageRef.current.slice();
        const imag = new Float32Array(real.length);
        fft2d(real, imag, width, height, false);
        fftshift(real, width, height);
        fftshift(imag, width, height);
        applyBandPassFilter(real, imag, width, height, bpInner, bpOuter);
        fftshift(real, width, height);
        fftshift(imag, width, height);
        fft2d(real, imag, width, height, true);
        renderData(real);
      }
    } else {
      renderData(rawVirtualImageRef.current);
    }
  }, [virtualImageBytes, shapeX, shapeY, colormap, viZoom, viPanX, viPanY, bpInner, bpOuter, gpuReady, autoRange, percentileLow, percentileHigh]);

  // Render virtual image overlay
  React.useEffect(() => {
    if (!virtualOverlayRef.current) return;
    const canvas = virtualOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const screenX = localPosY * viZoom + viPanX;
    const screenY = localPosX * viZoom + viPanY;

    ctx.strokeStyle = isDraggingVI ? "rgba(255, 255, 0, 0.9)" : "rgba(255, 100, 100, 0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(screenX - 8, screenY); ctx.lineTo(screenX + 8, screenY);
    ctx.moveTo(screenX, screenY - 8); ctx.lineTo(screenX, screenY + 8);
    ctx.stroke();
    ctx.beginPath(); ctx.arc(screenX, screenY, 4, 0, 2 * Math.PI); ctx.stroke();

    drawScaleBar(ctx, canvas.width, canvas.height, viZoom, pixelSize || 1, "nm");
  }, [localPosX, localPosY, isDraggingVI, viZoom, viPanX, viPanY, pixelSize]);

  // Render FFT (computed in JS from filtered image)
  React.useEffect(() => {
    if (!rawVirtualImageRef.current || !fftCanvasRef.current) return;
    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = shapeY;
    const height = shapeX;

    // Use raw or filtered data
    let sourceData = rawVirtualImageRef.current;
    if (bpInner > 0 || bpOuter > 0) {
      sourceData = rawVirtualImageRef.current.slice();
    }

    const real = sourceData.slice();
    const imag = new Float32Array(real.length);

    // Forward FFT
    fft2d(real, imag, width, height, false);
    fftshift(real, width, height);
    fftshift(imag, width, height);

    // Compute log magnitude
    const magnitude = new Float32Array(real.length);
    for (let i = 0; i < real.length; i++) {
      magnitude[i] = Math.log1p(Math.sqrt(real[i] * real[i] + imag[i] * imag[i]));
    }

    // Normalize
    let min = Infinity, max = -Infinity;
    for (let i = 0; i < magnitude.length; i++) {
      if (magnitude[i] < min) min = magnitude[i];
      if (magnitude[i] > max) max = magnitude[i];
    }

    const lut = COLORMAPS[colormap] || COLORMAPS.inferno;
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    const imgData = offCtx.createImageData(width, height);
    const rgba = imgData.data;
    const range = max > min ? max - min : 1;

    for (let i = 0; i < magnitude.length; i++) {
      const v = Math.round(((magnitude[i] - min) / range) * 255);
      const j = i * 4;
      const lutIdx = Math.max(0, Math.min(255, v)) * 3;
      rgba[j] = lut[lutIdx];
      rgba[j + 1] = lut[lutIdx + 1];
      rgba[j + 2] = lut[lutIdx + 2];
      rgba[j + 3] = 255;
    }
    offCtx.putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(fftPanX, fftPanY);
    ctx.scale(fftZoom, fftZoom);
    ctx.drawImage(offscreen, 0, 0);
    ctx.restore();
  }, [virtualImageBytes, shapeX, shapeY, colormap, fftZoom, fftPanX, fftPanY, bpInner, bpOuter]);

  // Render FFT overlay with high-pass filter circle
  React.useEffect(() => {
    if (!fftOverlayRef.current) return;
    const canvas = fftOverlayRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw band-pass filter circles (inner = HP, outer = LP)
    const centerX = (shapeY / 2) * fftZoom + fftPanX;
    const centerY = (shapeX / 2) * fftZoom + fftPanY;

    if (bpInner > 0) {
      ctx.strokeStyle = "rgba(255, 0, 0, 0.8)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, bpInner * fftZoom, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (bpOuter > 0) {
      ctx.strokeStyle = "rgba(0, 150, 255, 0.8)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.arc(centerX, centerY, bpOuter * fftZoom, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const fftPixelSize = pixelSize ? 1 / (shapeX * pixelSize) : 1;
    drawScaleBar(ctx, canvas.width, canvas.height, fftZoom, fftPixelSize * 1000, "1/µm");
  }, [fftZoom, fftPanX, fftPanY, pixelSize, shapeX, shapeY, bpInner, bpOuter]);

  // Generic zoom handler
  const createZoomHandler = (
    setZoom: React.Dispatch<React.SetStateAction<number>>,
    setPanX: React.Dispatch<React.SetStateAction<number>>,
    setPanY: React.Dispatch<React.SetStateAction<number>>,
    zoom: number, panX: number, panY: number,
    canvasRef: React.RefObject<HTMLCanvasElement | null>
  ) => (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom * zoomFactor));
    const zoomRatio = newZoom / zoom;
    setZoom(newZoom);
    setPanX(mouseX - (mouseX - panX) * zoomRatio);
    setPanY(mouseY - (mouseY - panY) * zoomRatio);
  };

  // ─────────────────────────────────────────────────────────────────────────
  // Mouse Handlers
  // ─────────────────────────────────────────────────────────────────────────

  // Helper: check if point is near the outer resize handle
  const isNearResizeHandle = (imgX: number, imgY: number): boolean => {
    if (roiMode === "rect") {
      // For rectangle, check near bottom-right corner
      const handleX = roiCenterX + roiWidth / 2;
      const handleY = roiCenterY + roiHeight / 2;
      const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
      return dist < RESIZE_HIT_AREA_PX / dpZoom;
    }
    if ((roiMode !== "circle" && roiMode !== "square" && roiMode !== "annular") || !roiRadius) return false;
    const offset = roiMode === "square" ? roiRadius : roiRadius * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterX + offset;
    const handleY = roiCenterY + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Helper: check if point is near the inner resize handle (annular mode only)
  const isNearResizeHandleInner = (imgX: number, imgY: number): boolean => {
    if (roiMode !== "annular" || !roiRadiusInner) return false;
    const offset = roiRadiusInner * CIRCLE_HANDLE_ANGLE;
    const handleX = roiCenterX + offset;
    const handleY = roiCenterY + offset;
    const dist = Math.sqrt((imgX - handleX) ** 2 + (imgY - handleY) ** 2);
    return dist < RESIZE_HIT_AREA_PX / dpZoom;
  };

  // Mouse handlers
  const handleDpMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Check if clicking on resize handle (inner first, then outer)
    if (isNearResizeHandleInner(imgX, imgY)) {
      setIsDraggingResizeInner(true);
      return;
    }
    if (isNearResizeHandle(imgX, imgY)) {
      setIsDraggingResize(true);
      return;
    }

    setIsDraggingDP(true);
    setLocalKx(imgX); setLocalKy(imgY);
    setRoiActive(true);
    setRoiCenterX(Math.round(Math.max(0, Math.min(detY - 1, imgX))));
    setRoiCenterY(Math.round(Math.max(0, Math.min(detX - 1, imgY))));
  };

  const handleDpMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = dpOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenX - dpPanX) / dpZoom;
    const imgY = (screenY - dpPanY) / dpZoom;

    // Handle inner resize dragging (annular mode)
    if (isDraggingResizeInner) {
      const dx = Math.abs(imgX - roiCenterX);
      const dy = Math.abs(imgY - roiCenterY);
      const newRadius = Math.sqrt(dx ** 2 + dy ** 2);
      // Inner radius must be less than outer radius
      setRoiRadiusInner(Math.max(1, Math.min(roiRadius - 1, Math.round(newRadius))));
      return;
    }

    // Handle outer resize dragging - use model state center, not local values
    if (isDraggingResize) {
      const dx = Math.abs(imgX - roiCenterX);
      const dy = Math.abs(imgY - roiCenterY);
      if (roiMode === "rect") {
        // For rectangle, update width and height independently
        setRoiWidth(Math.max(2, Math.round(dx * 2)));
        setRoiHeight(Math.max(2, Math.round(dy * 2)));
      } else {
        const newRadius = roiMode === "square" ? Math.max(dx, dy) : Math.sqrt(dx ** 2 + dy ** 2);
        // For annular mode, outer radius must be greater than inner radius
        const minRadius = roiMode === "annular" ? (roiRadiusInner || 0) + 1 : 1;
        setRoiRadius(Math.max(minRadius, Math.round(newRadius)));
      }
      return;
    }

    // Check hover state for resize handles
    if (!isDraggingDP) {
      setIsHoveringResizeInner(isNearResizeHandleInner(imgX, imgY));
      setIsHoveringResize(isNearResizeHandle(imgX, imgY));
      return;
    }

    setLocalKx(imgX); setLocalKy(imgY);
    setRoiCenterX(Math.round(Math.max(0, Math.min(detY - 1, imgX))));
    setRoiCenterY(Math.round(Math.max(0, Math.min(detX - 1, imgY))));
  };

  const handleDpMouseUp = () => { setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false); };
  const handleDpMouseLeave = () => { setIsDraggingDP(false); setIsDraggingResize(false); setIsDraggingResizeInner(false); setIsHoveringResize(false); setIsHoveringResizeInner(false); };
  const handleDpDoubleClick = () => { setDpZoom(1); setDpPanX(0); setDpPanY(0); };

  const handleViMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;
    setIsDraggingVI(true);
    setLocalPosX(imgX); setLocalPosY(imgY);
    setPosX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
    setPosY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
  };

  const handleViMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingVI) return;
    const canvas = virtualOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const screenX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const screenY = (e.clientY - rect.top) * (canvas.height / rect.height);
    const imgX = (screenY - viPanY) / viZoom;
    const imgY = (screenX - viPanX) / viZoom;
    setLocalPosX(imgX); setLocalPosY(imgY);
    setPosX(Math.round(Math.max(0, Math.min(shapeX - 1, imgX))));
    setPosY(Math.round(Math.max(0, Math.min(shapeY - 1, imgY))));
  };

  const handleViMouseUp = () => setIsDraggingVI(false);
  const handleViDoubleClick = () => { setViZoom(1); setViPanX(0); setViPanY(0); };
  const handleFftDoubleClick = () => { setFftZoom(1); setFftPanX(0); setFftPanY(0); };

  // FFT drag-to-pan handlers
  const handleFftMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDraggingFFT(true);
    setFftDragStart({ x: e.clientX, y: e.clientY, panX: fftPanX, panY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingFFT || !fftDragStart) return;
    const canvas = fftOverlayRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - fftDragStart.x) * scaleX;
    const dy = (e.clientY - fftDragStart.y) * scaleY;
    setFftPanX(fftDragStart.panX + dx);
    setFftPanY(fftDragStart.panY + dy);
  };

  const handleFftMouseUp = () => { setIsDraggingFFT(false); setFftDragStart(null); };
  const handleFftMouseLeave = () => { setIsDraggingFFT(false); setFftDragStart(null); };

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <Box className="show4dstem-root" sx={{ ...container.root, maxWidth: 1100 }}>
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h6" sx={{ ...typography.title }}>
          4D-STEM Explorer
        </Typography>
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography sx={{ ...typography.labelSmall }}>
            {shapeX}×{shapeY} scan | {detX}×{detY} det
          </Typography>
          <Typography
            component="span"
            onClick={() => {
              setBandpass([0, 0]);
              setDpZoom(1); setDpPanX(0); setDpPanY(0);
              setViZoom(1); setViPanX(0); setViPanY(0);
              setFftZoom(1); setFftPanX(0); setFftPanY(0);
              setRoiMode("point");
            }}
            sx={{ ...controlPanel.button }}
          >
            Reset
          </Typography>
        </Stack>
      </Stack>

      <Stack direction="row" spacing={2}>
        {/* LEFT: DP */}
        <Box>
          <Typography variant="caption" sx={{ ...typography.label, mb: 0.5, display: "block" }}>
            DP at ({Math.round(localPosX)}, {Math.round(localPosY)})
            <span style={{ color: "#0f0", marginLeft: 8 }}>k: ({Math.round(localKx)}, {Math.round(localKy)})</span>
          </Typography>
          <Box sx={{ ...container.imageBox, width: 400, height: 400 }}>
            <canvas ref={dpCanvasRef} width={detY} height={detX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
            <canvas
              ref={dpOverlayRef} width={detY} height={detX}
              onMouseDown={handleDpMouseDown} onMouseMove={handleDpMouseMove}
              onMouseUp={handleDpMouseUp} onMouseLeave={handleDpMouseLeave}
              onWheel={createZoomHandler(setDpZoom, setDpPanX, setDpPanY, dpZoom, dpPanX, dpPanY, dpOverlayRef)}
              onDoubleClick={handleDpDoubleClick}
              style={{
                position: "absolute",
                width: "100%",
                height: "100%",
                cursor: isHoveringResize || isDraggingResize ? "nwse-resize" : "crosshair"
              }}
            />
          </Box>
        </Box>
        {/* RIGHT: Virtual Image + FFT */}
        <Stack spacing={1}>
          <Box>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.5 }}>
              <Typography variant="caption" sx={{ ...typography.label }}>
                Virtual Image {(bpInner > 0 || bpOuter > 0) && (
                  <span>
                    {bpInner > 0 && <span style={{ color: "#f66" }}> HP:{bpInner}</span>}
                    {bpOuter > 0 && <span style={{ color: "#09f" }}> LP:{bpOuter}</span>}
                  </span>
                )}
              </Typography>
              <Typography
                component="span"
                onClick={async () => {
                  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
                  const zip = new JSZip();

                  // Add metadata JSON
                  const metadata = {
                    exported_at: new Date().toISOString(),
                    scan_position: { x: posX, y: posY },
                    scan_shape: { x: shapeX, y: shapeY },
                    detector_shape: { x: detX, y: detY },
                    roi: {
                      mode: roiMode,
                      center_x: roiCenterX,
                      center_y: roiCenterY,
                      radius_outer: roiRadius,
                      radius_inner: roiRadiusInner,
                    },
                    display: {
                      colormap: colormap,
                      log_scale: logScale,
                      auto_range: autoRange,
                      percentile_low: percentileLow,
                      percentile_high: percentileHigh,
                    },
                    calibration: {
                      bf_radius: bfRadius,
                      center_x: centerX,
                      center_y: centerY,
                      pixel_size: pixelSize,
                      det_pixel_size: detPixelSize,
                    },
                  };
                  zip.file("metadata.json", JSON.stringify(metadata, null, 2));

                  // Helper to convert canvas to blob
                  const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => {
                    return new Promise((resolve) => {
                      canvas.toBlob((blob) => resolve(blob!), 'image/png');
                    });
                  };

                  // Add images
                  const viCanvas = virtualCanvasRef.current;
                  if (viCanvas) {
                    const blob = await canvasToBlob(viCanvas);
                    zip.file("virtual_image.png", blob);
                  }
                  const dpCanvas = dpCanvasRef.current;
                  if (dpCanvas) {
                    const blob = await canvasToBlob(dpCanvas);
                    zip.file("diffraction_pattern.png", blob);
                  }
                  const fftCanvas = fftCanvasRef.current;
                  if (fftCanvas) {
                    const blob = await canvasToBlob(fftCanvas);
                    zip.file("fft.png", blob);
                  }

                  // Generate and download ZIP
                  const zipBlob = await zip.generateAsync({ type: "blob" });
                  const link = document.createElement('a');
                  link.download = `4dstem_export_${timestamp}.zip`;
                  link.href = URL.createObjectURL(zipBlob);
                  link.click();
                  URL.revokeObjectURL(link.href);
                }}
                sx={{ ...controlPanel.button }}
              >
                Export
              </Typography>
            </Stack>
            <Stack direction="row" spacing={1}>
              <Box sx={{ ...container.imageBox, width: 300, height: 300 }}>
                <canvas ref={virtualCanvasRef} width={shapeY} height={shapeX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
                <canvas
                  ref={virtualOverlayRef} width={shapeY} height={shapeX}
                  onMouseDown={handleViMouseDown} onMouseMove={handleViMouseMove}
                  onMouseUp={handleViMouseUp} onMouseLeave={handleViMouseUp}
                  onWheel={createZoomHandler(setViZoom, setViPanX, setViPanY, viZoom, viPanX, viPanY, virtualOverlayRef)}
                  onDoubleClick={handleViDoubleClick}
                  style={{ position: "absolute", width: "100%", height: "100%", cursor: "crosshair" }}
                />
              </Box>
              {/* Band-pass Filter slider (dual handles) */}
              <Stack spacing={0.5} sx={{ height: 300 }} alignItems="center">
                <Typography sx={{ color: "#09f", fontSize: 9 }}>{bpOuter || "-"}</Typography>
                <Slider
                  value={bandpass}
                  onChange={(_, v) => setBandpass(v as number[])}
                  min={0}
                  max={Math.min(shapeX, shapeY) / 2}
                  orientation="vertical"
                  size="small"
                  sx={{
                    flex: 1,
                    '& .MuiSlider-thumb': {
                      '&:nth-of-type(3)': { color: '#f00' },  // Inner (HP)
                      '&:nth-of-type(4)': { color: '#09f' },  // Outer (LP)
                    },
                    '& .MuiSlider-track': { background: 'linear-gradient(to top, #f00, #09f)' }
                  }}
                />
                <Typography sx={{ color: "#f00", fontSize: 9 }}>{bpInner || "-"}</Typography>
              </Stack>
            </Stack>
          </Box>

          <Box>
            <Typography variant="caption" sx={{ ...typography.label, mb: 0.5, display: "block" }}>
              FFT {(bpInner > 0 || bpOuter > 0) && (
                <span>
                  {bpInner > 0 && <span style={{ color: "#f66" }}> (red=HP)</span>}
                  {bpOuter > 0 && <span style={{ color: "#09f" }}> (blue=LP)</span>}
                </span>
              )}
            </Typography>
            <Box sx={{ ...container.imageBox, width: 300, height: 300 }}>
              <canvas ref={fftCanvasRef} width={shapeY} height={shapeX} style={{ position: "absolute", width: "100%", height: "100%", imageRendering: "pixelated" }} />
              <canvas
                ref={fftOverlayRef} width={shapeY} height={shapeX}
                onMouseDown={handleFftMouseDown}
                onMouseMove={handleFftMouseMove}
                onMouseUp={handleFftMouseUp}
                onMouseLeave={handleFftMouseLeave}
                onWheel={createZoomHandler(setFftZoom, setFftPanX, setFftPanY, fftZoom, fftPanX, fftPanY, fftOverlayRef)}
                onDoubleClick={handleFftDoubleClick}
                style={{ position: "absolute", width: "100%", height: "100%", cursor: isDraggingFFT ? "grabbing" : "grab" }}
              />
            </Box>
          </Box>
        </Stack>
      </Stack>

      {/* Controls - Organized in 3 rows */}
      <Stack spacing={1} sx={{ mt: 2 }}>

        {/* Row 1: Presets + Detector */}
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ minHeight: 36 }}>
          {/* Detector Presets - only show if bf_radius is calibrated */}
          {bfRadius > 0 && (
            <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
              <Typography sx={{ ...typography.label }}>Presets:</Typography>
              <Button
                size="small"
                onClick={() => { setRoiMode("circle"); setRoiRadius(bfRadius); setRoiCenterX(centerX); setRoiCenterY(centerY); }}
                sx={{ minWidth: 30, fontSize: 10, px: 0.5, color: "#0f0" }}
              >BF</Button>
              <Button
                size="small"
                onClick={() => { setRoiMode("annular"); setRoiRadiusInner(bfRadius * 0.5); setRoiRadius(bfRadius); setRoiCenterX(centerX); setRoiCenterY(centerY); }}
                sx={{ minWidth: 30, fontSize: 10, px: 0.5, color: "#0af" }}
              >ABF</Button>
              <Button
                size="small"
                onClick={() => { setRoiMode("annular"); setRoiRadiusInner(bfRadius * 1.0); setRoiRadius(bfRadius * 2.0); setRoiCenterX(centerX); setRoiCenterY(centerY); }}
                sx={{ minWidth: 40, fontSize: 10, px: 0.5, color: "#fa0" }}
              >LAADF</Button>
              <Button
                size="small"
                onClick={() => { setRoiMode("annular"); setRoiRadiusInner(bfRadius * 2.0); setRoiRadius(bfRadius * 4.0); setRoiCenterX(centerX); setRoiCenterY(centerY); }}
                sx={{ minWidth: 45, fontSize: 10, px: 0.5, color: "#f50" }}
              >HAADF</Button>
            </Stack>
          )}

          {/* Virtual Detector Mode */}
          <Stack direction="row" spacing={1} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>Detector:</Typography>
            <Select
              value={roiMode || "point"}
              onChange={(e) => setRoiMode(e.target.value)}
              size="small"
              sx={{ ...controlPanel.select }}
              MenuProps={{
                anchorOrigin: { vertical: "top", horizontal: "left" },
                transformOrigin: { vertical: "bottom", horizontal: "left" },
                sx: { zIndex: 9999 }
              }}
            >
              <MenuItem value="point">Point</MenuItem>
              <MenuItem value="circle">Circle</MenuItem>
              <MenuItem value="square">Square</MenuItem>
              <MenuItem value="rect">Rect</MenuItem>
              <MenuItem value="annular">Annular</MenuItem>
            </Select>
            {(roiMode === "circle" || roiMode === "square") && (
              <>
                <Typography sx={{ ...typography.value }}>{roiMode === "circle" ? "r:" : "½:"}</Typography>
                <Slider
                  value={roiRadius || 10}
                  onChange={(_, v) => setRoiRadius(v as number)}
                  min={1}
                  max={Math.min(detX, detY) / 2}
                  size="small"
                  sx={{ width: 80 }}
                />
                <Typography sx={{ ...typography.value, minWidth: 30 }}>
                  {Math.round(roiRadius || 10)}px
                </Typography>
              </>
            )}
            {roiMode === "rect" && (
              <>
                <Typography sx={{ ...typography.value }}>W:</Typography>
                <Slider
                  value={roiWidth || 20}
                  onChange={(_, v) => setRoiWidth(v as number)}
                  min={2}
                  max={detY}
                  size="small"
                  sx={{ width: 60 }}
                />
                <Typography sx={{ ...typography.value }}>H:</Typography>
                <Slider
                  value={roiHeight || 10}
                  onChange={(_, v) => setRoiHeight(v as number)}
                  min={2}
                  max={detX}
                  size="small"
                  sx={{ width: 60 }}
                />
                <Typography sx={{ ...typography.value, minWidth: 50 }}>
                  {Math.round(roiWidth || 20)}×{Math.round(roiHeight || 10)}
                </Typography>
              </>
            )}
            {roiMode === "annular" && (
              <>
                <Typography sx={{ color: "#0cf", fontSize: 11 }}>in:</Typography>
                <Slider
                  value={roiRadiusInner || 5}
                  onChange={(_, v) => setRoiRadiusInner(v as number)}
                  min={0}
                  max={Math.min(detX, detY) / 2}
                  size="small"
                  sx={{ width: 60 }}
                />
                <Typography sx={{ color: "#0f0", fontSize: 11 }}>out:</Typography>
                <Slider
                  value={roiRadius || 10}
                  onChange={(_, v) => setRoiRadius(v as number)}
                  min={1}
                  max={Math.min(detX, detY) / 2}
                  size="small"
                  sx={{ width: 60 }}
                />
                <Typography sx={{ ...typography.value, minWidth: 50 }}>
                  {Math.round(roiRadiusInner || 5)}-{Math.round(roiRadius || 10)}px
                </Typography>
              </>
            )}
          </Stack>

          {/* Path Animation Controls - only show if path is defined */}
          {pathLength > 0 && (
            <Stack direction="row" spacing={1} alignItems="center" sx={{ ...controlPanel.group }}>
              <Typography sx={{ ...typography.label }}>Path:</Typography>
              <Typography
                component="span"
                onClick={() => { setPathPlaying(false); setPathIndex(0); }}
                sx={{ color: "#888", fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }}
                title="Stop"
              >⏹</Typography>
              <Typography
                component="span"
                onClick={() => setPathPlaying(!pathPlaying)}
                sx={{ color: pathPlaying ? "#0f0" : "#888", fontSize: 14, cursor: "pointer", "&:hover": { color: "#fff" }, px: 0.5 }}
                title={pathPlaying ? "Pause" : "Play"}
              >{pathPlaying ? "⏸" : "▶"}</Typography>
              <Typography sx={{ ...typography.value, minWidth: 60 }}>
                {pathIndex + 1}/{pathLength}
              </Typography>
              <Slider
                value={pathIndex}
                onChange={(_, v) => { setPathPlaying(false); setPathIndex(v as number); }}
                min={0}
                max={Math.max(0, pathLength - 1)}
                size="small"
                sx={{ width: 100 }}
              />
            </Stack>
          )}
        </Stack>

        {/* Row 2: Colormap + Log + Contrast */}
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ minHeight: 36 }}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>Colormap:</Typography>
            <Select
              value={colormap}
              onChange={(e) => setColormap(String(e.target.value))}
              size="small"
              sx={{ ...controlPanel.select }}
              MenuProps={{
                anchorOrigin: { vertical: "top", horizontal: "left" },
                transformOrigin: { vertical: "bottom", horizontal: "left" },
                sx: { zIndex: 9999 }
              }}
            >
              <MenuItem value="inferno">Inferno</MenuItem>
              <MenuItem value="viridis">Viridis</MenuItem>
              <MenuItem value="plasma">Plasma</MenuItem>
              <MenuItem value="magma">Magma</MenuItem>
              <MenuItem value="hot">Hot</MenuItem>
              <MenuItem value="gray">Gray</MenuItem>
            </Select>
          </Stack>

          <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>Log:</Typography>
            <Switch
              checked={logScale ?? true}
              onChange={(e) => setLogScale(e.target.checked)}
              size="small"
              sx={{ '& .MuiSwitch-thumb': { width: 14, height: 14 }, '& .MuiSwitch-switchBase': { padding: '4px' } }}
            />
          </Stack>

          <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>Contrast:</Typography>
            <Switch
              checked={autoRange ?? false}
              onChange={(e) => setAutoRange(e.target.checked)}
              size="small"
              sx={{ '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } }}
            />
            {autoRange && (
              <>
                <Slider
                  value={[percentileLow || 1, percentileHigh || 99]}
                  onChange={(_, v) => {
                    const [low, high] = v as number[];
                    setPercentileLow(low);
                    setPercentileHigh(high);
                  }}
                  min={0}
                  max={100}
                  size="small"
                  sx={{ width: 80 }}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(v) => `${v}%`}
                />
                <Typography sx={{ ...typography.value, minWidth: 45 }}>
                  {Math.round(percentileLow || 1)}-{Math.round(percentileHigh || 99)}%
                </Typography>
              </>
            )}
          </Stack>
        </Stack>
      </Stack>
    </Box>
  );
}

export const render = createRender(Show4DSTEM);
