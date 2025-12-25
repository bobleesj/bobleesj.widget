/**
 * Show2D - Static 2D image viewer with gallery support.
 * 
 * Features:
 * - Single image or gallery mode with configurable columns
 * - Scroll to zoom, double-click to reset
 * - WebGPU-accelerated FFT with default 3x zoom
 * - Equal-sized FFT and histogram panels
 * - Click to select image in gallery mode
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";

import {
  colors,
  COLORMAP_NAMES,
  COLORMAPS,
  drawScaleBar,
  extractBytes,
  formatNumber,
} from "./core";
import { upwardMenuProps } from "./components";
import { getWebGPUFFT, WebGPUFFT } from "./webgpu-fft";
import { fft2d, fftshift, MIN_ZOOM, MAX_ZOOM } from "./shared";
import "./show2d.css";

// ============================================================================
// Types
// ============================================================================
type ZoomState = { zoom: number; panX: number; panY: number };

// ============================================================================
// Constants
// ============================================================================
const SINGLE_IMAGE_TARGET = 200;
const GALLERY_IMAGE_TARGET = 100;
const PANEL_SIZE = 150;
const DEFAULT_FFT_ZOOM = 3;
const DEFAULT_ZOOM_STATE: ZoomState = { zoom: 1, panX: 0, panY: 0 };

// ============================================================================
// Main Component
// ============================================================================
function Show2D() {
  // Model state
  const [nImages] = useModelState<number>("n_images");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");
  const [ncols] = useModelState<number>("ncols");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");

  // Customization
  const [scaleBarLengthPx] = useModelState<number>("scale_bar_length_px");
  const [scaleBarThicknessPx] = useModelState<number>("scale_bar_thickness_px");
  const [scaleBarFontSizePx] = useModelState<number>("scale_bar_font_size_px");
  const [panelSizePx] = useModelState<number>("panel_size_px");
  const [imageWidthPx] = useModelState<number>("image_width_px");

  // Scale bar
  const [pixelSizeAngstrom] = useModelState<number>("pixel_size_angstrom");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");

  // UI visibility
  const [showControls] = useModelState<boolean>("show_controls");
  const [showStats] = useModelState<boolean>("show_stats");
  const [statsMean] = useModelState<number[]>("stats_mean");
  const [statsMin] = useModelState<number[]>("stats_min");
  const [statsMax] = useModelState<number[]>("stats_max");
  const [statsStd] = useModelState<number[]>("stats_std");

  // FFT & Histogram
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [showHistogram, setShowHistogram] = useModelState<boolean>("show_histogram");
  const [histogramCounts] = useModelState<number[]>("histogram_counts");

  // Selection
  const [selectedIdx, setSelectedIdx] = useModelState<number>("selected_idx");

  // Canvas refs
  const canvasRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const overlayRefs = React.useRef<(HTMLCanvasElement | null)[]>([]);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const histCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const [canvasReady, setCanvasReady] = React.useState(0);  // Trigger re-render when refs attached

  // Zoom/Pan state - per-image when not linked, shared when linked
  const [zoomStates, setZoomStates] = React.useState<Map<number, ZoomState>>(new Map());
  const [linkedZoomState, setLinkedZoomState] = React.useState<ZoomState>(DEFAULT_ZOOM_STATE);
  const [linkedZoom, setLinkedZoom] = React.useState(false);  // Link zoom across gallery images
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Helper to get zoom state for an image
  const getZoomState = React.useCallback((idx: number): ZoomState => {
    if (linkedZoom) return linkedZoomState;
    return zoomStates.get(idx) || DEFAULT_ZOOM_STATE;
  }, [linkedZoom, linkedZoomState, zoomStates]);

  // Helper to set zoom state for an image
  const setZoomState = React.useCallback((idx: number, state: ZoomState) => {
    if (linkedZoom) {
      setLinkedZoomState(state);
    } else {
      setZoomStates(prev => new Map(prev).set(idx, state));
    }
  }, [linkedZoom]);

  // FFT zoom/pan state
  const [fftZoom, setFftZoom] = React.useState(DEFAULT_FFT_ZOOM);
  const [fftPanX, setFftPanX] = React.useState(0);
  const [fftPanY, setFftPanY] = React.useState(0);
  const [isDraggingFftPan, setIsDraggingFftPan] = React.useState(false);
  const [fftPanStart, setFftPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Resizable state
  const [canvasSize, setCanvasSize] = React.useState(SINGLE_IMAGE_TARGET);
  const [panelSize, setPanelSize] = React.useState(PANEL_SIZE);

  // Sync initial sizes from traits
  React.useEffect(() => {
    if (imageWidthPx > 0) setCanvasSize(imageWidthPx);
  }, [imageWidthPx]);

  React.useEffect(() => {
    if (panelSizePx > 0) setPanelSize(panelSizePx);
  }, [panelSizePx]);

  const [isResizingCanvas, setIsResizingCanvas] = React.useState(false);
  const [isResizingPanel, setIsResizingPanel] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number, y: number, size: number } | null>(null);

  // WebGPU FFT
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const rawDataRef = React.useRef<Float32Array[] | null>(null);

  // Layout calculations
  const isGallery = nImages > 1;
  const displayScale = canvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);
  const bytesPerImage = width * height;

  // Extract all image bytes
  const allBytes = React.useMemo(() => extractBytes(frameBytes), [frameBytes]);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  const [dataReady, setDataReady] = React.useState(false);

  // Parse frame data and store raw floats for FFT
  React.useEffect(() => {
    if (!allBytes || allBytes.length === 0) return;
    const dataArrays: Float32Array[] = [];
    for (let i = 0; i < nImages; i++) {
      const start = i * bytesPerImage;
      const imageData = allBytes.subarray(start, start + bytesPerImage);
      // Native conversion is much faster than manual loop
      dataArrays.push(new Float32Array(imageData));
    }
    rawDataRef.current = dataArrays;
    setDataReady(true); // Trigger re-render now that data is ready
  }, [allBytes, nImages, bytesPerImage]);

  // Prevent page scroll when scrolling on the active image canvas or FFT
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();

    // In gallery mode:
    // - If linkedZoom is ON, prevent scroll on ALL images (since all are zoomable)
    // - If linkedZoom is OFF, prevent scroll only on the selected image
    // In single mode: prevent scroll on the single image
    const targets: (HTMLCanvasElement | null)[] = [];

    if (isGallery) {
      if (linkedZoom) {
        // Add all available canvases
        targets.push(...canvasRefs.current);
      } else {
        // Add only selected
        targets.push(canvasRefs.current[selectedIdx]);
      }
    } else {
      targets.push(canvasRefs.current[0]);
    }

    // Always add FFT
    targets.push(fftCanvasRef.current);

    targets.forEach(t => t?.addEventListener("wheel", preventDefault, { passive: false }));

    return () => {
      targets.forEach(t => t?.removeEventListener("wheel", preventDefault));
    };
  }, [nImages, canvasReady, selectedIdx, isGallery, linkedZoom, dataReady]);

  // -------------------------------------------------------------------------
  // Render Images
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!allBytes || allBytes.length === 0) return;

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    for (let i = 0; i < nImages; i++) {
      const canvas = canvasRefs.current[i];
      if (!canvas) continue;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;

      const start = i * bytesPerImage;
      const imageData = allBytes.slice(start, start + bytesPerImage);

      // Create offscreen canvas at native resolution
      const offscreen = document.createElement("canvas");
      offscreen.width = width;
      offscreen.height = height;
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) continue;

      const imgData = offCtx.createImageData(width, height);
      const rgba = imgData.data;

      for (let j = 0; j < imageData.length; j++) {
        const v = imageData[j];
        const k = j * 4;
        const lutIdx = v * 3;
        rgba[k] = lut[lutIdx];
        rgba[k + 1] = lut[lutIdx + 1];
        rgba[k + 2] = lut[lutIdx + 2];
        rgba[k + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);

      // Draw to display canvas with proper scaling
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Get per-image zoom state (inline to avoid callback dependency issues)
      const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
      const { zoom, panX, panY } = zs;

      // Apply zoom/pan: zoom from center, then apply pan offset
      if (zoom !== 1 || panX !== 0 || panY !== 0) {
        ctx.save();
        // Translate to center, scale, translate back, then apply pan
        const cx = canvasW / 2;
        const cy = canvasH / 2;
        ctx.translate(cx + panX, cy + panY);
        ctx.scale(zoom, zoom);
        ctx.translate(-cx, -cy);
        ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
        ctx.restore();
      } else {
        ctx.drawImage(offscreen, 0, 0, width, height, 0, 0, canvasW, canvasH);
      }
    }
  }, [allBytes, nImages, width, height, cmap, bytesPerImage, displayScale, isGallery, canvasW, canvasH, canvasReady, linkedZoom, linkedZoomState, zoomStates]);

  // -------------------------------------------------------------------------
  // Render Overlays (scale bar, selection, zoom indicator)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    for (let i = 0; i < nImages; i++) {
      const overlay = overlayRefs.current[i];
      if (!overlay) continue;
      const ctx = overlay.getContext("2d");
      if (!ctx) continue;

      ctx.clearRect(0, 0, canvasW, canvasH);

      // Scale bar on ALL images (for local comparison in gallery)
      if (pixelSizeAngstrom > 0 && scaleBarVisible) {
        // Each image's scale bar reflects its own zoom state (inline)
        const zs = linkedZoom ? linkedZoomState : (zoomStates.get(i) || DEFAULT_ZOOM_STATE);
        drawScaleBar(ctx, canvasW, canvasH, width, pixelSizeAngstrom, displayScale * zs.zoom, scaleBarLengthPx, scaleBarThicknessPx, scaleBarFontSizePx);
      }

      // Highlight selected in gallery
      if (isGallery && i === selectedIdx) {
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 3;
        ctx.strokeRect(1.5, 1.5, canvasW - 3, canvasH - 3);
      }
    }
  }, [nImages, pixelSizeAngstrom, scaleBarVisible, selectedIdx, isGallery, canvasW, canvasH, width, displayScale, linkedZoom, linkedZoomState, zoomStates, dataReady, scaleBarLengthPx, scaleBarThicknessPx, scaleBarFontSizePx]);

  // -------------------------------------------------------------------------
  // Render FFT with WebGPU
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || !fftCanvasRef.current || !rawDataRef.current) return;
    if (!rawDataRef.current[selectedIdx]) return;

    const canvas = fftCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const renderFFT = async (fftMag: Float32Array) => {
      // Log scale and normalize
      let min = Infinity, max = -Infinity;
      const logData = new Float32Array(fftMag.length);
      for (let i = 0; i < fftMag.length; i++) {
        logData[i] = Math.log(1 + fftMag[i]);
        if (logData[i] < min) min = logData[i];
        if (logData[i] > max) max = logData[i];
      }

      const lut = COLORMAPS.inferno;
      const offscreen = document.createElement("canvas");
      offscreen.width = width;
      offscreen.height = height;
      const offCtx = offscreen.getContext("2d");
      if (!offCtx) return;

      const imgData = offCtx.createImageData(width, height);
      for (let i = 0; i < logData.length; i++) {
        const v = Math.floor(((logData[i] - min) / (max - min || 1)) * 255);
        const j = i * 4;
        imgData.data[j] = lut[v * 3];
        imgData.data[j + 1] = lut[v * 3 + 1];
        imgData.data[j + 2] = lut[v * 3 + 2];
        imgData.data[j + 3] = 255;
      }
      offCtx.putImageData(imgData, 0, 0);

      // Draw with FFT zoom/pan - center the zoomed view
      const scale = panelSize / Math.max(width, height);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, panelSize, panelSize);
      ctx.save();

      const centerOffsetX = (panelSize - width * scale * fftZoom) / 2 + fftPanX;
      const centerOffsetY = (panelSize - height * scale * fftZoom) / 2 + fftPanY;

      ctx.translate(centerOffsetX, centerOffsetY);
      ctx.scale(fftZoom, fftZoom);
      ctx.drawImage(offscreen, 0, 0, width * scale, height * scale);
      ctx.restore();
    };

    const computeFFT = async () => {
      const data = rawDataRef.current![selectedIdx];
      const real = data.slice();
      const imag = new Float32Array(data.length);

      if (gpuFFTRef.current && gpuReady) {
        // WebGPU FFT
        const { real: fReal, imag: fImag } = await gpuFFTRef.current.fft2D(real, imag, width, height, false);
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);

        const mag = new Float32Array(width * height);
        for (let i = 0; i < mag.length; i++) {
          mag[i] = Math.sqrt(fReal[i] ** 2 + fImag[i] ** 2);
        }
        await renderFFT(mag);
      } else {
        // CPU fallback
        fft2d(real, imag, width, height, false);
        fftshift(real, width, height);
        fftshift(imag, width, height);

        const mag = new Float32Array(width * height);
        for (let i = 0; i < mag.length; i++) {
          mag[i] = Math.sqrt(real[i] ** 2 + imag[i] ** 2);
        }
        await renderFFT(mag);
      }
    };

    computeFFT();
  }, [showFft, selectedIdx, width, height, gpuReady, allBytes, panelSize, fftZoom, fftPanX, fftPanY, dataReady]);

  // -------------------------------------------------------------------------
  // Render Histogram
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showHistogram || !histCanvasRef.current) return;

    const canvas = histCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = panelSize;
    const h = panelSize;
    
    // Always clear and fill background
    ctx.fillStyle = colors.bgPanel;
    ctx.fillRect(0, 0, w, h);

    // Only draw bars if we have data
    if (!histogramCounts || histogramCounts.length === 0) return;

    const maxCount = Math.max(...histogramCounts);
    if (maxCount === 0) return;

    // Add padding for centering
    const padding = 8;
    const drawWidth = w - 2 * padding;
    const drawHeight = h - padding - 5;  // 5px bottom margin for axis
    const barWidth = drawWidth / histogramCounts.length;
    
    ctx.fillStyle = colors.accent;
    for (let i = 0; i < histogramCounts.length; i++) {
      const barHeight = (histogramCounts[i] / maxCount) * drawHeight;
      ctx.fillRect(padding + i * barWidth, h - padding - barHeight, barWidth - 1, barHeight);
    }
  }, [showHistogram, histogramCounts, panelSize, selectedIdx, dataReady]);

  // -------------------------------------------------------------------------
  // Mouse Handlers for Zoom/Pan
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent, idx: number) => {
    // In gallery mode, only allow zoom on the selected image
    if (isGallery && idx !== selectedIdx) return;
    
    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    // Mouse position relative to canvas center
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const mouseX = (e.clientX - rect.left) * (canvas.width / rect.width) - cx;
    const mouseY = (e.clientY - rect.top) * (canvas.height / rect.height) - cy;

    const zs = getZoomState(idx);
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zs.zoom * zoomFactor));
    
    // Adjust pan to zoom toward mouse position (relative to center)
    const zoomRatio = newZoom / zs.zoom;
    const newPanX = zs.panX - mouseX * (zoomRatio - 1);
    const newPanY = zs.panY - mouseY * (zoomRatio - 1);

    setZoomState(idx, { zoom: newZoom, panX: newPanX, panY: newPanY });
  };

  const handleDoubleClick = (idx: number) => {
    setZoomState(idx, DEFAULT_ZOOM_STATE);
  };

  // Reset all zoom states to default
  const handleResetAll = () => {
    setZoomStates(new Map());
    setLinkedZoomState(DEFAULT_ZOOM_STATE);
  };

  // FFT zoom/pan handlers
  const handleFftWheel = (e: React.WheelEvent) => {
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    setFftZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, fftZoom * zoomFactor)));
  };

  const handleFftDoubleClick = () => {
    setFftZoom(DEFAULT_FFT_ZOOM);
    setFftPanX(0);
    setFftPanY(0);
  };

  const handleFftMouseDown = (e: React.MouseEvent) => {
    setIsDraggingFftPan(true);
    setFftPanStart({ x: e.clientX, y: e.clientY, pX: fftPanX, pY: fftPanY });
  };

  const handleFftMouseMove = (e: React.MouseEvent) => {
    if (!isDraggingFftPan || !fftPanStart) return;
    const dx = e.clientX - fftPanStart.x;
    const dy = e.clientY - fftPanStart.y;
    setFftPanX(fftPanStart.pX + dx);
    setFftPanY(fftPanStart.pY + dy);
  };

  const handleFftMouseUp = () => {
    setIsDraggingFftPan(false);
    setFftPanStart(null);
  };

  // Track which image is being panned
  const [panningIdx, setPanningIdx] = React.useState<number | null>(null);

  const handleMouseDown = (e: React.MouseEvent, idx: number) => {
    const zs = getZoomState(idx);
    if (isGallery && idx !== selectedIdx) {
      setSelectedIdx(idx);
      return; // Only select, don't start panning
    }
    setIsDraggingPan(true);
    setPanningIdx(idx);
    setPanStart({ x: e.clientX, y: e.clientY, pX: zs.panX, pY: zs.panY });
  };

  const handleMouseMove = (e: React.MouseEvent, idx: number) => {
    if (!isDraggingPan || !panStart || panningIdx === null) return;
    if (idx !== panningIdx) return;

    const canvas = canvasRefs.current[idx];
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const dx = (e.clientX - panStart.x) * scaleX;
    const dy = (e.clientY - panStart.y) * scaleY;

    const zs = getZoomState(idx);
    setZoomState(idx, { ...zs, panX: panStart.pX + dx, panY: panStart.pY + dy });
  };

  const handleMouseUp = () => {
    setIsDraggingPan(false);
    setPanStart(null);
    setPanningIdx(null);
  };

  // -------------------------------------------------------------------------
  // Resize Handlers
  // -------------------------------------------------------------------------
  const handleCanvasResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingCanvas(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: canvasSize });
  };

  const handlePanelResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingPanel(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: panelSize });
  };

  React.useEffect(() => {
    if (!isResizingCanvas && !isResizingPanel) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);

      if (isResizingCanvas) {
        const newSize = Math.max(100, Math.min(600, resizeStart.size + delta));
        setCanvasSize(newSize);
      } else if (isResizingPanel) {
        const newSize = Math.max(80, Math.min(250, resizeStart.size + delta));
        setPanelSize(newSize);
      }
    };

    const handleMouseUp = () => {
      setIsResizingCanvas(false);
      setIsResizingPanel(false);
      setResizeStart(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingCanvas, isResizingPanel, resizeStart]);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <Box className="show2d-root" sx={{ p: 1.5, bgcolor: colors.bg, borderRadius: 1, width: "fit-content" }}>
      {/* Title */}
      {title && (
        <Typography sx={{ color: colors.accent, fontWeight: "bold", mb: 1, fontSize: 13 }}>
          {title}
        </Typography>
      )}

      {/* Main layout */}
      <Stack direction="row" spacing={1.5}>
        {/* Images */}
        <Box>
          {isGallery ? (
            // Gallery mode
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(" + ncols + ", auto)",
                gap: 1,
              }}
            >
              {Array.from({ length: nImages }).map((_, i) => (
                <Box
                  key={i}
                  sx={{ cursor: i === selectedIdx ? "grab" : "pointer" }}
                >
                  <Box
                    sx={{
                      position: "relative",
                      bgcolor: "#000",
                      border: "1px solid " + (i === selectedIdx ? colors.accent : colors.border),
                      borderRadius: 0.5,
                    }}
                    onMouseDown={(e) => handleMouseDown(e, i)}
                    onMouseMove={(e) => handleMouseMove(e, i)}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    onWheel={(e) => handleWheel(e, i)}
                    onDoubleClick={() => handleDoubleClick(i)}
                  >
                    <canvas
                      ref={(el) => {
                        if (el && canvasRefs.current[i] !== el) {
                          canvasRefs.current[i] = el;
                          setCanvasReady(c => c + 1);
                        }
                      }}
                      width={canvasW}
                      height={canvasH}
                      style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
                    />
                    <canvas
                      ref={(el) => { overlayRefs.current[i] = el; }}
                      width={canvasW}
                      height={canvasH}
                      style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
                    />
                    {/* Resize handle */}
                    <Box
                      onMouseDown={handleCanvasResizeStart}
                      sx={{
                        position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                        cursor: "nwse-resize", opacity: 0.4,
                        background: "linear-gradient(135deg, transparent 50%, " + colors.textMuted + " 50%)",
                        "&:hover": { opacity: 1 }
                      }}
                    />
                  </Box>
                  <Typography sx={{ fontSize: 10, color: colors.textSecondary, textAlign: "center", mt: 0.25 }}>
                    {labels?.[i] || "Image " + (i + 1)}
                  </Typography>
                </Box>
              ))}
            </Box>
          ) : (
            // Single image mode with zoom/pan
            <Box>
              <Box
                sx={{
                  position: "relative",
                  bgcolor: "#000",
                  border: "1px solid " + colors.border,
                  borderRadius: 0.5,
                  cursor: "grab",
                }}
                onMouseDown={(e) => handleMouseDown(e, 0)}
                onMouseMove={(e) => handleMouseMove(e, 0)}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onWheel={(e) => handleWheel(e, 0)}
                onDoubleClick={() => handleDoubleClick(0)}
              >
                <canvas
                  ref={(el) => {
                    if (el && canvasRefs.current[0] !== el) {
                      canvasRefs.current[0] = el;
                      setCanvasReady(c => c + 1);
                    }
                  }}
                  width={canvasW}
                  height={canvasH}
                  style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
                />
                <canvas
                  ref={(el) => { overlayRefs.current[0] = el; }}
                  width={canvasW}
                  height={canvasH}
                  style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
                />
                {/* Resize handle */}
                <Box
                  onMouseDown={handleCanvasResizeStart}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                    cursor: "nwse-resize", opacity: 0.4,
                    background: "linear-gradient(135deg, transparent 50%, " + colors.textMuted + " 50%)",
                    "&:hover": { opacity: 1 }
                  }}
                />
              </Box>
              {labels?.[0] && (
                <Typography sx={{ fontSize: 11, color: colors.textSecondary, textAlign: "center", mt: 0.5 }}>
                  {labels[0]}
                </Typography>
              )}
            </Box>
          )}
        </Box>

        {/* Side panels - FFT and Histogram (equal sized) */}
        {(showFft || showHistogram) && (
          <Stack spacing={1}>
            {showFft && (
              <Box sx={{ position: "relative", bgcolor: colors.bgPanel, border: "1px solid " + colors.border, borderRadius: 0.5, p: 0.75 }}>
                <Typography sx={{ fontSize: 10, color: colors.textMuted, textTransform: "uppercase", mb: 0.5 }}>
                  FFT {isGallery && "(" + (labels?.[selectedIdx] || "#" + (selectedIdx + 1)) + ")"} {fftZoom !== DEFAULT_FFT_ZOOM && "(" + fftZoom.toFixed(1) + "×)"}
                </Typography>
                <canvas
                  ref={fftCanvasRef}
                  width={panelSize}
                  height={panelSize}
                  style={{ cursor: "grab", imageRendering: "pixelated", display: "block" }}
                  onWheel={handleFftWheel}
                  onDoubleClick={handleFftDoubleClick}
                  onMouseDown={handleFftMouseDown}
                  onMouseMove={handleFftMouseMove}
                  onMouseUp={handleFftMouseUp}
                  onMouseLeave={handleFftMouseUp}
                />
                {/* Panel resize handle */}
                <Box
                  onMouseDown={handlePanelResizeStart}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 10, height: 10,
                    cursor: "nwse-resize", opacity: 0.4,
                    background: "linear-gradient(135deg, transparent 50%, " + colors.textMuted + " 50%)",
                    "&:hover": { opacity: 1 }
                  }}
                />
              </Box>
            )}
            {showHistogram && (
              <Box sx={{ bgcolor: colors.bgPanel, border: "1px solid " + colors.border, borderRadius: 0.5, p: 0.75, position: "relative" }}>
                <Typography sx={{ fontSize: 10, color: colors.textMuted, textTransform: "uppercase", mb: 0.5 }}>
                  Histogram {isGallery && "(" + (labels?.[selectedIdx] || "#" + (selectedIdx + 1)) + ")"}
                </Typography>
                <canvas ref={histCanvasRef} width={panelSize} height={panelSize} style={{ display: "block" }} />
                {/* Panel resize handle (resizes both FFT and Histogram together) */}
                <Box
                  onMouseDown={handlePanelResizeStart}
                  sx={{
                    position: "absolute", bottom: 2, right: 2, width: 10, height: 10,
                    cursor: "nwse-resize", opacity: 0.4,
                    background: "linear-gradient(135deg, transparent 50%, " + colors.textMuted + " 50%)",
                    "&:hover": { opacity: 1 }
                  }}
                />
              </Box>
            )}
          </Stack>
        )}
      </Stack>

      {/* Options row */}
      {showControls && (
        <Stack direction="row" spacing={2} sx={{ mt: 1, flexWrap: "wrap" }} alignItems="center">
          {[
            { label: "Log", checked: logScale, onChange: () => setLogScale(!logScale) },
            { label: "Auto", checked: autoContrast, onChange: () => setAutoContrast(!autoContrast) },
            { label: "FFT", checked: showFft, onChange: () => setShowFft(!showFft) },
            { label: "Hist", checked: showHistogram, onChange: () => setShowHistogram(!showHistogram) },
          ].map(({ label, checked, onChange }) => (
            <Stack key={label} direction="row" alignItems="center" spacing={0.5}>
              <Switch size="small" checked={checked} onChange={onChange} sx={{ "& .MuiSwitch-thumb": { width: 12, height: 12 }, "& .MuiSwitch-track": { height: 14 } }} />
              <Typography sx={{ fontSize: 11, color: colors.textMuted }}>{label}</Typography>
            </Stack>
          ))}
          <Select
            size="small"
            value={cmap}
            onChange={(e) => setCmap(e.target.value)}
            MenuProps={upwardMenuProps}
            sx={{ minWidth: 80, bgcolor: colors.bgInput, color: colors.textPrimary, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } }}
          >
            {COLORMAP_NAMES.map((name) => (
              <MenuItem key={name} value={name} sx={{ fontSize: 11 }}>
                {name}
              </MenuItem>
            ))}
          </Select>
          {isGallery && (
            <Stack direction="row" alignItems="center" spacing={0.5}>
              <Switch size="small" checked={linkedZoom} onChange={() => setLinkedZoom(!linkedZoom)} sx={{ '& .MuiSwitch-thumb': { width: 12, height: 12 }, '& .MuiSwitch-switchBase': { padding: '4px' } }} />
              <Typography sx={{ fontSize: 11, color: colors.textMuted }}>Link Zoom</Typography>
            </Stack>
          )}
          <Box
            onClick={handleResetAll}
            sx={{ px: 1, py: 0.25, bgcolor: colors.bgPanel, border: "1px solid " + colors.border, borderRadius: 0.5, cursor: "pointer", "&:hover": { bgcolor: colors.bgInput } }}
          >
            <Typography sx={{ fontSize: 11, color: colors.textMuted }}>Reset</Typography>
          </Box>
          {getZoomState(selectedIdx).zoom !== 1 && (
            <Typography sx={{ fontSize: 11, color: colors.accent, fontWeight: "bold" }}>
              Zoom: {getZoomState(selectedIdx).zoom.toFixed(1)}×
            </Typography>
          )}
        </Stack>
      )}

      {/* Stats */}
      {showStats && (
        <Box sx={{ mt: 1, bgcolor: colors.bgPanel, px: 1.5, py: 0.75, borderRadius: 0.5, border: "1px solid " + colors.border, width: "fit-content", maxWidth: "100%" }}>
          {isGallery ? (
            // Gallery: show stats for selected image
            <Stack direction="row" spacing={2} flexWrap="wrap">
              <Typography sx={{ fontSize: 10, color: colors.textMuted }}>
                {labels?.[selectedIdx] || "Image " + (selectedIdx + 1)}:
              </Typography>
              {[
                { label: "Mean", value: statsMean?.[selectedIdx] },
                { label: "Min", value: statsMin?.[selectedIdx] },
                { label: "Max", value: statsMax?.[selectedIdx] },
                { label: "Std", value: statsStd?.[selectedIdx] },
              ].map(({ label, value }) => (
                <Stack key={label} direction="row" spacing={0.5} alignItems="baseline">
                  <Typography sx={{ fontSize: 10, color: colors.textDim }}>{label}</Typography>
                  <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: colors.accent }}>
                    {value !== undefined ? formatNumber(value) : "-"}
                  </Typography>
                </Stack>
              ))}
            </Stack>
          ) : (
            // Single: show stats
            <Stack direction="row" spacing={2} flexWrap="wrap">
              {[
                { label: "Mean", value: statsMean?.[0] },
                { label: "Min", value: statsMin?.[0] },
                { label: "Max", value: statsMax?.[0] },
                { label: "Std", value: statsStd?.[0] },
              ].map(({ label, value }) => (
                <Stack key={label} direction="row" spacing={0.5} alignItems="baseline">
                  <Typography sx={{ fontSize: 10, color: colors.textDim }}>{label}</Typography>
                  <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: colors.accent }}>
                    {value !== undefined ? formatNumber(value) : "-"}
                  </Typography>
                </Stack>
              ))}
            </Stack>
          )}
        </Box>
      )}
    </Box>
  );
}

export const render = createRender(Show2D);
