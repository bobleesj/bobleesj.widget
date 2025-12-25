/**
 * Show3D - Interactive 3D stack viewer with playback controls.
 * Uses shared core utilities from js/core/.
 * 
 * Features:
 * - Scroll to zoom, double-click to reset
 * - Adjustable ROI size via slider
 * - FPS slider control
 * - WebGPU-accelerated FFT with default 3x zoom
 * - Equal-sized FFT and histogram panels
 */

import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Slider from "@mui/material/Slider";
import IconButton from "@mui/material/IconButton";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Switch from "@mui/material/Switch";

import { upwardMenuProps } from "./components";

import {
  colors,
  COLORMAP_NAMES,
  COLORMAPS,
  applyColormapToImage,
  calculateDisplayScale,
  drawScaleBar,
  drawROI,
  extractBytes,
  formatNumber,
} from "./core";
import { getWebGPUFFT, WebGPUFFT } from "./webgpu-fft";
import { fft2d, fftshift, MIN_ZOOM, MAX_ZOOM } from "./shared";
import "./show3d.css";

// ============================================================================
// Constants
// ============================================================================
const CANVAS_TARGET_SIZE = 400;
const PANEL_SIZE = 150; // Equal size for both FFT and histogram
const DEFAULT_FFT_ZOOM = 3; // Default FFT zoom to see center details
const DEFAULT_FPS = 1; // 1 fps = 1000ms per frame

// ROI shapes
const ROI_SHAPES = ["circle", "square", "rectangle"] as const;
type RoiShape = typeof ROI_SHAPES[number];

// ============================================================================
// Main Component
// ============================================================================
function Show3D() {
  // Model state (synced with Python)
  const [sliceIdx, setSliceIdx] = useModelState<number>("slice_idx");
  const [nSlices] = useModelState<number>("n_slices");
  const [width] = useModelState<number>("width");
  const [height] = useModelState<number>("height");
  const [frameBytes] = useModelState<DataView>("frame_bytes");
  const [labels] = useModelState<string[]>("labels");
  const [title] = useModelState<string>("title");
  const [cmap, setCmap] = useModelState<string>("cmap");

  // Playback
  const [playing, setPlaying] = useModelState<boolean>("playing");
  const [reverse, setReverse] = useModelState<boolean>("reverse");
  const [fps, setFps] = useModelState<number>("fps");
  const [loop, setLoop] = useModelState<boolean>("loop");
  const [loopStart, setLoopStart] = useModelState<number>("loop_start");
  const [loopEnd, setLoopEnd] = useModelState<number>("loop_end");

  // Stats
  const [showStats] = useModelState<boolean>("show_stats");
  const [statsMean] = useModelState<number>("stats_mean");
  const [statsMin] = useModelState<number>("stats_min");
  const [statsMax] = useModelState<number>("stats_max");
  const [statsStd] = useModelState<number>("stats_std");

  // Display options
  const [logScale, setLogScale] = useModelState<boolean>("log_scale");
  const [autoContrast, setAutoContrast] = useModelState<boolean>("auto_contrast");

  // Scale bar
  const [pixelSize] = useModelState<number>("pixel_size");
  const [scaleBarVisible] = useModelState<boolean>("scale_bar_visible");
  const [scaleBarLengthPx] = useModelState<number>("scale_bar_length_px");
  const [scaleBarThicknessPx] = useModelState<number>("scale_bar_thickness_px");
  const [scaleBarFontSizePx] = useModelState<number>("scale_bar_font_size_px");

  // Customization (from Python)
  const [panelSizePxTrait] = useModelState<number>("panel_size_px");
  const [imageWidthPxTrait] = useModelState<number>("image_width_px");

  // Timestamps
  const [timestamps] = useModelState<number[]>("timestamps");
  const [timestampUnit] = useModelState<string>("timestamp_unit");
  const [currentTimestamp] = useModelState<number>("current_timestamp");

  // ROI
  const [roiActive, setRoiActive] = useModelState<boolean>("roi_active");
  const [roiShape, setRoiShape] = useModelState<RoiShape>("roi_shape");
  const [roiX, setRoiX] = useModelState<number>("roi_x");
  const [roiY, setRoiY] = useModelState<number>("roi_y");
  const [roiRadius, setRoiRadius] = useModelState<number>("roi_radius");
  const [roiWidth, setRoiWidth] = useModelState<number>("roi_width");
  const [roiHeight, setRoiHeight] = useModelState<number>("roi_height");
  const [roiMean] = useModelState<number>("roi_mean");

  // FFT (shows both FFT and histogram together)
  const [showFft, setShowFft] = useModelState<boolean>("show_fft");
  const [fftBytes] = useModelState<DataView>("fft_bytes");

  // Histogram data (displayed with FFT)
  const [histogramBins] = useModelState<number[]>("histogram_bins");
  const [histogramCounts] = useModelState<number[]>("histogram_counts");

  // Canvas refs
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const overlayRef = React.useRef<HTMLCanvasElement>(null);
  const fftCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const histCanvasRef = React.useRef<HTMLCanvasElement>(null);

  // Local state
  const [isDraggingROI, setIsDraggingROI] = React.useState(false);
  const playIntervalRef = React.useRef<number | null>(null);

  // Zoom/Pan state
  const [zoom, setZoom] = React.useState(1);
  const [panX, setPanX] = React.useState(0);
  const [panY, setPanY] = React.useState(0);
  const [isDraggingPan, setIsDraggingPan] = React.useState(false);
  const [panStart, setPanStart] = React.useState<{ x: number, y: number, pX: number, pY: number } | null>(null);

  // Resizable panel state (all JS-side, no Python calls)
  const [mainCanvasSize, setMainCanvasSize] = React.useState(CANVAS_TARGET_SIZE);
  const [panelSize, setPanelSize] = React.useState(PANEL_SIZE);
  const [isResizingMain, setIsResizingMain] = React.useState(false);
  const [isResizingPanel, setIsResizingPanel] = React.useState(false);
  const [resizeStart, setResizeStart] = React.useState<{ x: number, y: number, size: number } | null>(null);

  // Sync sizes from Python traits
  React.useEffect(() => {
    if (imageWidthPxTrait > 0) setMainCanvasSize(imageWidthPxTrait);
  }, [imageWidthPxTrait]);

  React.useEffect(() => {
    if (panelSizePxTrait > 0) setPanelSize(panelSizePxTrait);
  }, [panelSizePxTrait]);

  // WebGPU FFT
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);
  const rawFrameDataRef = React.useRef<Float32Array | null>(null);

  // Calculate display scale based on dynamic canvas size
  const displayScale = mainCanvasSize / Math.max(width, height);
  const canvasW = Math.round(width * displayScale);
  const canvasH = Math.round(height * displayScale);

  // Effective loop end
  const effectiveLoopEnd = loopEnd < 0 ? nSlices - 1 : loopEnd;

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
        console.log("Show3D: WebGPU FFT ready!");
      } else {
        console.log("Show3D: Using CPU FFT fallback");
      }
    });
  }, []);

  // Prevent page scroll when scrolling on main image canvas only
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const el = overlayRef.current;
    el?.addEventListener("wheel", preventDefault, { passive: false });
    return () => el?.removeEventListener("wheel", preventDefault);
  }, []);

  // -------------------------------------------------------------------------
  // Playback Logic
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (playing) {
      const intervalMs = 1000 / fps;
      playIntervalRef.current = window.setInterval(() => {
        setSliceIdx((prev: number) => {
          const start = Math.max(0, Math.min(loopStart, nSlices - 1));
          const end = Math.max(start, Math.min(effectiveLoopEnd, nSlices - 1));
          let next = prev + (reverse ? -1 : 1);

          if (reverse) {
            if (next < start) {
              return loop ? end : start;
            }
          } else {
            if (next > end) {
              return loop ? start : end;
            }
          }
          return next;
        });
      }, intervalMs);
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    };
  }, [playing, fps, reverse, loop, loopStart, effectiveLoopEnd, nSlices, setSliceIdx]);

  // -------------------------------------------------------------------------
  // Parse frame data
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!frameBytes) return;
    const bytes = extractBytes(frameBytes);
    if (bytes.length === 0) return;
    const floatData = new Float32Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) {
      floatData[i] = bytes[i];
    }
    rawFrameDataRef.current = floatData;
  }, [frameBytes]);

  // -------------------------------------------------------------------------
  // Render Main Canvas with Zoom/Pan
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!canvasRef.current || !frameBytes) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const data = extractBytes(frameBytes);
    if (data.length === 0) return;

    const lut = COLORMAPS[cmap] || COLORMAPS.inferno;

    // Create offscreen canvas at native resolution
    const offscreen = document.createElement("canvas");
    offscreen.width = width;
    offscreen.height = height;
    const offCtx = offscreen.getContext("2d");
    if (!offCtx) return;

    const imgData = offCtx.createImageData(width, height);
    const rgba = imgData.data;

    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      const j = i * 4;
      const lutIdx = v * 3;
      rgba[j] = lut[lutIdx];
      rgba[j + 1] = lut[lutIdx + 1];
      rgba[j + 2] = lut[lutIdx + 2];
      rgba[j + 3] = 255;
    }
    offCtx.putImageData(imgData, 0, 0);

    // Draw with zoom/pan
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.drawImage(offscreen, 0, 0, width * displayScale, height * displayScale);
    ctx.restore();
  }, [frameBytes, width, height, cmap, zoom, panX, panY, displayScale]);

  // -------------------------------------------------------------------------
  // Render Overlay
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!overlayRef.current) return;
    const ctx = overlayRef.current.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvasW, canvasH);

    // Scale bar (pixel_size is in nm, convert to Angstroms for drawScaleBar)
    if (pixelSize > 0 && scaleBarVisible) {
      const pixelSizeAngstrom = pixelSize * 10; // nm -> Angstroms
      drawScaleBar(ctx, canvasW, canvasH, width, pixelSizeAngstrom, displayScale * zoom, scaleBarLengthPx, scaleBarThicknessPx, scaleBarFontSizePx);
    }

    // Zoom indicator
    if (zoom !== 1) {
      ctx.font = "11px sans-serif";
      ctx.fillStyle = "white";
      ctx.textAlign = "left";
      ctx.fillText(String(zoom.toFixed(1)) + "×", 10, canvasH - 10);
    }

    // ROI
    if (roiActive) {
      const screenX = roiX * displayScale * zoom + panX;
      const screenY = roiY * displayScale * zoom + panY;
      const screenRadius = roiRadius * displayScale * zoom;
      const screenWidth = roiWidth * displayScale * zoom;
      const screenHeight = roiHeight * displayScale * zoom;
      drawROI(ctx, screenX, screenY, roiShape || "circle", screenRadius, screenWidth, screenHeight, isDraggingROI);
    }
  }, [pixelSize, scaleBarVisible, scaleBarLengthPx, scaleBarThicknessPx, scaleBarFontSizePx, roiActive, roiShape, roiX, roiY, roiRadius, roiWidth, roiHeight, isDraggingROI, width, canvasW, canvasH, displayScale, zoom, panX, panY]);

  // -------------------------------------------------------------------------
  // Render FFT with WebGPU and zoom/pan
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || !fftCanvasRef.current || !rawFrameDataRef.current) return;
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

      // Draw centered with fixed zoom for better visibility
      const scale = panelSize / Math.max(width, height);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, panelSize, panelSize);
      ctx.save();

      // Center the FFT with default zoom
      const centerOffsetX = (panelSize - width * scale * DEFAULT_FFT_ZOOM) / 2;
      const centerOffsetY = (panelSize - height * scale * DEFAULT_FFT_ZOOM) / 2;

      ctx.translate(centerOffsetX, centerOffsetY);
      ctx.scale(DEFAULT_FFT_ZOOM, DEFAULT_FFT_ZOOM);
      ctx.drawImage(offscreen, 0, 0, width * scale, height * scale);
      ctx.restore();
    };

    const computeFFT = async () => {
      const data = rawFrameDataRef.current!;
      const real = data.slice();
      const imag = new Float32Array(data.length);

      if (gpuFFTRef.current && gpuReady) {
        // WebGPU FFT
        const { real: fReal, imag: fImag } = await gpuFFTRef.current.fft2D(real, imag, width, height, false);
        fftshift(fReal, width, height);
        fftshift(fImag, width, height);

        // Compute magnitude
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
  }, [showFft, frameBytes, width, height, gpuReady, panelSize]);

  // -------------------------------------------------------------------------
  // Render Histogram (shown together with FFT)
  // -------------------------------------------------------------------------
  React.useEffect(() => {
    if (!showFft || !histCanvasRef.current) return;

    const canvas = histCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = panelSize;
    const h = panelSize;

    // Always clear and fill background
    ctx.fillStyle = colors.bgPanel;
    ctx.fillRect(0, 0, w, h);

    // Only draw bars if we have data
    if (!histogramBins || !histogramCounts || histogramCounts.length === 0) return;

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
  }, [showFft, histogramBins, histogramCounts, panelSize]);

  // -------------------------------------------------------------------------
  // Mouse Handlers for Zoom/Pan
  // -------------------------------------------------------------------------
  const handleWheel = (e: React.WheelEvent) => {
    const canvas = overlayRef.current;
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

  const handleDoubleClick = () => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  };

  // -------------------------------------------------------------------------
  // ROI Drag Handlers
  // -------------------------------------------------------------------------
  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (roiActive) {
      setIsDraggingROI(true);
      updateROI(e);
    } else {
      // Pan mode
      setIsDraggingPan(true);
      setPanStart({ x: e.clientX, y: e.clientY, pX: panX, pY: panY });
    }
  };

  const handleCanvasMouseMove = (e: React.MouseEvent) => {
    if (isDraggingROI) {
      updateROI(e);
    } else if (isDraggingPan && panStart) {
      const canvas = overlayRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const dx = (e.clientX - panStart.x) * scaleX;
      const dy = (e.clientY - panStart.y) * scaleY;
      setPanX(panStart.pX + dx);
      setPanY(panStart.pY + dy);
    }
  };

  const handleCanvasMouseUp = () => {
    setIsDraggingROI(false);
    setIsDraggingPan(false);
    setPanStart(null);
  };

  const handleCanvasMouseLeave = () => {
    setIsDraggingROI(false);
    setIsDraggingPan(false);
    setPanStart(null);
  };

  const updateROI = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const canvas = overlayRef.current;
    if (!canvas) return;
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const screenX = (e.clientX - rect.left) * scaleX;
    const screenY = (e.clientY - rect.top) * scaleY;
    const x = Math.floor((screenX - panX) / (displayScale * zoom));
    const y = Math.floor((screenY - panY) / (displayScale * zoom));
    setRoiX(Math.max(0, Math.min(width - 1, x)));
    setRoiY(Math.max(0, Math.min(height - 1, y)));
  };



  // -------------------------------------------------------------------------
  // Resize Handlers (all JS-side, no Python round-trips)
  // -------------------------------------------------------------------------
  const handleMainResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingMain(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: mainCanvasSize });
  };

  const handlePanelResizeStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsResizingPanel(true);
    setResizeStart({ x: e.clientX, y: e.clientY, size: panelSize });
  };

  // Global resize handlers (attached to document for smooth dragging)
  React.useEffect(() => {
    if (!isResizingMain && !isResizingPanel) return;

    const handleMouseMove = (e: MouseEvent) => {
      if (!resizeStart) return;
      const delta = Math.max(e.clientX - resizeStart.x, e.clientY - resizeStart.y);

      if (isResizingMain) {
        const newSize = Math.max(150, Math.min(600, resizeStart.size + delta));
        setMainCanvasSize(newSize);
      } else if (isResizingPanel) {
        const newSize = Math.max(80, Math.min(250, resizeStart.size + delta));
        setPanelSize(newSize);
      }
    };

    const handleMouseUp = () => {
      setIsResizingMain(false);
      setIsResizingPanel(false);
      setResizeStart(null);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingMain, isResizingPanel, resizeStart]);

  // -------------------------------------------------------------------------
  // Keyboard Navigation
  // -------------------------------------------------------------------------
  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case " ":
        e.preventDefault();
        setPlaying(!playing);
        break;
      case "ArrowLeft":
      case "ArrowDown":
        e.preventDefault();
        setSliceIdx(Math.max(0, sliceIdx - 1));
        break;
      case "ArrowRight":
      case "ArrowUp":
        e.preventDefault();
        setSliceIdx(Math.min(nSlices - 1, sliceIdx + 1));
        break;
      case "Home":
        e.preventDefault();
        setSliceIdx(0);
        break;
      case "End":
        e.preventDefault();
        setSliceIdx(nSlices - 1);
        break;
      case "r":
      case "R":
        handleDoubleClick();
        break;
    }
  };

  return (
    <Box
      className="show3d-root"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      sx={{ p: 1.5, bgcolor: colors.bg, borderRadius: 1 }}
    >
      {/* Title */}
      {title && (
        <Typography sx={{ color: colors.accent, fontWeight: "bold", mb: 1, fontSize: 13 }}>
          {title}
        </Typography>
      )}

      {/* Main layout */}
      <Stack direction="row" spacing={1.5}>
        {/* Image area */}
        <Box>
          <Box
            sx={{ position: "relative", bgcolor: "#000", border: "1px solid " + colors.border, borderRadius: 0.5, cursor: roiActive ? "crosshair" : "grab" }}
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseLeave}
            onWheel={handleWheel}
            onDoubleClick={handleDoubleClick}
          >
            <canvas
              ref={canvasRef}
              width={canvasW}
              height={canvasH}
              style={{ width: canvasW, height: canvasH, imageRendering: "pixelated" }}
            />
            <canvas
              ref={overlayRef}
              width={canvasW}
              height={canvasH}
              style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
            />
            {/* Resize handle - bottom right corner */}
            <Box
              onMouseDown={handleMainResizeStart}
              sx={{
                position: "absolute", bottom: 0, right: 0, width: 16, height: 16,
                cursor: "nwse-resize", opacity: 0.6,
                background: "linear-gradient(135deg, transparent 50%, " + colors.accent + " 50%)",
                borderRadius: "0 0 4px 0",
                "&:hover": { opacity: 1 }
              }}
            />
          </Box>
          {/* Label & Timestamp */}
          <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
            <Typography sx={{ color: colors.textSecondary, fontSize: 11 }}>
              {labels?.[sliceIdx] || "Frame " + String(sliceIdx + 1)}
            </Typography>
            {timestamps && timestamps.length > 0 && (
              <Typography sx={{ color: colors.accent, fontSize: 11, fontFamily: "monospace" }}>
                {formatNumber(currentTimestamp, 2)} {timestampUnit}
              </Typography>
            )}
          </Stack>
        </Box>

        {/* Side panels - FFT and Histogram (shown together) */}
        {showFft && (
          <Stack spacing={1}>
            <Box sx={{ position: "relative", bgcolor: colors.bgPanel, border: "1px solid " + colors.border, borderRadius: 0.5, p: 0.75 }}>
              <Typography sx={{ fontSize: 10, color: colors.textMuted, textTransform: "uppercase", mb: 0.5 }}>
                FFT
              </Typography>
              <canvas
                ref={fftCanvasRef}
                width={panelSize}
                height={panelSize}
                style={{ width: panelSize, height: panelSize, imageRendering: "pixelated" }}
              />
              {/* Resize handle */}
              <Box
                onMouseDown={handlePanelResizeStart}
                sx={{
                  position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                  cursor: "nwse-resize", opacity: 0.5,
                  background: "linear-gradient(135deg, transparent 50%, " + colors.textMuted + " 50%)",
                  "&:hover": { opacity: 1 }
                }}
              />
            </Box>
            <Box sx={{ position: "relative", bgcolor: colors.bgPanel, border: "1px solid " + colors.border, borderRadius: 0.5, p: 0.75 }}>
              <Typography sx={{ fontSize: 10, color: colors.textMuted, textTransform: "uppercase", mb: 0.5 }}>
                Histogram
              </Typography>
              <canvas ref={histCanvasRef} width={panelSize} height={panelSize} style={{ width: panelSize, height: panelSize }} />
              {/* Resize handle */}
              <Box
                onMouseDown={handlePanelResizeStart}
                sx={{
                  position: "absolute", bottom: 2, right: 2, width: 12, height: 12,
                  cursor: "nwse-resize", opacity: 0.5,
                  background: "linear-gradient(135deg, transparent 50%, " + colors.textMuted + " 50%)",
                  "&:hover": { opacity: 1 }
                }}
              />
            </Box>
          </Stack>
        )}
      </Stack>

      {/* Controls */}
      <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: "wrap" }} alignItems="center">
        {/* Playback buttons */}
        <Stack direction="row" spacing={0.5} sx={{ bgcolor: colors.bgPanel, px: 1, py: 0.5, borderRadius: 0.5, border: "1px solid " + colors.border }}>
          <IconButton
            size="small"
            onClick={() => { setReverse(true); setPlaying(true); }}
            sx={{ color: reverse && playing ? colors.accent : colors.textMuted, fontSize: 12, p: 0.5 }}
          >
            ◀◀
          </IconButton>
          <IconButton
            size="small"
            onClick={() => setPlaying(!playing)}
            sx={{ color: colors.accent, fontSize: 14, p: 0.5 }}
          >
            {playing ? "⏸" : "▶"}
          </IconButton>
          <IconButton
            size="small"
            onClick={() => { setReverse(false); setPlaying(true); }}
            sx={{ color: !reverse && playing ? colors.accent : colors.textMuted, fontSize: 12, p: 0.5 }}
          >
            ▶▶
          </IconButton>
          <IconButton
            size="small"
            onClick={() => { setPlaying(false); setSliceIdx(loopStart); }}
            sx={{ color: colors.textMuted, fontSize: 12, p: 0.5 }}
          >
            ⏹
          </IconButton>
        </Stack>

        {/* Frame slider */}
        <Slider
          value={sliceIdx}
          min={0}
          max={nSlices - 1}
          onChange={(_, v) => setSliceIdx(v as number)}
          sx={{ color: colors.accent, flex: 1, minWidth: 150 }}
        />

        {/* Frame counter */}
        <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: colors.textMuted, minWidth: 50, textAlign: "center" }}>
          {sliceIdx + 1}/{nSlices}
        </Typography>
      </Stack>

      {/* Second row: Organized control groups */}
      <Stack direction="row" spacing={1.5} sx={{ mt: 1, flexWrap: "wrap" }} alignItems="center">
        {/* Playback group */}
        <Stack direction="row" spacing={1} alignItems="center" sx={{ bgcolor: colors.bgPanel, px: 1, py: 0.5, borderRadius: 0.5, border: "1px solid " + colors.border }}>
          <Typography sx={{ fontSize: 10, color: colors.textDim }}>ms</Typography>
          <Slider
            value={Math.round(1000 / (fps || 1))}
            min={100}
            max={2000}
            step={100}
            onChange={(_, v) => setFps(1000 / (v as number))}
            sx={{ color: colors.accent, width: 50 }}
          />
          <Typography sx={{ fontSize: 10, color: colors.textMuted, minWidth: 30 }}>{Math.round(1000 / (fps || 1))}</Typography>
          <Switch size="small" checked={loop} onChange={() => setLoop(!loop)} sx={{ '& .MuiSwitch-thumb': { width: 10, height: 10 }, '& .MuiSwitch-switchBase': { padding: '5px' } }} />
          <Typography sx={{ fontSize: 10, color: colors.textMuted }}>Loop</Typography>
        </Stack>

        {/* Display group */}
        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ bgcolor: colors.bgPanel, px: 1, py: 0.5, borderRadius: 0.5, border: "1px solid " + colors.border }}>
          <Switch size="small" checked={logScale} onChange={() => setLogScale(!logScale)} sx={{ '& .MuiSwitch-thumb': { width: 10, height: 10 }, '& .MuiSwitch-switchBase': { padding: '5px' } }} />
          <Typography sx={{ fontSize: 10, color: colors.textMuted, mr: 0.5 }}>Log</Typography>
          <Switch size="small" checked={autoContrast} onChange={() => setAutoContrast(!autoContrast)} sx={{ '& .MuiSwitch-thumb': { width: 10, height: 10 }, '& .MuiSwitch-switchBase': { padding: '5px' } }} />
          <Typography sx={{ fontSize: 10, color: colors.textMuted, mr: 0.5 }}>Auto</Typography>
          <Switch size="small" checked={showFft} onChange={() => setShowFft(!showFft)} sx={{ '& .MuiSwitch-thumb': { width: 10, height: 10 }, '& .MuiSwitch-switchBase': { padding: '5px' } }} />
          <Typography sx={{ fontSize: 10, color: colors.textMuted }}>FFT</Typography>
        </Stack>

        {/* ROI group */}
        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ bgcolor: colors.bgPanel, px: 1, py: 0.5, borderRadius: 0.5, border: "1px solid " + colors.border }}>
          <Switch size="small" checked={roiActive} onChange={() => { setRoiActive(!roiActive); if (!roiActive) { setRoiX(Math.floor(width / 2)); setRoiY(Math.floor(height / 2)); } }} sx={{ '& .MuiSwitch-thumb': { width: 10, height: 10 }, '& .MuiSwitch-switchBase': { padding: '5px' } }} />
          <Typography sx={{ fontSize: 10, color: colors.textMuted }}>ROI</Typography>
          {roiActive && (
            <>
              <Select
                size="small"
                value={roiShape || "circle"}
                onChange={(e) => setRoiShape(e.target.value as RoiShape)}
                MenuProps={upwardMenuProps}
                sx={{ minWidth: 60, bgcolor: colors.bgInput, color: colors.textPrimary, fontSize: 10, ml: 0.5, "& .MuiSelect-select": { py: 0.25, px: 0.5 } }}
              >
                {ROI_SHAPES.map((shape) => (
                  <MenuItem key={shape} value={shape} sx={{ fontSize: 10 }}>
                    {shape === "circle" ? "○" : shape === "square" ? "□" : "▭"} {shape}
                  </MenuItem>
                ))}
              </Select>
              <Slider
                value={roiShape === "rectangle" ? roiWidth : roiRadius}
                min={5}
                max={Math.min(width, height) / 2}
                onChange={(_, v) => roiShape === "rectangle" ? setRoiWidth(v as number) : setRoiRadius(v as number)}
                sx={{ color: colors.accent, width: 40, ml: 0.5 }}
              />
              <Typography sx={{ fontSize: 10, color: colors.textMuted, minWidth: 20 }}>
                {roiShape === "rectangle" ? roiWidth : roiRadius}
              </Typography>
            </>
          )}
        </Stack>

        {/* Colormap */}
        <Select
          size="small"
          value={cmap}
          onChange={(e) => setCmap(e.target.value)}
          MenuProps={upwardMenuProps}
          sx={{ minWidth: 75, bgcolor: colors.bgInput, color: colors.textPrimary, fontSize: 11, "& .MuiSelect-select": { py: 0.5 } }}
        >
          {COLORMAP_NAMES.map((name) => (
            <MenuItem key={name} value={name} sx={{ fontSize: 11 }}>
              {name}
            </MenuItem>
          ))}
        </Select>
      </Stack>

      {/* Stats bar */}
      {showStats && (
        <Stack direction="row" spacing={2} sx={{ mt: 1, bgcolor: colors.bgPanel, px: 1.5, py: 0.75, borderRadius: 0.5, border: "1px solid " + colors.border }}>
          {[
            { label: "Mean", value: statsMean },
            { label: "Min", value: statsMin },
            { label: "Max", value: statsMax },
            { label: "Std", value: statsStd },
            { label: "ROI", value: roiActive ? roiMean : null },
          ].map(({ label, value }) => (
            <Stack key={label} direction="row" spacing={0.5} alignItems="baseline">
              <Typography sx={{ fontSize: 10, color: colors.textDim }}>{label}</Typography>
              <Typography sx={{ fontSize: 11, fontFamily: "monospace", color: colors.accent }}>
                {value !== null ? formatNumber(value) : "-"}
              </Typography>
            </Stack>
          ))}
        </Stack>
      )}
    </Box>
  );
}

export const render = createRender(Show3D);
