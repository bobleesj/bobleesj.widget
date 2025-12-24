import * as React from "react";
import { createRender, useModelState } from "@anywidget/react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Slider from "@mui/material/Slider";
import JSZip from "jszip";
import { getWebGPUFFT, WebGPUFFT } from "./webgpu-fft";
import { COLORMAP_POINTS, COLORMAP_NAMES, fft2d, fftshift, applyBandPassFilter } from "./shared";
import { colors, typography, controlPanel, container } from "./CONFIG";
import { upwardMenuProps } from "./components";
import "./reconstruct.css";

// Use COLORMAP_POINTS for interpolation (reconstruct uses point-based interpolation)
const COLORMAPS = COLORMAP_POINTS;

// ============================================================================
// Scale Bar Component
// ============================================================================
const ScaleBar = ({ zoom, size, label }: { zoom: number; size: number; label: string }) => {
  const scaleBarPx = 50;
  const realPixels = Math.round(scaleBarPx / zoom);

  return (
    <Box sx={{
      position: "absolute",
      bottom: 8,
      right: 8,
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      pointerEvents: "none"
    }}>
      <Typography sx={{ color: "#fff", fontSize: 9, textShadow: "0 0 3px #000", mb: 0.3 }}>
        {realPixels} px
      </Typography>
      <Box sx={{
        width: scaleBarPx,
        height: 3,
        bgcolor: "#fff",
        boxShadow: "0 0 3px #000"
      }} />
    </Box>
  );
};

// ============================================================================
// Zoom Indicator Component
// ============================================================================
const ZoomIndicator = ({ zoom }: { zoom: number }) => (
  <Typography sx={{
    position: "absolute",
    bottom: 8,
    left: 8,
    color: "#fff",
    fontSize: 10,
    textShadow: "0 0 3px #000",
    pointerEvents: "none"
  }}>
    {zoom.toFixed(1)}×
  </Typography>
);

// ============================================================================
// Main Component
// ============================================================================
function ReconstructWidget() {
  // Model state
  const [shapeX] = useModelState<number>("shape_x");
  const [shapeY] = useModelState<number>("shape_y");

  // Sign toggle
  const [phaseSign, setPhaseSign] = useModelState<number>("phase_sign");

  // Display options
  const [colormap, setColormap] = useModelState<string>("colormap");
  const [percentileLow, setPercentileLow] = useModelState<number>("percentile_low");
  const [percentileHigh, setPercentileHigh] = useModelState<number>("percentile_high");

  // Image bytes (raw from Python - no filtering applied)
  const [dpcBytes] = useModelState<DataView>("dpc_bytes");
  const [idpcBytes] = useModelState<DataView>("idpc_bytes");
  const [comXBytes] = useModelState<DataView>("com_x_bytes");
  const [comYBytes] = useModelState<DataView>("com_y_bytes");

  // Stats
  const [rotationAngleDeg] = useModelState<number>("rotation_angle_deg");
  const [nmse] = useModelState<number>("nmse");
  const [iterationsDone] = useModelState<number>("iterations_done");
  const [reconstructionTimeMs] = useModelState<number>("reconstruction_time_ms");
  const [dpcTimeMs] = useModelState<number>("dpc_time_ms");

  // Local state
  const [localColormap, setLocalColormap] = React.useState("inferno");
  const [bandpass, setBandpass] = React.useState<number[]>([0, 0]); // [inner (HP), outer (LP)]
  const [zoom, setZoom] = React.useState(1);
  const [panOffset, setPanOffset] = React.useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = React.useState(false);
  const [dragStart, setDragStart] = React.useState({ x: 0, y: 0 });

  // Destructure bandpass for convenience
  const bpInner = bandpass[0];
  const bpOuter = bandpass[1];

  // WebGPU FFT state
  const gpuFFTRef = React.useRef<WebGPUFFT | null>(null);
  const [gpuReady, setGpuReady] = React.useState(false);

  // Raw image data refs (store Float32 for FFT processing)
  const rawDpcRef = React.useRef<Float32Array | null>(null);
  const rawIdpcRef = React.useRef<Float32Array | null>(null);
  const rawComXRef = React.useRef<Float32Array | null>(null);
  const rawComYRef = React.useRef<Float32Array | null>(null);

  // Canvas refs
  const dpcCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const idpcCanvasRef = React.useRef<HTMLCanvasElement>(null);
  const imageContainerRef = React.useRef<HTMLDivElement>(null);

  // Prevent page scroll when scrolling on images (for Jupyter notebook compatibility)
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    const el = imageContainerRef.current;
    if (el) {
      el.addEventListener("wheel", preventDefault, { passive: false });
      return () => el.removeEventListener("wheel", preventDefault);
    }
  }, []);

  // Initialize WebGPU FFT on mount
  React.useEffect(() => {
    getWebGPUFFT().then(fft => {
      if (fft) {
        gpuFFTRef.current = fft;
        setGpuReady(true);
      }
    });
  }, []);

  // Sync colormap from model
  React.useEffect(() => {
    if (colormap) setLocalColormap(colormap);
  }, [colormap]);

  // Convert bytes to Float32 and store in refs
  const bytesToFloat32 = (bytes: DataView | null): Float32Array | null => {
    if (!bytes) return null;
    const data = new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const float32 = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      float32[i] = data[i];  // Keep 0-255 range for now
    }
    return float32;
  };

  // Update raw data refs when bytes change
  React.useEffect(() => {
    rawDpcRef.current = bytesToFloat32(dpcBytes);
  }, [dpcBytes]);

  React.useEffect(() => {
    rawIdpcRef.current = bytesToFloat32(idpcBytes);
  }, [idpcBytes]);

  React.useEffect(() => {
    rawComXRef.current = bytesToFloat32(comXBytes);
  }, [comXBytes]);

  React.useEffect(() => {
    rawComYRef.current = bytesToFloat32(comYBytes);
  }, [comYBytes]);

  // Render filtered image to canvas - optimized for speed
  const renderFilteredImage = React.useCallback((
    canvasRef: React.RefObject<HTMLCanvasElement>,
    rawData: Float32Array | null,
    width: number,
    height: number,
    cmapName: string,
    hpRadius: number,
    lpRadius: number,
    pLow: number,
    pHigh: number
  ) => {
    const canvas = canvasRef.current;
    if (!canvas || !rawData || rawData.length !== width * height) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Get colormap LUT (flat array: [r0,g0,b0, r1,g1,b1, ...])
    const cmapNested = COLORMAPS[cmapName] || COLORMAPS.inferno;

    // Build flat LUT for fast lookup (256 entries)
    const lut = new Uint8Array(256 * 3);
    const n = cmapNested.length - 1;
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * n;
      const idx = Math.min(Math.floor(t), n - 1);
      const f = t - idx;
      lut[i * 3] = Math.round(cmapNested[idx][0] * (1 - f) + cmapNested[idx + 1][0] * f);
      lut[i * 3 + 1] = Math.round(cmapNested[idx][1] * (1 - f) + cmapNested[idx + 1][1] * f);
      lut[i * 3 + 2] = Math.round(cmapNested[idx][2] * (1 - f) + cmapNested[idx + 1][2] * f);
    }

    // Apply band-pass filter if needed (async)
    const applyFilter = async (): Promise<Float32Array> => {
      if (hpRadius > 0 || lpRadius > 0) {
        if (gpuFFTRef.current && gpuReady) {
          // GPU filtering
          const real = rawData.slice();
          const imag = new Float32Array(real.length);
          const { real: fReal, imag: fImag } = await gpuFFTRef.current.fft2D(real, imag, width, height, false);
          fftshift(fReal, width, height);
          fftshift(fImag, width, height);
          applyBandPassFilter(fReal, fImag, width, height, hpRadius, lpRadius);
          fftshift(fReal, width, height);
          fftshift(fImag, width, height);
          const { real: invReal } = await gpuFFTRef.current.fft2D(fReal, fImag, width, height, true);
          return invReal;
        } else {
          // CPU fallback
          const real = rawData.slice();
          const imag = new Float32Array(real.length);
          fft2d(real, imag, width, height, false);
          fftshift(real, width, height);
          fftshift(imag, width, height);
          applyBandPassFilter(real, imag, width, height, hpRadius, lpRadius);
          fftshift(real, width, height);
          fftshift(imag, width, height);
          fft2d(real, imag, width, height, true);
          return real;
        }
      }
      return rawData;
    };

    // Render function (sync, fast) - applies contrast via percentile
    const renderData = (data: Float32Array) => {
      // Sort a sample for percentile calculation (fast approximation)
      const sorted = Float32Array.from(data).sort((a, b) => a - b);
      const len = sorted.length;
      const loIdx = Math.floor((pLow / 100) * (len - 1));
      const hiIdx = Math.floor((pHigh / 100) * (len - 1));
      const min = sorted[loIdx];
      const max = sorted[hiIdx];
      const range = max - min || 1;
      const scale = 255 / range;

      // Render with colormap using fast LUT lookup
      const imageData = ctx.createImageData(width, height);
      const rgba = imageData.data;

      for (let i = 0; i < data.length; i++) {
        const v = Math.round((data[i] - min) * scale);
        const lutIdx = Math.max(0, Math.min(255, v)) * 3;
        const pi = i * 4;
        rgba[pi] = lut[lutIdx];
        rgba[pi + 1] = lut[lutIdx + 1];
        rgba[pi + 2] = lut[lutIdx + 2];
        rgba[pi + 3] = 255;
      }

      ctx.putImageData(imageData, 0, 0);
    };

    // If no filtering, render synchronously (fast!)
    if (hpRadius === 0 && lpRadius === 0) {
      renderData(rawData);
    } else {
      // With filtering, must be async
      applyFilter().then(renderData);
    }
  }, [gpuReady]);

  // Render all images when filter, colormap, or contrast changes
  React.useEffect(() => {
    renderFilteredImage(dpcCanvasRef, rawDpcRef.current, shapeY, shapeX, localColormap, bpInner, bpOuter, percentileLow, percentileHigh);
  }, [dpcBytes, shapeX, shapeY, localColormap, bpInner, bpOuter, percentileLow, percentileHigh, gpuReady, renderFilteredImage]);

  React.useEffect(() => {
    renderFilteredImage(idpcCanvasRef, rawIdpcRef.current, shapeY, shapeX, localColormap, bpInner, bpOuter, percentileLow, percentileHigh);
  }, [idpcBytes, shapeX, shapeY, localColormap, bpInner, bpOuter, percentileLow, percentileHigh, gpuReady, renderFilteredImage]);

  // Zoom/pan handlers
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((prev) => Math.min(Math.max(prev * delta, 0.5), 10));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - panOffset.x, y: e.clientY - panOffset.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPanOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Double-click to reset zoom
  const handleDoubleClick = () => {
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  // Reset handler
  const handleReset = () => {
    setPhaseSign(1);
    setLocalColormap("inferno");
    setColormap("inferno");
    setPercentileLow(1);
    setPercentileHigh(99);
    setBandpass([0, 0]);
    setZoom(1);
    setPanOffset({ x: 0, y: 0 });
  };

  // Export handler
  const handleExport = async () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const zip = new JSZip();

    const metadata = {
      exported_at: new Date().toISOString(),
      scan_shape: { x: shapeX, y: shapeY },
      rotation_angle_deg: rotationAngleDeg,
      phase_sign: phaseSign,
      dpc_time_ms: dpcTimeMs,
      idpc: {
        iterations: iterationsDone,
        nmse: nmse,
        time_ms: reconstructionTimeMs,
      },
      display: {
        colormap: localColormap,
        percentile_low: percentileLow,
        percentile_high: percentileHigh,
        highpass_radius: bpInner,
        lowpass_radius: bpOuter,
      },
    };
    zip.file("metadata.json", JSON.stringify(metadata, null, 2));

    const canvasToBlob = (canvas: HTMLCanvasElement): Promise<Blob> => {
      return new Promise((resolve) => {
        canvas.toBlob((blob) => resolve(blob!), 'image/png');
      });
    };

    if (dpcCanvasRef.current) zip.file("dpc.png", await canvasToBlob(dpcCanvasRef.current));
    if (idpcCanvasRef.current) zip.file("idpc.png", await canvasToBlob(idpcCanvasRef.current));

    const zipBlob = await zip.generateAsync({ type: "blob" });
    const link = document.createElement('a');
    link.download = `phase_export_${timestamp}.zip`;
    link.href = URL.createObjectURL(zipBlob);
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const canvasStyle = {
    width: "100%",
    height: "100%",
    imageRendering: "pixelated" as const,
    transform: `scale(${zoom}) translate(${panOffset.x / zoom}px, ${panOffset.y / zoom}px)`,
    cursor: zoom > 1 ? "grab" : "default"
  };

  const imageSize = 300;
  const maxFilterValue = Math.max(Math.min(shapeX, shapeY) / 2, 50);

  return (
    <Box sx={{ ...container.root }}>
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
        <Typography variant="h6" sx={{ ...typography.title }}>
          Phase Reconstruction
        </Typography>
        <Stack direction="row" spacing={1} alignItems="center">
          <Typography sx={{ ...typography.labelSmall }}>
            {shapeX}×{shapeY} | rot: {rotationAngleDeg.toFixed(1)}° | DPC: {dpcTimeMs.toFixed(1)}ms | iDPC: {reconstructionTimeMs.toFixed(1)}ms ({iterationsDone} iters){zoom !== 1 ? ` | ${zoom.toFixed(1)}×` : ""}{gpuReady ? " | GPU" : ""}
          </Typography>
          <Typography
            component="span"
            onClick={handleReset}
            sx={{ ...controlPanel.button }}
          >
            Reset
          </Typography>
          <Typography
            component="span"
            onClick={handleExport}
            sx={{ ...controlPanel.button }}
          >
            Export
          </Typography>
        </Stack>
      </Stack>

      {/* Main content: Images + Filter slider */}
      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        {/* 2x2 Image grid */}
        <Box
          ref={imageContainerRef}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={handleDoubleClick}
        >
          {/* Row 1: DPC and iDPC */}
          <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
            <Box>
              <Typography sx={{ ...typography.label, mb: 0.5 }}>
                DPC (FFT) {(bpInner > 0 || bpOuter > 0) && <span style={{ color: colors.accent }}>(filtered)</span>}
              </Typography>
              <Box sx={{ ...container.imageBox, width: imageSize, height: imageSize }}>
                <canvas ref={dpcCanvasRef} width={shapeY} height={shapeX} style={canvasStyle} />
                <ScaleBar zoom={zoom} size={shapeX} label="px" />
                {zoom !== 1 && <ZoomIndicator zoom={zoom} />}
              </Box>
            </Box>
            <Box>
              <Typography sx={{ ...typography.label, mb: 0.5 }}>
                iDPC ({iterationsDone} iters) {(bpInner > 0 || bpOuter > 0) && <span style={{ color: colors.accent }}>(filtered)</span>}
              </Typography>
              <Box sx={{ ...container.imageBox, width: imageSize, height: imageSize }}>
                <canvas ref={idpcCanvasRef} width={shapeY} height={shapeX} style={canvasStyle} />
                <ScaleBar zoom={zoom} size={shapeX} label="px" />
                {zoom !== 1 && <ZoomIndicator zoom={zoom} />}
              </Box>
            </Box>
          </Stack>
        </Box>

        {/* Band-pass Filter - single dual-handle vertical slider */}
        <Stack spacing={0.5} sx={{ height: imageSize }} alignItems="center">
          <Typography sx={{ color: "#09f", fontSize: 9 }}>{bpOuter || "-"}</Typography>
          <Slider
            value={bandpass}
            onChange={(_, v) => setBandpass(v as number[])}
            min={0}
            max={maxFilterValue}
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

      {/* Controls - minimal */}
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        {/* Sign toggle */}
        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
          <Typography sx={{ ...typography.label }}>Sign:</Typography>
          <Typography
            onClick={() => setPhaseSign(1)}
            sx={{
              color: phaseSign === 1 ? colors.accentGreen : colors.textDim,
              fontSize: 12,
              cursor: "pointer",
              px: 0.5,
              fontWeight: phaseSign === 1 ? "bold" : "normal"
            }}
          >+</Typography>
          <Typography
            onClick={() => setPhaseSign(-1)}
            sx={{
              color: phaseSign === -1 ? colors.accentRed : colors.textDim,
              fontSize: 12,
              cursor: "pointer",
              px: 0.5,
              fontWeight: phaseSign === -1 ? "bold" : "normal"
            }}
          >−</Typography>
        </Stack>

        {/* Colormap */}
        <Stack direction="row" spacing={1} alignItems="center" sx={{ ...controlPanel.group }}>
          <Typography sx={{ ...typography.label }}>Colormap:</Typography>
          <Select
            value={localColormap}
            onChange={(e) => { setLocalColormap(e.target.value); setColormap(e.target.value); }}
            size="small"
            sx={{ ...controlPanel.select }}
            MenuProps={upwardMenuProps}
          >
            {COLORMAP_NAMES.map((name) => (
              <MenuItem key={name} value={name}>{name.charAt(0).toUpperCase() + name.slice(1)}</MenuItem>
            ))}
          </Select>
        </Stack>

        {/* Contrast */}
        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
          <Typography sx={{ ...typography.label }}>Contrast:</Typography>
          <Typography sx={{ ...typography.value }}>{percentileLow.toFixed(0)}%</Typography>
          <Slider
            value={[percentileLow, percentileHigh]}
            onChange={(_, v) => {
              const [lo, hi] = v as number[];
              setPercentileLow(lo);
              setPercentileHigh(hi);
            }}
            min={0}
            max={100}
            size="small"
            sx={{ width: 100 }}
          />
          <Typography sx={{ ...typography.value }}>{percentileHigh.toFixed(0)}%</Typography>
        </Stack>
      </Stack>
    </Box>
  );
}

export const render = createRender(ReconstructWidget);
