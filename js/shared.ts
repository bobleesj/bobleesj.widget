/**
 * Shared utilities for widget components.
 * Re-exports core utilities and adds FFT, band-pass filtering, zoom/pan helpers.
 */

import * as React from "react";

// Re-export colormaps from core
export { COLORMAP_NAMES, COLORMAP_POINTS, COLORMAPS, createColormapLUT, applyColormapValue, applyColormapToImage } from "./core/colormaps";

// ============================================================================
// CPU FFT Implementation (Cooley-Tukey radix-2) - Fallback when WebGPU unavailable
// ============================================================================

export function fft1d(real: Float32Array, imag: Float32Array, inverse: boolean = false) {
  const n = real.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }

  // Cooley-Tukey FFT
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curReal = 1, curImag = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k;
        const oddIdx = i + k + halfLen;

        const tReal = curReal * real[oddIdx] - curImag * imag[oddIdx];
        const tImag = curReal * imag[oddIdx] + curImag * real[oddIdx];

        real[oddIdx] = real[evenIdx] - tReal;
        imag[oddIdx] = imag[evenIdx] - tImag;
        real[evenIdx] += tReal;
        imag[evenIdx] += tImag;

        const newReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = newReal;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

export function fft2d(real: Float32Array, imag: Float32Array, width: number, height: number, inverse: boolean = false) {
  // FFT on rows
  const rowReal = new Float32Array(width);
  const rowImag = new Float32Array(width);
  for (let y = 0; y < height; y++) {
    const offset = y * width;
    for (let x = 0; x < width; x++) {
      rowReal[x] = real[offset + x];
      rowImag[x] = imag[offset + x];
    }
    fft1d(rowReal, rowImag, inverse);
    for (let x = 0; x < width; x++) {
      real[offset + x] = rowReal[x];
      imag[offset + x] = rowImag[x];
    }
  }

  // FFT on columns
  const colReal = new Float32Array(height);
  const colImag = new Float32Array(height);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      colReal[y] = real[y * width + x];
      colImag[y] = imag[y * width + x];
    }
    fft1d(colReal, colImag, inverse);
    for (let y = 0; y < height; y++) {
      real[y * width + x] = colReal[y];
      imag[y * width + x] = colImag[y];
    }
  }
}

export function fftshift(data: Float32Array, width: number, height: number) {
  const halfW = width >> 1;
  const halfH = height >> 1;
  const temp = new Float32Array(width * height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const newY = (y + halfH) % height;
      const newX = (x + halfW) % width;
      temp[newY * width + newX] = data[y * width + x];
    }
  }
  data.set(temp);
}

// ============================================================================
// Band-pass Filter
// ============================================================================

/** Apply band-pass filter in frequency domain (keeps frequencies between inner and outer radius) */
export function applyBandPassFilter(
  real: Float32Array,
  imag: Float32Array,
  width: number,
  height: number,
  innerRadius: number,  // High-pass: remove frequencies below this
  outerRadius: number   // Low-pass: remove frequencies above this
) {
  const centerX = width >> 1;
  const centerY = height >> 1;
  const innerSq = innerRadius * innerRadius;
  const outerSq = outerRadius * outerRadius;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      const distSq = dx * dx + dy * dy;
      const idx = y * width + x;

      // Zero out frequencies outside the band
      if (distSq < innerSq || (outerRadius > 0 && distSq > outerSq)) {
        real[idx] = 0;
        imag[idx] = 0;
      }
    }
  }
}

// ============================================================================
// Zoom/Pan Helpers
// ============================================================================

export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 10;

/** Create a zoom handler for canvas wheel events */
export function createZoomHandler(
  setZoom: (fn: (prev: number) => number) => void,
  setPanX: (fn: (prev: number) => number) => void,
  setPanY: (fn: (prev: number) => number) => void,
  zoom: number,
  panX: number,
  panY: number,
  canvasRef: React.RefObject<HTMLCanvasElement>
) {
  return (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.min(Math.max(zoom * delta, MIN_ZOOM), MAX_ZOOM);
    const scale = newZoom / zoom;
    
    setPanX(prev => mouseX - scale * (mouseX - prev));
    setPanY(prev => mouseY - scale * (mouseY - prev));
    setZoom(() => newZoom);
  };
}

/** Prevent page scroll when scrolling on elements */
export function usePreventScroll(refs: React.RefObject<HTMLElement>[]) {
  React.useEffect(() => {
    const preventDefault = (e: WheelEvent) => e.preventDefault();
    refs.forEach(ref => ref.current?.addEventListener("wheel", preventDefault, { passive: false }));
    return () => refs.forEach(ref => ref.current?.removeEventListener("wheel", preventDefault));
  }, [refs]);
}
