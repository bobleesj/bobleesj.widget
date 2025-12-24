/**
 * Shared color palette for all bobleesj.widget components.
 * Provides consistent theming across Show2D, Show3D, Show4D, and Reconstruct.
 */

export const colors = {
  // Backgrounds
  bg: "#1a1a1a",
  bgPanel: "#222",
  bgInput: "#333",
  bgCanvas: "#000",

  // Borders
  border: "#444",
  borderLight: "#555",

  // Text
  textPrimary: "#fff",
  textSecondary: "#aaa",
  textMuted: "#888",
  textDim: "#666",

  // Accent colors
  accent: "#0af",
  accentGreen: "#0f0",
  accentRed: "#f00",
  accentOrange: "#fa0",
  accentCyan: "#0cf",
  accentYellow: "#ff0",
};

// CSS variable export for vanilla JS widgets
export const cssVars = `
  --bg: ${colors.bg};
  --bg-panel: ${colors.bgPanel};
  --bg-input: ${colors.bgInput};
  --bg-canvas: ${colors.bgCanvas};
  --border: ${colors.border};
  --border-light: ${colors.borderLight};
  --text-primary: ${colors.textPrimary};
  --text-secondary: ${colors.textSecondary};
  --text-muted: ${colors.textMuted};
  --text-dim: ${colors.textDim};
  --accent: ${colors.accent};
  --accent-green: ${colors.accentGreen};
  --accent-red: ${colors.accentRed};
`;
