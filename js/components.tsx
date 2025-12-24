/**
 * Shared styling constants and simple UI components for bobleesj.widget.
 * 
 * ARCHITECTURE NOTE: Only styling should be shared here.
 * Widget-specific logic (resize handlers, zoom handlers) should be inlined per-widget.
 */

import * as React from "react";
import Switch from "@mui/material/Switch";
import Select from "@mui/material/Select";
import MenuItem from "@mui/material/MenuItem";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { colors, controlPanel, typography } from "./CONFIG";

// ============================================================================
// Switch Style Constants
// ============================================================================
export const switchStyles = {
    small: {
        '& .MuiSwitch-thumb': { width: 12, height: 12 },
        '& .MuiSwitch-switchBase': { padding: '4px' },
    },
    medium: {
        '& .MuiSwitch-thumb': { width: 14, height: 14 },
        '& .MuiSwitch-switchBase': { padding: '4px' },
    },
};

// ============================================================================
// Select MenuProps for upward dropdown (all widgets use this)
// ============================================================================
export const upwardMenuProps = {
    anchorOrigin: { vertical: "top" as const, horizontal: "left" as const },
    transformOrigin: { vertical: "bottom" as const, horizontal: "left" as const },
    sx: { zIndex: 9999 },
};

// ============================================================================
// LabeledSwitch - Label + Switch combo (optional, use if needed)
// ============================================================================
interface LabeledSwitchProps {
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
    size?: "small" | "medium";
}

export function LabeledSwitch({ label, checked, onChange, size = "small" }: LabeledSwitchProps) {
    return (
        <Stack direction="row" spacing={0.5} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>{label}:</Typography>
            <Switch
                checked={checked}
                onChange={(e) => onChange(e.target.checked)}
                size="small"
                sx={switchStyles[size]}
            />
        </Stack>
    );
}

// ============================================================================
// LabeledSelect - Label + Select dropdown combo (optional, use if needed)
// ============================================================================
interface LabeledSelectProps<T extends string> {
    label: string;
    value: T;
    options: readonly T[] | T[];
    onChange: (value: T) => void;
    formatLabel?: (value: T) => string;
}

export function LabeledSelect<T extends string>({
    label,
    value,
    options,
    onChange,
    formatLabel,
}: LabeledSelectProps<T>) {
    return (
        <Stack direction="row" spacing={1} alignItems="center" sx={{ ...controlPanel.group }}>
            <Typography sx={{ ...typography.label }}>{label}:</Typography>
            <Select
                value={value}
                onChange={(e) => onChange(e.target.value as T)}
                size="small"
                sx={{ ...controlPanel.select }}
                MenuProps={upwardMenuProps}
            >
                {options.map((opt) => (
                    <MenuItem key={opt} value={opt}>
                        {formatLabel ? formatLabel(opt) : opt}
                    </MenuItem>
                ))}
            </Select>
        </Stack>
    );
}
