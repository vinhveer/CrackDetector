# UI (PARSE 7)

## Overview

The desktop UI runs the pipeline fully in-memory and visualizes the main stages and per-region outputs.

## Stage tabs

The left side uses tabs to display stage images (BGR `uint8` as produced by OpenCV):

- `Input`
- `Preprocess`
- `DINO Boxes`
- `SAM Raw Mask`
- `Geometry Input`
- `Geometry Kept`
- `Final Overlay`
- `Region Inspector`

All image views support:

- Mouse wheel zoom
- Click-and-drag pan

## Region table

The right-side table shows one row per region, sorted:

- Kept first
- Then by skeleton length (descending)

Columns:

- `region_id`
- `kept`
- `dropped_reason`
- `area`
- `skeleton_length`
- `width_mean`
- `endpoint_count`
- `dino_conf`

Clicking a row highlights that region (mask + bbox) on the currently selected stage tab.

## Runtime toggles

Top bar controls:

- **Enable geometry filter**: toggles PARSE 6 hard-reject
- **Preprocess variants**: `base`, `blur_boost`, `ridge`
- **Prompt mode**: `disabled`, `one_pass`, `multi_pass`

Changing a toggle re-runs the pipeline in-memory.

## Saving

By default, the UI disables disk debug writes.

Use **Save Debug Bundle** to export:

- `pipeline_result.json` (shallow metadata via `PipelineResult.to_dict(shallow=True)`)
- selected stage images as `.jpg`/`.png`
