# DCIMG DataLoader Integration Plan for masknmf-toolbox

## Goal
- Add `.dcimg` support by implementing a new loader class based on `LazyFrameLoader`.
- Keep existing motion-correction/compression APIs unchanged.
- Support large files without full-memory loading.

## Confirmed Direction
- Main approach: add `DcimgArray` to toolbox loaders.
- Do not rely on a one-off converter as the core solution.
- Use optional dependency `lens-biophotonics/dcimg` from latest GitHub code.
- For local testing, process only first 500 frames.

## External Dependency Rule
- Use GitHub source, not old PyPI release.
- Install example:
  - `pip install "git+https://github.com/lens-biophotonics/dcimg.git@3f1e2eec27a4e414903b0fcd1da711e7f565dcce"`
- Known caution:
  - In current validation, `dcimg` read succeeded with `numpy<2` and failed with `numpy>=2`.
  - `cvxpy` may require `numpy>=2`, but registration path itself does not depend on `cvxpy`.
  - Keep this as a known environment compatibility note and proceed with current registration-focused setup.

## Scope
### In scope
- Add `DcimgArray(LazyFrameLoader)` in `masknmf/arrays/data_loaders.py`.
- Export it from `masknmf/arrays/__init__.py` and `masknmf/__init__.py`.
- Add a dev script for sample conversion/validation.
- Update README usage notes.

### Out of scope
- Changes to motion-correction algorithms.
- Changes to demixing/compression algorithms.

## Implementation Details
1. Add `DcimgArray`
- File: `masknmf/arrays/data_loaders.py`
- Base class: `LazyFrameLoader`
- Required members:
  - `dtype` -> return `np.float32` for consistency with existing loaders
  - `shape` -> `(T, H, W)`
  - `_compute_at_indices(indices)` -> support `int`, `list`, `slice`
- Behavior:
  - lazy import `dcimg`
  - clear `ImportError` message if not installed
  - always return numpy array with float32

2. Export from package
- Add `DcimgArray` to:
  - `masknmf/arrays/__init__.py`
  - `masknmf/__init__.py`

3. Add dev sample script
- New file: `dev/dcimg_to_hdf5_sample.py`
- Default input sample:
  - `demo_data/250206-UK6-1-F=4_power=5mW_00001.dcimg`
- Safety defaults for large file:
  - `--max_frames` default `500`
  - process only `frames[0:max_frames]`
- Output format (toolbox-compatible):
  - dataset `/motion_corrected`
  - shape `(min(T, max_frames), H, W)`
  - attrs: `source_path`, `source_format`, `frames`, `height`, `width`, `dtype`

4. README update
- Mention `.dcimg` support.
- Mention GitHub install rule for `dcimg`.
- Add example command using `--max_frames 500`.

## Test Plan
1. Unit tests for `DcimgArray`
- `shape` is `(T,H,W)`
- indexing by `int/list/slice` works
- output dtype is float32

2. Integration tests (small run)
- `DcimgArray` -> `compute_template`
- `RegistrationArray(...).export(...)` creates `/motion_corrected`

3. Sample-data validation
- Use `demo_data/250206-UK6-1-F=4_power=5mW_00001.dcimg`
- Run with `max_frames=500`
- Verify output is readable by `masknmf.Hdf5Array(path, "motion_corrected")`

## Acceptance Criteria
- `.dcimg` can be used as input in normal toolbox pipeline.
- Sample run completes with first 500 frames only.
- No breaking change for existing users.

## Default Values
- `max_frames`: `500`
- `batch_size`: `200` (dev/test)
- `compression`: `lzf` (if HDF5 writer is used)
- `first_4px_correction`: `False`
