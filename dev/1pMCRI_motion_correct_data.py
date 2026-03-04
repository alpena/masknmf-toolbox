import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch

import masknmf


@dataclass(slots=True)
class MotionCorrectionConfig:
    """Parameters for piecewise-rigid motion correction and output export."""

    input_image_path: str
    out_path: str
    out_tiff_path: str | None = None
    export_tiff_stack: bool = True
    num_blocks_dim1: int = 10
    num_blocks_dim2: int = 10
    overlaps_dim1: int = 5
    overlaps_dim2: int = 5
    max_rigid_shifts_dim1: int = 15
    max_rigid_shifts_dim2: int = 15
    max_deviation_rigid_dim1: int = 2
    max_deviation_rigid_dim2: int = 2
    frame_batch_size: int = 20
    spatial_highpass_sigma: float = 3.0
    input_max_frames: int | None = None
    dcimg_first_4px_correction: bool = False
    device: str = "auto"


class PrefixFrameLoader(masknmf.LazyFrameLoader):
    """Temporal prefix view over another frame loader."""

    def __init__(self, base_loader: masknmf.LazyFrameLoader, max_frames: int):
        if max_frames <= 0:
            raise ValueError("max_frames must be > 0")
        self._base_loader = base_loader
        self._shape = (
            min(max_frames, base_loader.shape[0]),
            base_loader.shape[1],
            base_loader.shape[2],
        )

    @property
    def dtype(self):
        return self._base_loader.dtype

    @property
    def shape(self):
        return self._shape

    def _compute_at_indices(self, indices):
        if isinstance(indices, int):
            if indices < 0:
                indices += self.shape[0]
            if indices < 0 or indices >= self.shape[0]:
                raise IndexError("Frame index out of range")
            return self._base_loader[indices]
        if isinstance(indices, list):
            if len(indices) == 0:
                return np.empty((0, self.shape[1], self.shape[2]), dtype=np.float32)
            normalized = []
            for index in indices:
                idx = int(index)
                if idx < 0:
                    idx += self.shape[0]
                if idx < 0 or idx >= self.shape[0]:
                    raise IndexError("Frame index out of range")
                normalized.append(idx)
            frame_list = [np.asarray(self._base_loader[idx]) for idx in normalized]
            return np.stack(frame_list, axis=0)

        start = indices.start or 0
        stop = indices.stop or self.shape[0]
        step = indices.step or 1
        if step == 0:
            raise ValueError("slice step cannot be zero")
        if step > 0 and start >= stop:
            return np.empty((0, self.shape[1], self.shape[2]), dtype=np.float32)
        if step < 0 and start <= stop:
            return np.empty((0, self.shape[1], self.shape[2]), dtype=np.float32)
        data = self._base_loader[slice(start, stop, step)]
        if data.ndim == 2:
            data = data[None, :, :]
        return np.asarray(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run piecewise-rigid motion correction for 1p imaging from a TIFF or DCIMG path."
    )
    parser.add_argument(
        "--input_image_path",
        default=None,
        help="Path to a single input movie (.tif/.tiff/.dcimg).",
    )
    parser.add_argument(
        "--out_path",
        default=None,
        help="Path to single-file output HDF5. Required in single-file mode.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Explicit list of input files for batch mode.",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        help="Directory to scan for input files in batch mode.",
    )
    parser.add_argument(
        "--glob_pattern",
        default="*.dcimg",
        help="Glob pattern used with --input_dir (e.g. *.dcimg, *.tif).",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to scan input_dir recursively.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Batch output directory. Required in batch mode.",
    )
    parser.add_argument(
        "--out_suffix",
        default="_moco",
        help="Suffix appended to output stem in batch mode.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files when output exists.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite existing outputs.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print planned jobs without executing them.",
    )
    parser.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue remaining jobs if one fails.",
    )
    parser.add_argument(
        "--out_tiff_path",
        default=None,
        help="Path to output TIFF stack. If omitted, uses out_path with .tif suffix.",
    )
    parser.add_argument(
        "--export_tiff_stack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to also export motion-corrected frames as TIFF stack.",
    )
    parser.add_argument(
        "--num_blocks_dim1",
        type=int,
        default=10,
        help="Number of blocks along image height for piecewise-rigid registration.",
    )
    parser.add_argument(
        "--num_blocks_dim2",
        type=int,
        default=10,
        help="Number of blocks along image width for piecewise-rigid registration.",
    )
    parser.add_argument(
        "--overlaps_dim1",
        type=int,
        default=5,
        help="Block overlap size in pixels along image height.",
    )
    parser.add_argument(
        "--overlaps_dim2",
        type=int,
        default=5,
        help="Block overlap size in pixels along image width.",
    )
    parser.add_argument(
        "--max_rigid_shifts_dim1",
        type=int,
        default=15,
        help="Maximum allowed global rigid shift in pixels along image height.",
    )
    parser.add_argument(
        "--max_rigid_shifts_dim2",
        type=int,
        default=15,
        help="Maximum allowed global rigid shift in pixels along image width.",
    )
    parser.add_argument(
        "--max_deviation_rigid_dim1",
        type=int,
        default=2,
        help="Maximum per-block shift deviation from global rigid shift along height.",
    )
    parser.add_argument(
        "--max_deviation_rigid_dim2",
        type=int,
        default=2,
        help="Maximum per-block shift deviation from global rigid shift along width.",
    )
    parser.add_argument(
        "--frame_batch_size",
        type=int,
        default=20,
        help="Frames processed per batch (trade-off: speed vs memory usage).",
    )
    parser.add_argument(
        "--spatial_highpass_sigma",
        type=float,
        default=3.0,
        help="Sigma for spatial high-pass filter used for shift estimation only. Set <=0 to disable.",
    )
    parser.add_argument(
        "--input_max_frames",
        type=int,
        default=None,
        help="If set, restrict processing to first N frames (useful for dry runs).",
    )
    parser.add_argument(
        "--dcimg_first_4px_correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enable first_4px_correction when reading .dcimg.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used by masknmf motion correction.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MotionCorrectionConfig:
    if args.input_image_path is None or args.out_path is None:
        raise ValueError(
            "config_from_args requires --input_image_path and --out_path."
        )
    return MotionCorrectionConfig(
        input_image_path=args.input_image_path,
        out_path=args.out_path,
        out_tiff_path=args.out_tiff_path,
        export_tiff_stack=args.export_tiff_stack,
        num_blocks_dim1=args.num_blocks_dim1,
        num_blocks_dim2=args.num_blocks_dim2,
        overlaps_dim1=args.overlaps_dim1,
        overlaps_dim2=args.overlaps_dim2,
        max_rigid_shifts_dim1=args.max_rigid_shifts_dim1,
        max_rigid_shifts_dim2=args.max_rigid_shifts_dim2,
        max_deviation_rigid_dim1=args.max_deviation_rigid_dim1,
        max_deviation_rigid_dim2=args.max_deviation_rigid_dim2,
        frame_batch_size=args.frame_batch_size,
        spatial_highpass_sigma=args.spatial_highpass_sigma,
        input_max_frames=args.input_max_frames,
        dcimg_first_4px_correction=args.dcimg_first_4px_correction,
        device=args.device,
    )


def build_config_for_path(
    args: argparse.Namespace, input_path: Path, out_path: Path
) -> MotionCorrectionConfig:
    return MotionCorrectionConfig(
        input_image_path=str(input_path),
        out_path=str(out_path),
        out_tiff_path=args.out_tiff_path,
        export_tiff_stack=args.export_tiff_stack,
        num_blocks_dim1=args.num_blocks_dim1,
        num_blocks_dim2=args.num_blocks_dim2,
        overlaps_dim1=args.overlaps_dim1,
        overlaps_dim2=args.overlaps_dim2,
        max_rigid_shifts_dim1=args.max_rigid_shifts_dim1,
        max_rigid_shifts_dim2=args.max_rigid_shifts_dim2,
        max_deviation_rigid_dim1=args.max_deviation_rigid_dim1,
        max_deviation_rigid_dim2=args.max_deviation_rigid_dim2,
        frame_batch_size=args.frame_batch_size,
        spatial_highpass_sigma=args.spatial_highpass_sigma,
        input_max_frames=args.input_max_frames,
        dcimg_first_4px_correction=args.dcimg_first_4px_correction,
        device=args.device,
    )


def discover_input_paths(
    explicit_inputs: list[str],
    input_dir: str | None,
    glob_pattern: str,
    recursive: bool,
) -> list[Path]:
    files: list[Path] = []

    for item in explicit_inputs:
        path = Path(item).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input does not exist: {path}")
        files.append(path)

    if input_dir is not None:
        root = Path(input_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"input_dir does not exist: {root}")
        matched = root.rglob(glob_pattern) if recursive else root.glob(glob_pattern)
        files.extend([p.resolve() for p in matched if p.is_file()])

    unique_sorted = sorted(set(files))
    return unique_sorted


def output_path_for_batch(output_dir: Path, input_path: Path, out_suffix: str) -> Path:
    return (output_dir / f"{input_path.stem}{out_suffix}.h5").resolve()


def load_movie_loader(
    input_image_path: str | Path,
    dcimg_first_4px_correction: bool = False,
) -> masknmf.LazyFrameLoader:
    path = Path(input_image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return masknmf.TiffArray(str(path))
    if suffix == ".dcimg":
        return masknmf.DcimgArray(
            filename=str(path),
            first_4px_correction=dcimg_first_4px_correction,
        )

    raise ValueError(
        f"Unsupported input format: {path.suffix}. "
        "Expected .tif/.tiff/.dcimg."
    )


def build_highpass_filter(sigma: float, device: str):
    kernel = masknmf.motion_correction.spatial_filters.compute_highpass_filter_kernel(
        [sigma, sigma]
    ).to(device)
    relu = torch.nn.ReLU()

    def filter_function(frames: torch.Tensor) -> torch.Tensor:
        filtered = masknmf.motion_correction.spatial_filters.image_filter(frames, kernel)
        return relu(filtered)

    return filter_function


def export_tiff_stack(
    registered_movie: masknmf.RegistrationArray,
    out_tiff_path: str | Path,
    batch_size: int,
) -> Path:
    tiff_path = Path(out_tiff_path).resolve()
    tiff_path.parent.mkdir(parents=True, exist_ok=True)
    if tiff_path.exists():
        raise FileExistsError(f"TIFF output already exists: {tiff_path}")

    num_frames = registered_movie.shape[0]
    print(f"Exporting motion-corrected TIFF stack to: {tiff_path}")
    with tifffile.TiffWriter(str(tiff_path), bigtiff=True) as tif:
        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            subset = np.asarray(registered_movie[start:end], dtype=np.float32)
            tif.write(subset, contiguous=True)

    return tiff_path


def run_motion_correction(config: MotionCorrectionConfig) -> Path:
    data_loader = load_movie_loader(
        config.input_image_path,
        dcimg_first_4px_correction=config.dcimg_first_4px_correction,
    )
    if config.input_max_frames is not None:
        if config.input_max_frames <= 0:
            raise ValueError("input_max_frames must be > 0")
        data_loader = PrefixFrameLoader(data_loader, max_frames=config.input_max_frames)

    pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrector(
        num_blocks=(config.num_blocks_dim1, config.num_blocks_dim2),
        overlaps=(config.overlaps_dim1, config.overlaps_dim2),
        max_rigid_shifts=(config.max_rigid_shifts_dim1, config.max_rigid_shifts_dim2),
        max_deviation_rigid=(config.max_deviation_rigid_dim1, config.max_deviation_rigid_dim2),
        batch_size=config.frame_batch_size,
        device=config.device,
    )

    print(f"Input movie shape: {data_loader.shape}")
    print(f"Motion correction device: {pwrigid_strategy.device}")

    if config.spatial_highpass_sigma > 0:
        print(f"Using spatial high-pass for shift estimation (sigma={config.spatial_highpass_sigma})")
        filter_fn = build_highpass_filter(config.spatial_highpass_sigma, pwrigid_strategy.device)
        filtered_reference = masknmf.FilteredArray(
            raw_data_loader=data_loader,
            filter_function=filter_fn,
            batching=config.frame_batch_size,
            device=pwrigid_strategy.device,
        )
        pwrigid_strategy.compute_template(filtered_reference)
        moco_results = masknmf.RegistrationArray(
            reference_movie=filtered_reference,
            strategy=pwrigid_strategy,
            target_movie=data_loader,
        )
    else:
        print("Spatial high-pass disabled.")
        pwrigid_strategy.compute_template(data_loader)
        moco_results = masknmf.RegistrationArray(
            reference_movie=data_loader,
            strategy=pwrigid_strategy,
        )

    out_path = Path(config.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting motion-corrected movie to: {out_path}")
    moco_results.export(str(out_path))
    if config.export_tiff_stack:
        tiff_out_path = (
            Path(config.out_tiff_path).resolve()
            if config.out_tiff_path is not None
            else out_path.with_suffix(".tif")
        )
        export_tiff_stack(
            registered_movie=moco_results,
            out_tiff_path=tiff_out_path,
            batch_size=config.frame_batch_size,
        )
    print("Done.")
    return out_path


def main() -> None:
    args = parse_args()
    batch_inputs = discover_input_paths(
        explicit_inputs=args.inputs,
        input_dir=args.input_dir,
        glob_pattern=args.glob_pattern,
        recursive=args.recursive,
    )

    # Single-file mode (backward compatible)
    if len(batch_inputs) == 0:
        if args.input_image_path is None or args.out_path is None:
            raise ValueError(
                "Provide either single-file args (--input_image_path and --out_path) "
                "or batch args (--inputs and/or --input_dir)."
            )
        config = config_from_args(args)
        run_motion_correction(config)
        return

    # Batch mode
    if args.output_dir is None:
        raise ValueError("Batch mode requires --output_dir.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovered {len(batch_inputs)} input file(s).")
    failures: list[tuple[Path, str]] = []
    completed = 0

    for index, input_path in enumerate(batch_inputs, start=1):
        out_path = output_path_for_batch(output_dir, input_path, args.out_suffix)
        print(f"[{index}/{len(batch_inputs)}] {input_path.name} -> {out_path.name}")

        if out_path.exists():
            if args.skip_existing and not args.overwrite:
                print("  skipping (output exists)")
                continue
            if args.overwrite:
                out_path.unlink()

        if args.dry_run:
            print("  dry-run: not executed")
            continue

        config = build_config_for_path(args, input_path=input_path, out_path=out_path)
        try:
            run_motion_correction(config)
            completed += 1
        except Exception as exc:
            print(f"  failed: {exc}")
            failures.append((input_path, str(exc)))
            if not args.continue_on_error:
                break

    print(f"Completed {completed} file(s).")
    if failures:
        print(f"Failures: {len(failures)}")
        for path, message in failures:
            print(f"  - {path}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
