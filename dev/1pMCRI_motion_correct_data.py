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
    frame_batch_size: int = 500
    spatial_highpass_sigma: float = 3.0
    device: str = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run piecewise-rigid motion correction for 1p imaging from a TIFF path."
    )
    parser.add_argument("--input_image_path", required=True, help="Path to input TIFF movie.")
    parser.add_argument("--out_path", required=True, help="Path to output HDF5 file.")
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
        default=500,
        help="Frames processed per batch (trade-off: speed vs memory usage).",
    )
    parser.add_argument(
        "--spatial_highpass_sigma",
        type=float,
        default=3.0,
        help="Sigma for spatial high-pass filter used for shift estimation only. Set <=0 to disable.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used by masknmf motion correction.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> MotionCorrectionConfig:
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
        device=args.device,
    )


def load_tiff_movie(input_image_path: str | Path) -> np.ndarray:
    path = Path(input_image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")

    movie = tifffile.imread(path)
    if movie.ndim == 2:
        movie = movie[None, :, :]
    if movie.ndim != 3:
        raise ValueError(f"Expected a 2D/3D TIFF movie, got shape {movie.shape}")

    return np.asarray(movie)


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
    data = load_tiff_movie(config.input_image_path)

    pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrector(
        num_blocks=(config.num_blocks_dim1, config.num_blocks_dim2),
        overlaps=(config.overlaps_dim1, config.overlaps_dim2),
        max_rigid_shifts=(config.max_rigid_shifts_dim1, config.max_rigid_shifts_dim2),
        max_deviation_rigid=(config.max_deviation_rigid_dim1, config.max_deviation_rigid_dim2),
        batch_size=config.frame_batch_size,
        device=config.device,
    )

    print(f"Input movie shape: {data.shape}")
    print(f"Motion correction device: {pwrigid_strategy.device}")

    if config.spatial_highpass_sigma > 0:
        print(f"Using spatial high-pass for shift estimation (sigma={config.spatial_highpass_sigma})")
        filter_fn = build_highpass_filter(config.spatial_highpass_sigma, pwrigid_strategy.device)
        filtered_reference = masknmf.FilteredArray(
            raw_data_loader=data,
            filter_function=filter_fn,
            batching=config.frame_batch_size,
            device=pwrigid_strategy.device,
        )
        pwrigid_strategy.compute_template(filtered_reference)
        moco_results = masknmf.RegistrationArray(
            reference_movie=filtered_reference,
            strategy=pwrigid_strategy,
            target_movie=data,
        )
    else:
        print("Spatial high-pass disabled.")
        pwrigid_strategy.compute_template(data)
        moco_results = masknmf.RegistrationArray(reference_movie=data, strategy=pwrigid_strategy)

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
    config = config_from_args(args)
    run_motion_correction(config)


if __name__ == "__main__":
    main()
