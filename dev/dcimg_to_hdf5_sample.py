import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

import masknmf


DEFAULT_INPUT = Path("demo_data/250206-UK6-1-F=4_power=5mW_00001.dcimg")
DEFAULT_OUTPUT = Path("demo_data/output/250206-UK6-1-F=4_power=5mW_00001_first500.h5")


def parse_max_frames(value: str) -> int | None:
    if value.lower() in {"none", "all"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "max_frames must be a positive integer, or 'none'/'all'."
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a DCIMG file to toolbox-compatible HDF5 using DcimgArray."
    )
    parser.add_argument(
        "--input_dcimg_path",
        default=str(DEFAULT_INPUT),
        help="Path to input .dcimg file.",
    )
    parser.add_argument(
        "--out_h5_path",
        default=str(DEFAULT_OUTPUT),
        help="Path to output HDF5 file.",
    )
    parser.add_argument(
        "--max_frames",
        type=parse_max_frames,
        default=500,
        help="Maximum number of frames from the beginning; use 'none' to export all frames.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Number of frames to read and write per batch.",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "lzf", "gzip"],
        default="lzf",
        help="HDF5 compression mode.",
    )
    parser.add_argument(
        "--first_4px_correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to enable first_4px_correction in the dcimg reader.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to overwrite an existing output file.",
    )
    parser.add_argument(
        "--row_start",
        type=int,
        default=None,
        help="Crop start row (height axis, inclusive). Default: 0.",
    )
    parser.add_argument(
        "--row_end",
        type=int,
        default=None,
        help="Crop end row (height axis, exclusive). Default: full height.",
    )
    parser.add_argument(
        "--col_start",
        type=int,
        default=None,
        help="Crop start column (width axis, inclusive). Default: 0.",
    )
    parser.add_argument(
        "--col_end",
        type=int,
        default=None,
        help="Crop end column (width axis, exclusive). Default: full width.",
    )
    return parser.parse_args()


def convert_dcimg_to_hdf5(
    input_dcimg_path: str | Path,
    out_h5_path: str | Path,
    max_frames: int | None = 500,
    batch_size: int = 200,
    compression: str = "lzf",
    first_4px_correction: bool = False,
    overwrite: bool = False,
    row_start: int | None = None,
    row_end: int | None = None,
    col_start: int | None = None,
    col_end: int | None = None,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    input_path = Path(input_dcimg_path).resolve()
    out_path = Path(out_h5_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dcimg not found: {input_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {out_path}. Use --overwrite to replace it."
        )

    arr = masknmf.DcimgArray(
        filename=input_path,
        first_4px_correction=first_4px_correction,
    )

    num_frames_total, height, width = arr.shape
    num_frames = num_frames_total if max_frames is None else min(num_frames_total, max_frames)

    rs = row_start if row_start is not None else 0
    re = row_end   if row_end   is not None else height
    cs = col_start if col_start is not None else 0
    ce = col_end   if col_end   is not None else width

    if not (0 <= rs < re <= height):
        raise ValueError(
            f"Invalid row crop [{rs}, {re}) for source height {height}."
        )
    if not (0 <= cs < ce <= width):
        raise ValueError(
            f"Invalid col crop [{cs}, {ce}) for source width {width}."
        )

    out_height = re - rs
    out_width  = ce - cs

    compression_value = None if compression == "none" else compression
    start_time = time.perf_counter()

    print(f"Input: {input_path}")
    print(f"Input shape: {arr.shape}")
    if max_frames is None:
        print(f"Exporting all {num_frames} frames")
    else:
        print(f"Exporting first {num_frames} frames")
    if (rs, re, cs, ce) != (0, height, 0, width):
        print(f"Spatial crop: rows [{rs}:{re}], cols [{cs}:{ce}] → ({out_height}, {out_width})")
    print(f"Output: {out_path}")

    mode = "w" if overwrite else "w-"
    with h5py.File(str(out_path), mode) as h5f:
        dset = h5f.create_dataset(
            "motion_corrected",
            shape=(num_frames, out_height, out_width),
            dtype=np.float32,
            chunks=(min(batch_size, num_frames), out_height, out_width),
            compression=compression_value,
        )

        for start in range(0, num_frames, batch_size):
            end = min(start + batch_size, num_frames)
            dset[start:end, :, :] = np.asarray(arr[start:end], dtype=np.float32)[:, rs:re, cs:ce]
            print(f"  wrote frames {start}:{end} / {num_frames}")

        dset.attrs["source_path"] = str(input_path)
        dset.attrs["source_format"] = "dcimg"
        dset.attrs["frames"] = int(num_frames)
        dset.attrs["frames_total_in_source"] = int(num_frames_total)
        dset.attrs["height"] = int(out_height)
        dset.attrs["width"] = int(out_width)
        dset.attrs["source_height"] = int(height)
        dset.attrs["source_width"] = int(width)
        dset.attrs["crop_row"] = [int(rs), int(re)]
        dset.attrs["crop_col"] = [int(cs), int(ce)]
        dset.attrs["dtype"] = str(np.float32)
        dset.attrs["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
        dset.attrs["processing_seconds"] = float(time.perf_counter() - start_time)

    return out_path


def main() -> None:
    args = parse_args()
    output_path = convert_dcimg_to_hdf5(
        input_dcimg_path=args.input_dcimg_path,
        out_h5_path=args.out_h5_path,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        compression=args.compression,
        first_4px_correction=args.first_4px_correction,
        overwrite=args.overwrite,
        row_start=args.row_start,
        row_end=args.row_end,
        col_start=args.col_start,
        col_end=args.col_end,
    )
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
