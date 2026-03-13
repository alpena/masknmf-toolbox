import argparse
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tifffile
import torch
from tqdm import tqdm

import masknmf
from masknmf.motion_correction.registration_methods import compute_pwrigid_patch_midpoints


@dataclass(slots=True)
class MotionCorrectionConfig:
    """Parameters for piecewise-rigid motion correction and output export."""

    input_image_path: str
    out_path: str
    out_tiff_path: str | None = None
    export_tiff_stack: bool = True
    output_dtype: str = "uint16"
    output_compression: str = "none"
    export_extract_optimized_h5: bool = True
    extract_orientation_fix: str = "transpose_xy"
    extract_chunk_t: int = 1
    extract_chunk_x: int = 0
    extract_chunk_y: int = 0
    num_blocks_dim1: int = 20
    num_blocks_dim2: int = 20
    overlaps_dim1: int = 25
    overlaps_dim2: int = 25
    max_rigid_shifts_dim1: int = 15
    max_rigid_shifts_dim2: int = 15
    max_deviation_rigid_dim1: int = 10
    max_deviation_rigid_dim2: int = 10
    frame_batch_size: int = 50
    spatial_highpass_sigma: float = 3.0
    async_export: bool = True
    prefetch_batches: int = 4
    writer_queue_batches: int = 2
    pin_memory: bool = True
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
        "--output_dtype",
        default="uint16",
        choices=["uint16", "float32"],
        help="Data type of movie dataset in output HDF5 (/mov in standard mode, /motion_corrected in legacy mode).",
    )
    parser.add_argument(
        "--output_compression",
        default="none",
        choices=["none", "lzf", "gzip"],
        help="Compression used when writing HDF5 datasets.",
    )
    parser.add_argument(
        "--export_extract_optimized_h5",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write standard pipeline dataset '/mov'. Disable only for legacy '/motion_corrected' compatibility.",
    )
    parser.add_argument(
        "--extract_orientation_fix",
        default="transpose_xy",
        choices=["none", "transpose_xy"],
        help="Orientation fix used when writing EXTRACT-ready H5.",
    )
    parser.add_argument(
        "--extract_chunk_t",
        type=int,
        default=1,
        help="Chunk size in t for direct EXTRACT H5 export.",
    )
    parser.add_argument(
        "--extract_chunk_x",
        type=int,
        default=0,
        help="Chunk size in x (stored axis-2) for direct EXTRACT H5 export. Set <=0 for full axis.",
    )
    parser.add_argument(
        "--extract_chunk_y",
        type=int,
        default=0,
        help="Chunk size in y (stored axis-3) for direct EXTRACT H5 export. Set <=0 for full axis.",
    )
    parser.add_argument(
        "--num_blocks_dim1",
        type=int,
        default=20,
        help="Number of blocks along image height for piecewise-rigid registration.",
    )
    parser.add_argument(
        "--num_blocks_dim2",
        type=int,
        default=20,
        help="Number of blocks along image width for piecewise-rigid registration.",
    )
    parser.add_argument(
        "--overlaps_dim1",
        type=int,
        default=25,
        help="Block overlap size in pixels along image height.",
    )
    parser.add_argument(
        "--overlaps_dim2",
        type=int,
        default=25,
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
        default=10,
        help="Maximum per-block shift deviation from global rigid shift along height.",
    )
    parser.add_argument(
        "--max_deviation_rigid_dim2",
        type=int,
        default=10,
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
        "--async_export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use reader/writer overlap for EXTRACT H5 export when possible.",
    )
    parser.add_argument(
        "--prefetch_batches",
        type=int,
        default=4,
        help="Number of raw batches to keep queued in CPU memory ahead of the GPU worker.",
    )
    parser.add_argument(
        "--writer_queue_batches",
        type=int,
        default=2,
        help="Maximum number of processed batches waiting for HDF5 writer flush.",
    )
    parser.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pinned host buffers before CUDA transfer in async export mode.",
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
        output_dtype=args.output_dtype,
        output_compression=args.output_compression,
        export_extract_optimized_h5=args.export_extract_optimized_h5,
        extract_orientation_fix=args.extract_orientation_fix,
        extract_chunk_t=args.extract_chunk_t,
        extract_chunk_x=args.extract_chunk_x,
        extract_chunk_y=args.extract_chunk_y,
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
        async_export=args.async_export,
        prefetch_batches=args.prefetch_batches,
        writer_queue_batches=args.writer_queue_batches,
        pin_memory=args.pin_memory,
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
        output_dtype=args.output_dtype,
        output_compression=args.output_compression,
        export_extract_optimized_h5=args.export_extract_optimized_h5,
        extract_orientation_fix=args.extract_orientation_fix,
        extract_chunk_t=args.extract_chunk_t,
        extract_chunk_x=args.extract_chunk_x,
        extract_chunk_y=args.extract_chunk_y,
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
        async_export=args.async_export,
        prefetch_batches=args.prefetch_batches,
        writer_queue_batches=args.writer_queue_batches,
        pin_memory=args.pin_memory,
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


def _determine_shifts_shape(
    strategy: masknmf.MotionCorrectionStrategy,
    num_frames: int,
    frame_shape: tuple[int, int],
) -> tuple[int, ...] | None:
    if isinstance(strategy, masknmf.PiecewiseRigidMotionCorrector):
        block_centers = compute_pwrigid_patch_midpoints(
            num_blocks=strategy.num_blocks,
            overlaps=strategy.overlaps,
            fov_height=frame_shape[0],
            fov_width=frame_shape[1],
        )
        return (num_frames, block_centers.shape[0], block_centers.shape[1], 2)
    if isinstance(strategy, masknmf.RigidMotionCorrector):
        return (num_frames, 2)
    if isinstance(strategy, masknmf.DummyMotionCorrector):
        return None
    raise ValueError("Unsupported motion correction strategy")


def _compute_extract_layout(
    num_frames: int,
    fov_h: int,
    fov_w: int,
    motion_dtype: str,
    orientation_fix: str,
    chunk_t: int,
    chunk_x: int,
    chunk_y: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], np.dtype]:
    if orientation_fix not in {"none", "transpose_xy"}:
        raise ValueError(
            f"Unsupported extract_orientation_fix: {orientation_fix}. "
            "Use 'none' or 'transpose_xy'."
        )

    motion_np_dtype = np.uint16 if motion_dtype == "uint16" else np.float32
    if orientation_fix == "none":
        out_shape = (num_frames, fov_h, fov_w)
        matlab_view = (fov_w, fov_h, num_frames)
    else:
        out_shape = (num_frames, fov_w, fov_h)
        matlab_view = (fov_h, fov_w, num_frames)

    chunk_t_eff = max(1, min(int(chunk_t), num_frames))
    chunk_x_eff = out_shape[1] if int(chunk_x) <= 0 else max(1, min(int(chunk_x), out_shape[1]))
    chunk_y_eff = out_shape[2] if int(chunk_y) <= 0 else max(1, min(int(chunk_y), out_shape[2]))

    bytes_per_voxel = np.dtype(motion_np_dtype).itemsize
    max_chunk_bytes = 4 * 1024**3 - 1
    chunk_bytes = chunk_t_eff * chunk_x_eff * chunk_y_eff * bytes_per_voxel
    while chunk_bytes > max_chunk_bytes and chunk_t_eff > 1:
        chunk_t_eff = max(1, chunk_t_eff // 2)
        chunk_bytes = chunk_t_eff * chunk_x_eff * chunk_y_eff * bytes_per_voxel
    while chunk_bytes > max_chunk_bytes and (chunk_x_eff > 1 or chunk_y_eff > 1):
        if chunk_x_eff >= chunk_y_eff and chunk_x_eff > 1:
            chunk_x_eff = max(1, chunk_x_eff // 2)
        elif chunk_y_eff > 1:
            chunk_y_eff = max(1, chunk_y_eff // 2)
        chunk_bytes = chunk_t_eff * chunk_x_eff * chunk_y_eff * bytes_per_voxel

    print(f"  dataset: /mov")
    print(f"  orientation_fix: {orientation_fix}")
    print(
        f"  stored shape (t,x,y): ({out_shape[0]}, {out_shape[1]}, {out_shape[2]}) "
        f"| MATLAB view (h,w,t): {matlab_view}"
    )
    print(
        f"  chunks (t,x,y): ({chunk_t_eff}, {chunk_x_eff}, {chunk_y_eff}) "
        f"[{chunk_bytes / (1024**2):.1f} MiB]"
    )
    return out_shape, matlab_view, (chunk_t_eff, chunk_x_eff, chunk_y_eff), motion_np_dtype


def _load_frame_block(
    loader,
    start: int,
    end: int,
    dcimg_file=None,
) -> np.ndarray:
    indices = slice(start, end)
    if dcimg_file is not None and isinstance(loader, masknmf.DcimgArray):
        return loader._read_from_handle(dcimg_file, indices)
    frames = loader[indices]
    if frames.ndim == 2:
        frames = frames[None, :, :]
    return np.asarray(frames, dtype=np.float32)


def _prepare_frames_for_device(
    frames: np.ndarray,
    device: str,
    pin_memory: bool,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(frames, dtype=np.float32, order="C"))
    use_cuda = str(device).startswith("cuda")
    if use_cuda and pin_memory:
        tensor = tensor.pin_memory()
    return tensor.to(device, dtype=torch.float32, non_blocking=(use_cuda and pin_memory))


def _format_extract_batch(
    corrected_frames: np.ndarray,
    shifts: np.ndarray | None,
    motion_dtype: str,
    orientation_fix: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    if motion_dtype == "uint16":
        to_store = np.clip(np.rint(corrected_frames), 0, 65535).astype(np.uint16)
    else:
        to_store = corrected_frames.astype(np.float32, copy=False)

    if orientation_fix == "transpose_xy":
        to_store = to_store.transpose(0, 2, 1)

    shifts_out = None if shifts is None else np.asarray(shifts, dtype=np.float32)
    return to_store, shifts_out


def export_tiff_stack(
    registered_movie: masknmf.RegistrationArray,
    out_tiff_path: str | Path,
    batch_size: int,
    output_dtype: str = "uint16",
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
            if output_dtype == "uint16":
                subset = np.clip(np.rint(subset), 0, 65535).astype(np.uint16)
            else:
                subset = subset.astype(np.float32)
            tif.write(subset, contiguous=True)

    return tiff_path


def export_standard_h5_streaming(
    registered_movie: masknmf.RegistrationArray,
    out_path: str | Path,
    batch_size: int,
    motion_dtype: str = "uint16",
    compression: str = "lzf",
) -> Path:
    output_path = Path(out_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"HDF5 output already exists: {output_path}")

    num_frames, fov_h, fov_w = registered_movie.shape
    compression_value = None if compression == "none" else compression

    if isinstance(registered_movie.strategy, masknmf.PiecewiseRigidMotionCorrector):
        shifts_shape = (
            num_frames,
            registered_movie.block_centers.shape[0],
            registered_movie.block_centers.shape[1],
            2,
        )
    elif isinstance(registered_movie.strategy, masknmf.RigidMotionCorrector):
        shifts_shape = (num_frames, 2)
    elif isinstance(registered_movie.strategy, masknmf.DummyMotionCorrector):
        shifts_shape = None
    else:
        raise ValueError("Unsupported motion correction strategy")

    motion_np_dtype = np.uint16 if motion_dtype == "uint16" else np.float32

    print(f"Exporting motion-corrected movie to: {output_path}")
    with h5py.File(str(output_path), "w") as h5f:
        motion_dset = h5f.create_dataset(
            "motion_corrected",
            shape=(num_frames, fov_h, fov_w),
            dtype=motion_np_dtype,
            chunks=(min(batch_size, num_frames), fov_h, fov_w),
            compression=compression_value,
        )
        f_per_pixel_dset = h5f.create_dataset(
            "F_per_pixel",
            shape=(fov_h, fov_w),
            dtype=np.float32,
            compression=compression_value,
        )
        f_per_pixel_dset.attrs["producer"] = "masknmf_motion_correct_data"
        f_per_pixel_dset.attrs["source_dataset"] = "/motion_corrected"
        f_per_pixel_dset.attrs["frame_begin"] = np.int64(1)
        f_per_pixel_dset.attrs["frame_end"] = np.int64(num_frames)
        f_per_pixel_dset.attrs["orientation_fix"] = "none"
        if shifts_shape is not None:
            shifts_dset = h5f.create_dataset(
                "shifts",
                shape=shifts_shape,
                dtype=np.float32,
                compression=compression_value,
            )
        else:
            shifts_dset = None

        sum_image = np.zeros((fov_h, fov_w), dtype=np.float64)
        starts = list(range(0, num_frames, batch_size))
        for start in tqdm(starts, desc="Exporting H5 batches", unit="batch"):
            end = min(start + batch_size, num_frames)
            moco_subset, shifts_subset = registered_movie._index_frames_tensor(slice(start, end))
            moco_subset = np.asarray(moco_subset, dtype=np.float32)

            if motion_dtype == "uint16":
                # Keep toolbox dataset format while reducing storage and memory pressure.
                to_store = np.clip(np.rint(moco_subset), 0, 65535).astype(np.uint16)
            else:
                to_store = moco_subset.astype(np.float32)

            motion_dset[start:end, :, :] = to_store
            sum_image += np.asarray(to_store, dtype=np.float32).sum(axis=0, dtype=np.float64)
            if shifts_dset is not None:
                shifts_dset[start:end, ...] = np.asarray(shifts_subset, dtype=np.float32)

        f_per_pixel_dset[:, :] = (sum_image / float(num_frames)).astype(np.float32)

    return output_path


def export_extract_h5_streaming(
    registered_movie: masknmf.RegistrationArray,
    out_path: str | Path,
    batch_size: int,
    motion_dtype: str = "uint16",
    compression: str = "lzf",
    orientation_fix: str = "transpose_xy",
    chunk_t: int = 1,
    chunk_x: int = 0,
    chunk_y: int = 0,
) -> Path:
    output_path = Path(out_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"HDF5 output already exists: {output_path}")
    if orientation_fix not in {"none", "transpose_xy"}:
        raise ValueError(
            f"Unsupported extract_orientation_fix: {orientation_fix}. "
            "Use 'none' or 'transpose_xy'."
        )

    num_frames, fov_h, fov_w = registered_movie.shape
    compression_value = None if compression == "none" else compression

    # Keep shifts layout unchanged from legacy exporter.
    if isinstance(registered_movie.strategy, masknmf.PiecewiseRigidMotionCorrector):
        shifts_shape = (
            num_frames,
            registered_movie.block_centers.shape[0],
            registered_movie.block_centers.shape[1],
            2,
        )
    elif isinstance(registered_movie.strategy, masknmf.RigidMotionCorrector):
        shifts_shape = (num_frames, 2)
    elif isinstance(registered_movie.strategy, masknmf.DummyMotionCorrector):
        shifts_shape = None
    else:
        raise ValueError("Unsupported motion correction strategy")

    motion_np_dtype = np.uint16 if motion_dtype == "uint16" else np.float32

    if orientation_fix == "none":
        # Stored shape (t,h,w) -> MATLAB view (w,h,t)
        out_shape = (num_frames, fov_h, fov_w)
        matlab_view = (fov_w, fov_h, num_frames)
    else:
        # Stored shape (t,w,h) -> MATLAB view (h,w,t)
        out_shape = (num_frames, fov_w, fov_h)
        matlab_view = (fov_h, fov_w, num_frames)

    chunk_t_eff = max(1, min(int(chunk_t), num_frames))
    chunk_x_eff = out_shape[1] if int(chunk_x) <= 0 else max(1, min(int(chunk_x), out_shape[1]))
    chunk_y_eff = out_shape[2] if int(chunk_y) <= 0 else max(1, min(int(chunk_y), out_shape[2]))

    bytes_per_voxel = np.dtype(motion_np_dtype).itemsize
    max_chunk_bytes = 4 * 1024**3 - 1
    chunk_bytes = chunk_t_eff * chunk_x_eff * chunk_y_eff * bytes_per_voxel
    while chunk_bytes > max_chunk_bytes and chunk_t_eff > 1:
        chunk_t_eff = max(1, chunk_t_eff // 2)
        chunk_bytes = chunk_t_eff * chunk_x_eff * chunk_y_eff * bytes_per_voxel
    while chunk_bytes > max_chunk_bytes and (chunk_x_eff > 1 or chunk_y_eff > 1):
        if chunk_x_eff >= chunk_y_eff and chunk_x_eff > 1:
            chunk_x_eff = max(1, chunk_x_eff // 2)
        elif chunk_y_eff > 1:
            chunk_y_eff = max(1, chunk_y_eff // 2)
        chunk_bytes = chunk_t_eff * chunk_x_eff * chunk_y_eff * bytes_per_voxel

    print(f"Exporting EXTRACT-optimized movie to: {output_path}")
    print(f"  dataset: /mov")
    print(f"  orientation_fix: {orientation_fix}")
    print(
        f"  stored shape (t,x,y): ({out_shape[0]}, {out_shape[1]}, {out_shape[2]}) "
        f"| MATLAB view (h,w,t): {matlab_view}"
    )
    print(
        f"  chunks (t,x,y): ({chunk_t_eff}, {chunk_x_eff}, {chunk_y_eff}) "
        f"[{chunk_bytes / (1024**2):.1f} MiB]"
    )

    with h5py.File(str(output_path), "w") as h5f:
        mov_dset = h5f.create_dataset(
            "mov",
            shape=out_shape,
            dtype=motion_np_dtype,
            chunks=(chunk_t_eff, chunk_x_eff, chunk_y_eff),
            compression=compression_value,
        )
        f_per_pixel_shape = (out_shape[1], out_shape[2])
        f_per_pixel_dset = h5f.create_dataset(
            "F_per_pixel",
            shape=f_per_pixel_shape,
            dtype=np.float32,
            compression=compression_value,
        )
        f_per_pixel_dset.attrs["producer"] = "masknmf_motion_correct_data"
        f_per_pixel_dset.attrs["source_dataset"] = "/mov"
        f_per_pixel_dset.attrs["frame_begin"] = np.int64(1)
        f_per_pixel_dset.attrs["frame_end"] = np.int64(num_frames)
        f_per_pixel_dset.attrs["orientation_fix"] = orientation_fix
        if shifts_shape is not None:
            shifts_dset = h5f.create_dataset(
                "shifts",
                shape=shifts_shape,
                dtype=np.float32,
                compression=compression_value,
            )
        else:
            shifts_dset = None

        sum_image = np.zeros(f_per_pixel_shape, dtype=np.float64)
        starts = list(range(0, num_frames, batch_size))
        for start in tqdm(starts, desc="Exporting EXTRACT H5 batches", unit="batch"):
            end = min(start + batch_size, num_frames)
            moco_subset, shifts_subset = registered_movie._index_frames_tensor(slice(start, end))
            moco_subset = np.asarray(moco_subset, dtype=np.float32)

            if motion_dtype == "uint16":
                to_store = np.clip(np.rint(moco_subset), 0, 65535).astype(np.uint16)
            else:
                to_store = moco_subset.astype(np.float32)

            if orientation_fix == "transpose_xy":
                to_store_oriented = to_store.transpose(0, 2, 1)
            else:
                to_store_oriented = to_store

            mov_dset[start:end, :, :] = to_store_oriented
            sum_image += np.asarray(to_store_oriented, dtype=np.float32).sum(axis=0, dtype=np.float64)

            if shifts_dset is not None:
                shifts_dset[start:end, ...] = np.asarray(shifts_subset, dtype=np.float32)

        f_per_pixel_dset[:, :] = (sum_image / float(num_frames)).astype(np.float32)

    return output_path


def export_extract_h5_streaming_async(
    data_loader,
    strategy: masknmf.MotionCorrectionStrategy,
    out_path: str | Path,
    batch_size: int,
    motion_dtype: str = "uint16",
    compression: str = "lzf",
    orientation_fix: str = "transpose_xy",
    chunk_t: int = 1,
    chunk_x: int = 0,
    chunk_y: int = 0,
    filter_function=None,
    prefetch_batches: int = 4,
    writer_queue_batches: int = 2,
    pin_memory: bool = True,
) -> Path:
    output_path = Path(out_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"HDF5 output already exists: {output_path}")
    if prefetch_batches < 1:
        raise ValueError("prefetch_batches must be >= 1")
    if writer_queue_batches < 1:
        raise ValueError("writer_queue_batches must be >= 1")

    num_frames, fov_h, fov_w = data_loader.shape
    compression_value = None if compression == "none" else compression
    shifts_shape = _determine_shifts_shape(strategy, num_frames, (fov_h, fov_w))
    print(f"Exporting EXTRACT-optimized movie to: {output_path}")
    out_shape, _, chunk_shape, motion_np_dtype = _compute_extract_layout(
        num_frames=num_frames,
        fov_h=fov_h,
        fov_w=fov_w,
        motion_dtype=motion_dtype,
        orientation_fix=orientation_fix,
        chunk_t=chunk_t,
        chunk_x=chunk_x,
        chunk_y=chunk_y,
    )
    print(
        f"  async pipeline: enabled (prefetch_batches={prefetch_batches}, "
        f"writer_queue_batches={writer_queue_batches}, pin_memory={pin_memory})"
    )

    sentinel = object()
    input_queue: queue.Queue = queue.Queue(maxsize=prefetch_batches)
    output_queue: queue.Queue = queue.Queue(maxsize=writer_queue_batches)
    worker_error: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def raise_worker_error() -> None:
        try:
            exc = worker_error.get_nowait()
        except queue.Empty:
            return
        stop_event.set()
        raise exc

    def put_with_checks(q: queue.Queue, item) -> None:
        while True:
            raise_worker_error()
            if stop_event.is_set():
                raise RuntimeError("async export aborted")
            try:
                q.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def get_with_checks(q: queue.Queue):
        while True:
            raise_worker_error()
            if stop_event.is_set():
                raise RuntimeError("async export aborted")
            try:
                return q.get(timeout=0.1)
            except queue.Empty:
                continue

    def reader_worker() -> None:
        try:
            if isinstance(data_loader, masknmf.DcimgArray):
                with data_loader._open_dcimg() as dcimg_file:
                    for start in range(0, num_frames, batch_size):
                        if stop_event.is_set():
                            break
                        end = min(start + batch_size, num_frames)
                        raw_frames = _load_frame_block(data_loader, start, end, dcimg_file=dcimg_file)
                        put_with_checks(input_queue, (start, end, raw_frames))
            else:
                for start in range(0, num_frames, batch_size):
                    if stop_event.is_set():
                        break
                    end = min(start + batch_size, num_frames)
                    raw_frames = _load_frame_block(data_loader, start, end)
                    put_with_checks(input_queue, (start, end, raw_frames))
        except Exception as exc:
            worker_error.put(exc)
            stop_event.set()
        finally:
            while True:
                try:
                    input_queue.put(sentinel, timeout=0.1)
                    break
                except queue.Full:
                    if stop_event.is_set():
                        break
                    continue

    def writer_worker() -> None:
        sum_image = np.zeros((out_shape[1], out_shape[2]), dtype=np.float64)
        try:
            with h5py.File(str(output_path), "w") as h5f:
                mov_dset = h5f.create_dataset(
                    "mov",
                    shape=out_shape,
                    dtype=motion_np_dtype,
                    chunks=chunk_shape,
                    compression=compression_value,
                )
                f_per_pixel_dset = h5f.create_dataset(
                    "F_per_pixel",
                    shape=(out_shape[1], out_shape[2]),
                    dtype=np.float32,
                    compression=compression_value,
                )
                f_per_pixel_dset.attrs["producer"] = "masknmf_motion_correct_data"
                f_per_pixel_dset.attrs["source_dataset"] = "/mov"
                f_per_pixel_dset.attrs["frame_begin"] = np.int64(1)
                f_per_pixel_dset.attrs["frame_end"] = np.int64(num_frames)
                f_per_pixel_dset.attrs["orientation_fix"] = orientation_fix
                if shifts_shape is not None:
                    shifts_dset = h5f.create_dataset(
                        "shifts",
                        shape=shifts_shape,
                        dtype=np.float32,
                        compression=compression_value,
                    )
                else:
                    shifts_dset = None

                while True:
                    item = output_queue.get()
                    if item is sentinel:
                        break
                    start, end, to_store_oriented, shifts_subset = item
                    mov_dset[start:end, :, :] = to_store_oriented
                    sum_image += np.asarray(
                        to_store_oriented, dtype=np.float32
                    ).sum(axis=0, dtype=np.float64)
                    if shifts_dset is not None and shifts_subset is not None:
                        shifts_dset[start:end, ...] = shifts_subset

                f_per_pixel_dset[:, :] = (sum_image / float(num_frames)).astype(np.float32)
        except Exception as exc:
            worker_error.put(exc)
            stop_event.set()

    reader_thread = threading.Thread(target=reader_worker, name="masknmf-reader")
    writer_thread = threading.Thread(target=writer_worker, name="masknmf-writer")
    reader_thread.start()
    writer_thread.start()

    try:
        num_batches = (num_frames + batch_size - 1) // batch_size
        with tqdm(total=num_batches, desc="Exporting EXTRACT H5 batches", unit="batch") as progress:
            while True:
                item = get_with_checks(input_queue)
                if item is sentinel:
                    break
                start, end, raw_frames = item
                raw_tensor = _prepare_frames_for_device(raw_frames, strategy.device, pin_memory)
                try:
                    reference_tensor = (
                        filter_function(raw_tensor) if filter_function is not None else raw_tensor
                    )
                    corrected_tensor, shifts_tensor = strategy.correct_tensors(
                        reference_movie_frames=reference_tensor,
                        target_movie_frames=raw_tensor,
                    )
                    corrected_np = corrected_tensor.detach().cpu().numpy()
                    shifts_np = None if shifts_shape is None else shifts_tensor.detach().cpu().numpy()
                finally:
                    del raw_tensor
                    if filter_function is not None:
                        del reference_tensor

                to_store_oriented, shifts_subset = _format_extract_batch(
                    corrected_frames=corrected_np,
                    shifts=shifts_np,
                    motion_dtype=motion_dtype,
                    orientation_fix=orientation_fix,
                )
                put_with_checks(output_queue, (start, end, to_store_oriented, shifts_subset))
                progress.update(1)
    finally:
        stop_event.set()
        while True:
            try:
                output_queue.put(sentinel, timeout=0.1)
                break
            except queue.Full:
                if not writer_thread.is_alive():
                    break
                continue
        reader_thread.join()
        writer_thread.join()

    if reader_thread.is_alive() or writer_thread.is_alive():
        raise RuntimeError("Async EXTRACT export threads did not terminate cleanly.")

    raise_worker_error()
    if not output_path.exists():
        raise RuntimeError(f"Async EXTRACT export did not produce output: {output_path}")
    return output_path


def run_motion_correction(config: MotionCorrectionConfig) -> Path:
    out_path = Path(config.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"HDF5 output already exists: {out_path}")

    if config.export_tiff_stack:
        tiff_out_path = (
            Path(config.out_tiff_path).resolve()
            if config.out_tiff_path is not None
            else out_path.with_suffix(".tif")
        )
        if tiff_out_path.exists():
            raise FileExistsError(f"TIFF output already exists: {tiff_out_path}")

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

    filter_fn = None
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
    else:
        print("Spatial high-pass disabled.")
        pwrigid_strategy.compute_template(data_loader)

    if (
        config.export_extract_optimized_h5
        and config.async_export
        and not config.export_tiff_stack
    ):
        export_extract_h5_streaming_async(
            data_loader=data_loader,
            strategy=pwrigid_strategy,
            out_path=out_path,
            batch_size=config.frame_batch_size,
            motion_dtype=config.output_dtype,
            compression=config.output_compression,
            orientation_fix=config.extract_orientation_fix,
            chunk_t=config.extract_chunk_t,
            chunk_x=config.extract_chunk_x,
            chunk_y=config.extract_chunk_y,
            filter_function=filter_fn,
            prefetch_batches=config.prefetch_batches,
            writer_queue_batches=config.writer_queue_batches,
            pin_memory=config.pin_memory,
        )
        print("Done.")
        return out_path

    if config.export_extract_optimized_h5 and config.async_export and config.export_tiff_stack:
        print("Async EXTRACT export disabled because TIFF export also requested; using legacy synchronous path.")

    if config.spatial_highpass_sigma > 0:
        moco_results = masknmf.RegistrationArray(
            reference_movie=filtered_reference,
            strategy=pwrigid_strategy,
            target_movie=data_loader,
        )
    else:
        moco_results = masknmf.RegistrationArray(
            reference_movie=data_loader,
            strategy=pwrigid_strategy,
        )

    if config.export_extract_optimized_h5:
        export_extract_h5_streaming(
            registered_movie=moco_results,
            out_path=out_path,
            batch_size=config.frame_batch_size,
            motion_dtype=config.output_dtype,
            compression=config.output_compression,
            orientation_fix=config.extract_orientation_fix,
            chunk_t=config.extract_chunk_t,
            chunk_x=config.extract_chunk_x,
            chunk_y=config.extract_chunk_y,
        )
    else:
        export_standard_h5_streaming(
            registered_movie=moco_results,
            out_path=out_path,
            batch_size=config.frame_batch_size,
            motion_dtype=config.output_dtype,
            compression=config.output_compression,
        )
    if config.export_tiff_stack:
        export_tiff_stack(
            registered_movie=moco_results,
            out_tiff_path=tiff_out_path,
            batch_size=config.frame_batch_size,
            output_dtype=config.output_dtype,
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
