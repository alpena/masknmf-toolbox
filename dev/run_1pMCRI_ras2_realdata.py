"""
Real-data runner for 1pMCRI motion correction.

Target file:
    demo_data/250810-Ras2-GC#78.dcimg

Run:
    python dev/run_1pMCRI_ras2_realdata.py
"""

import importlib.util
import os
import shutil
from pathlib import Path


def load_1pmcri_module():
    module_path = Path(__file__).with_name("1pMCRI_motion_correct_data.py")
    spec = importlib.util.spec_from_file_location("mcri_motion_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick_fast_output_root() -> Path:
    """
    Pick fast local drive when available.
    - Windows: E:\
    - Linux/Ubuntu: /mnt/nvme
    Fallback: demo_data/output
    """
    if os.name == "nt":
        win_fast = Path("E:/")
        if win_fast.exists():
            return win_fast / "masknmf-output"
    else:
        linux_fast = Path("/mnt/nvme")
        if linux_fast.exists():
            return linux_fast / "masknmf-output"
    return Path("demo_data/output").resolve()


def main() -> None:
    module = load_1pmcri_module()

    # User options
    export_extract_optimized_h5 = True  # True: write '/mov' directly, False: legacy '/motion_corrected'
    extract_orientation_fix = "transpose_xy"  # 'transpose_xy' or 'none'

    input_path = Path("demo_data/250810-Ras2-GC#78.dcimg").resolve()
    output_root = pick_fast_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = (output_root / "250810-Ras2-GC#78_moco_first100_smoke_direct.h5").resolve()
    final_output_dir = Path("demo_data/output").resolve()
    final_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = (final_output_dir / output_path.name).resolve()
    temp_tiff_path = output_path.with_suffix(".tif")
    final_tiff_path = final_output_path.with_suffix(".tif")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Rename/remove it before running."
        )
    if final_output_path.exists():
        raise FileExistsError(
            f"Final output already exists: {final_output_path}. "
            "Rename/remove it before running."
        )
    if temp_tiff_path.exists():
        raise FileExistsError(
            f"Temp TIFF already exists: {temp_tiff_path}. "
            "Rename/remove it before running."
        )
    if final_tiff_path.exists():
        raise FileExistsError(
            f"Final TIFF already exists: {final_tiff_path}. "
            "Rename/remove it before running."
        )

    config = module.MotionCorrectionConfig(
        input_image_path=str(input_path),
        out_path=str(output_path),
        export_tiff_stack=False,
        # Real-data run: process all frames
        input_max_frames=100,
        frame_batch_size=50,
        output_dtype="uint16",
        output_compression="none",
        export_extract_optimized_h5=export_extract_optimized_h5,
        extract_orientation_fix=extract_orientation_fix,
        extract_chunk_t=256,
        extract_chunk_x=256,
        extract_chunk_y=256,
        device="auto",
    )

    print("Running 1pMCRI real-data script")
    print(f"  input : {input_path}")
    print(f"  temp output : {output_path}")
    print(f"  final output: {final_output_path}")
    frames_text = "full frames" if config.input_max_frames is None else f"first {config.input_max_frames} frames"
    tiff_text = "TIFF export" if config.export_tiff_stack else "no TIFF"
    output_mode = "direct-extract (/mov)" if config.export_extract_optimized_h5 else "legacy (/motion_corrected)"
    print(
        "  settings: "
        f"{frames_text}, "
        f"batch_size={config.frame_batch_size}, "
        f"output_dtype={config.output_dtype}, "
        f"compression={config.output_compression}, "
        f"{tiff_text}, "
        f"output_mode={output_mode}, "
        f"extract_orientation_fix={config.extract_orientation_fix}"
    )

    module.run_motion_correction(config)
    if output_path != final_output_path:
        print(f"Moving output to final location: {final_output_path}")
        shutil.move(str(output_path), str(final_output_path))
    if config.export_tiff_stack and temp_tiff_path.exists() and temp_tiff_path != final_tiff_path:
        print(f"Moving TIFF to final location: {final_tiff_path}")
        shutil.move(str(temp_tiff_path), str(final_tiff_path))
    print("Real-data run completed successfully.")


if __name__ == "__main__":
    main()
