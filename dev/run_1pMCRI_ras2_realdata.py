"""
Real-data runner for 1pMCRI motion correction.

Target file:
    demo_data/250810-Ras2-GC#78.dcimg

Run:
    python dev/run_1pMCRI_ras2_realdata.py
"""

import importlib.util
from pathlib import Path


def load_1pmcri_module():
    module_path = Path(__file__).with_name("1pMCRI_motion_correct_data.py")
    spec = importlib.util.spec_from_file_location("mcri_motion_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_1pmcri_module()

    input_path = Path("demo_data/250810-Ras2-GC#78.dcimg").resolve()
    output_path = Path("demo_data/output/250810-Ras2-GC#78_moco_uint16.h5").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Rename/remove it before running."
        )

    config = module.MotionCorrectionConfig(
        input_image_path=str(input_path),
        out_path=str(output_path),
        export_tiff_stack=False,
        # Real-data run: process all frames
        input_max_frames=None,
        frame_batch_size=50,
        output_dtype="uint16",
        output_compression="none",
        device="auto",
    )

    print("Running 1pMCRI real-data script")
    print(f"  input : {input_path}")
    print(f"  output: {output_path}")
    print("  settings: full frames, batch_size=50, output_dtype=uint16, compression=none, no TIFF")

    module.run_motion_correction(config)
    print("Real-data run completed successfully.")


if __name__ == "__main__":
    main()
