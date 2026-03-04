"""
Minimal runnable example for 1pMCRI motion correction.

Run:
    python dev/run_1pMCRI_example.py
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

    input_path = Path("demo_data/250206-UK6-1-F=4_power=5mW_00001.dcimg").resolve()
    output_path = Path("demo_data/output/example_moco_from_dcimg.h5").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    config = module.MotionCorrectionConfig(
        input_image_path=str(input_path),
        out_path=str(output_path),
        input_max_frames=600,
        frame_batch_size=50,
        device="auto",
    )

    print("Running 1pMCRI example")
    print(f"  input : {input_path}")
    print(f"  output: {output_path}")
    print("  settings: input_max_frames=200, frame_batch_size=50, export_tiff_stack=False")

    module.run_motion_correction(config)
    print("Example completed successfully.")


if __name__ == "__main__":
    main()
