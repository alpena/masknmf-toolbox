import numpy as np
import torch
import masknmf
import pytest
import importlib.util
import h5py
from pathlib import Path


def _load_1pmcri_module():
    module_path = Path(__file__).resolve().parents[1] / "dev" / "1pMCRI_motion_correct_data.py"
    spec = importlib.util.spec_from_file_location("mcri_motion_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dcimg_loader_is_exported():
    assert hasattr(masknmf, "DcimgArray")


def test_prefix_frame_loader_basic_indexing():
    module = _load_1pmcri_module()
    PrefixFrameLoader = module.PrefixFrameLoader

    base = np.arange(10 * 4 * 5, dtype=np.float32).reshape(10, 4, 5)
    loader = PrefixFrameLoader(base_loader=base, max_frames=3)

    assert loader.shape == (3, 4, 5)
    np.testing.assert_array_equal(loader[:], base[:3])
    np.testing.assert_array_equal(loader[0], base[0])
    np.testing.assert_array_equal(loader[-1], base[2])
    np.testing.assert_array_equal(loader[[0, 2]], base[[0, 2]])


def test_prefix_frame_loader_out_of_range_raises():
    module = _load_1pmcri_module()
    PrefixFrameLoader = module.PrefixFrameLoader

    base = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    loader = PrefixFrameLoader(base_loader=base, max_frames=2)

    with pytest.raises(IndexError):
        _ = loader[2]
    with pytest.raises(IndexError):
        _ = loader[[0, 2]]


def test_rigid_motion_gpu():
    x = (np.random.rand(4, 32, 32) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3))
    rigid.compute_template(x)
    out = masknmf.RegistrationArray(x, rigid)[:]
    assert out.shape == x.shape

def test_rigid_motion_cpu():
    x = (np.random.rand(4, 32, 32) * 4096).astype(np.int16)
    rigid = masknmf.RigidMotionCorrector(max_shifts=(3, 3))
    rigid.compute_template(x)
    out = masknmf.RegistrationArray(x, rigid)[:]
    assert out.shape == x.shape


def test_async_extract_export_with_dummy_strategy(tmp_path):
    module = _load_1pmcri_module()

    movie = np.arange(4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5)
    out_path = tmp_path / "dummy_extract_async.h5"

    module.export_extract_h5_streaming_async(
        data_loader=movie,
        strategy=masknmf.DummyMotionCorrector(),
        out_path=out_path,
        batch_size=2,
        motion_dtype="float32",
        compression="none",
        orientation_fix="none",
        chunk_t=2,
        chunk_x=0,
        chunk_y=0,
        filter_function=None,
        prefetch_batches=2,
        writer_queue_batches=2,
        pin_memory=False,
    )

    with h5py.File(out_path, "r") as h5f:
        np.testing.assert_allclose(h5f["/mov"][...], movie)
        expected_mean = movie.mean(axis=0).astype(np.float32)
        np.testing.assert_allclose(h5f["/F_per_pixel"][...], expected_mean)
        assert "/shifts" not in h5f
