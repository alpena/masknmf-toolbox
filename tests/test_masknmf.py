import numpy as np
import torch
import masknmf
import pytest
import importlib.util
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
