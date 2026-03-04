# masknmf-toolbox

[**Installation**](https://github.com/apasarkar/masknmf-toolbox#Installation) |
[**API**](https://github.com/apasarkar/masknmf-toolbox#API) |
[**Data Formats**](https://github.com/apasarkar/masknmf-toolbox#examples) |
[**Paper**](https://github.com/apasarkar/masknmf-toolbox#Paper) |

PyTorch implementation of the masknmf framework for {calcium, voltage, glutamate} imaging analysis. Supports GPU-accelerated:
- Motion Correction
- Compression and Denoising
- Signal Demixing
- High-performance visualization

## Installation
 
Tests are run against Python 3.11 and 3.12, on Linux and Windows using `pip` and `miniforge3`.

### Download the repository

Until the package is published to PyPI, you have a few options to download the software:
1. `git clone` (recommended)
2. Download directly from [GitHub (code -> Download ZIP)](https://github.com/apasarkar/masknmf-toolbox)
3. Install directly from a branch (see [Skip the cloning step](https://github.com/apasarkar/masknmf-toolbox#skip-the-cloning-step))

```bash
git clone https://github.com/apasarkar/masknmf-toolbox.git
cd masknmf-toolbox
```

### pip

Virtual environments are outside the scope of this README, but in general we recommend:
- [UV](https://docs.astral.sh/uv/) (strongly recommended, prepend `uv` to all `pip` commands)
```bash
uv pip install .
```
- [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments)
```bash
pip install .
```

### miniforge3

The only tested and supported flavor of `anaconda` is [miniforge3](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers):

```bash
conda create -n masknmf -c conda-forge python=3.12
pip install .
```

### Skip the cloning step

If your environment is already set up, you can skip the cloning step and install directly from a branch of the repository:

```bash
# with standard venvs 
$ pip install git+https://github.com/apasarkar/masknmf-toolbox.git@main
# or with UV
$ uv pip install git+https://github.com/apasarkar/masknmf-toolbox.git@main

Installed 1 package in 0.63ms
 - masknmf-toolbox==0.1.0 
 + masknmf-toolbox==0.1.0 
```

## GPU Dependencies

The default installation of PyTorch will not have cuda enabled.
To get the cuda-enabled PyTorch installation

Find which Cuda version you're using (e.g. cuda_12.6)

```bash
nvcc --version
% or
nvidia-smi
```

Windows should have `nvcc` available in the command prompt if you have installed the CUDA toolkit.

If not, you can find it in the CUDA installation directory:

- Windows:`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin`
- Linux/Unix: `/usr/local/cuda/bin/nvcc`

```bash
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:55:00_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```

Install the version of PyTorch that matches your cuda and operating system [on the PyTorch Getting Started](https://pytorch.org/get-started/locally/).

## API
See the notebooks folder for demos on how to use the motion correction, compression, and demixing APIs.

## Data Formats
Support currently provided for 
- multipage .tiff files 
- hdf5 files. 
- .dcimg files (via optional `dcimg` dependency)

For `.dcimg`, use the latest GitHub version of `dcimg`:

```bash
pip install "git+https://github.com/lens-biophotonics/dcimg.git@3f1e2eec27a4e414903b0fcd1da711e7f565dcce"
```

Note:
- In our current tests, `dcimg` failed with `numpy>=2` and worked with `numpy<2`.
- Registration itself does not require `cvxpy`, but `cvxpy` in this project depends on `numpy>=2`.
- If this conflict affects your workflow, use a separate environment for dcimg conversion.

or install toolbox extra dependencies:

```bash
pip install ".[dcimg]"
```

Support for other formats can be easily added by defining a data loader class that implements `LazyFrameLoader`.

### DCIMG quick start

Convert dcimg to toolbox-compatible HDF5 (standard dataset: `motion_corrected`):

```bash
python dev/dcimg_to_hdf5_sample.py \
  --input_dcimg_path demo_data/250206-UK6-1-F=4_power=5mW_00001.dcimg \
  --out_h5_path demo_data/output/250206-UK6-1-F=4_power=5mW_00001_first500.h5 \
  --max_frames 500
```

Run piecewise-rigid motion correction directly from dcimg and export standard toolbox HDF5:

```bash
python dev/1pMCRI_motion_correct_data.py \
  --input_image_path demo_data/250206-UK6-1-F=4_power=5mW_00001.dcimg \
  --out_path demo_data/output/250206-UK6-1-F=4_power=5mW_moco.h5 \
  --input_max_frames 500 \
  --no-export_tiff_stack
```

Batch mode (single script, multiple files):

```bash
python dev/1pMCRI_motion_correct_data.py \
  --input_dir demo_data \
  --glob_pattern "*.dcimg" \
  --output_dir demo_data/output \
  --out_suffix "_moco" \
  --input_max_frames 500 \
  --frame_batch_size 20 \
  --no-export_tiff_stack
```

## Paper

If you use this method, please cite the accompanying [paper](https://www.biorxiv.org/content/10.1101/2023.09.14.557777v1)

> _maskNMF: A denoise-sparsen-detect approach for extracting neural signals from dense imaging data_. (2023). A. Pasarkar\*, I. Kinsella, P. Zhou, M. Wu, D. Pan, J.L. Fan, Z. Wang, L. Abdeladim, D.S. Peterka, H. Adesnik, N. Ji, L. Paninski.
