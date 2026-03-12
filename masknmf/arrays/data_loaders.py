from .array_interfaces import LazyFrameLoader
import tifffile
import numpy as np
import h5py
from pathlib import Path
from typing import *


class TiffArray(LazyFrameLoader):
    def __init__(self, filename):
        """
        TiffArray data loading object. Supports loading data from multipage tiff files.

        Args:
            filename (str): Path to file

        """
        self.filename = filename

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        with tifffile.TiffFile(self.filename) as tffl:
            num_frames = len(tffl.pages)
            for page in tffl.pages[0:1]:
                image = page.asarray()
            x, y = page.shape
        return num_frames, x, y

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        if isinstance(indices, int):
            data = tifffile.imread(self.filename, key=[indices]).squeeze()
        elif isinstance(indices, list):
            data = tifffile.imread(self.filename, key=indices).squeeze()
        else:
            indices_list = list(
                range(
                    indices.start or 0, indices.stop or self.shape[0], indices.step or 1
                )
            )
            data = tifffile.imread(self.filename, key=indices_list).squeeze()
        return data.astype(self.dtype)


class Hdf5Array(LazyFrameLoader):
    def __init__(self, filename: str, field: str) -> None:
        """
        Generic lazy loader for Hdf5 files video files, where data is stored as (T, x, y). T is number of frames,
        x and y are the field of view dimensions (height and width).

        Args:
            filename (str): Path to filename
            field (str): Field of hdf5 file containing data
        """
        if not isinstance(field, str):
            raise ValueError("Field must be a string")
        self.filename = filename
        self.field = field
        with h5py.File(self.filename, "r") as file:
            # Access the 'field' dataset
            field_dataset = file[self.field]

            # Get the shape of the array
            self._shape = field_dataset.shape

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        with h5py.File(self.filename, "r") as file:
            # Access the 'field' dataset
            field_dataset = file[self.field]
            if isinstance(indices, int):
                data = field_dataset[indices, :, :].squeeze()
            elif isinstance(indices, list):
                data = field_dataset[indices, :, :].squeeze()
            else:
                indices_list = list(
                    range(
                        indices.start or 0,
                        indices.stop or self.shape[0],
                        indices.step or 1,
                    )
                )
                data = field_dataset[indices_list, :, :].squeeze()
        return data.astype(self.dtype)


class DcimgArray(LazyFrameLoader):
    def __init__(self, filename: str | Path, first_4px_correction: bool = False) -> None:
        """
        DCIMG lazy loader. Supports loading data from Hamamatsu .dcimg files.

        Args:
            filename: Path to .dcimg file
            first_4px_correction: If True, ask dcimg reader to apply first-4-pixel correction.
        """
        self.filename = str(filename)
        self.first_4px_correction = bool(first_4px_correction)
        self._shape = self._discover_shape()

    def _open_dcimg(self):
        try:
            from dcimg import DCIMGFile
        except ImportError as exc:
            raise ImportError(
                "DcimgArray requires the `dcimg` package from GitHub. "
                "Install with: pip install "
                "\"git+https://github.com/lens-biophotonics/dcimg.git@3f1e2eec27a4e414903b0fcd1da711e7f565dcce\""
            ) from exc

        try:
            return DCIMGFile(
                self.filename, first_4px_correction=self.first_4px_correction
            )
        except TypeError as exc:
            if self.first_4px_correction:
                raise TypeError(
                    "Installed dcimg version does not support "
                    "`first_4px_correction` argument."
                ) from exc
            return DCIMGFile(self.filename)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open DCIMG file: {self.filename}. "
                "Known issue: some dcimg versions fail with numpy>=2; "
                "try numpy<2 in a dedicated conversion environment."
            ) from exc

    def _discover_shape(self) -> tuple[int, int, int]:
        with self._open_dcimg() as dcimg_file:
            shape = tuple(dcimg_file.shape)

        if len(shape) != 3:
            raise ValueError(
                f"Expected a 3D movie from dcimg reader, got shape {shape}."
            )
        return cast(tuple[int, int, int], shape)

    @property
    def dtype(self) -> str:
        """
        str
            data type
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    def _normalize_single_index(self, index: int) -> int:
        norm = int(index)
        if norm < 0:
            norm += self.shape[0]
        if norm < 0 or norm >= self.shape[0]:
            raise IndexError(
                f"Frame index {index} is out of range for movie with "
                f"{self.shape[0]} frames."
            )
        return norm

    def _read_from_handle(
        self,
        dcimg_file,
        indices: Union[list, int, slice],
    ) -> np.ndarray:
        if isinstance(indices, int):
            data = np.asarray(dcimg_file[self._normalize_single_index(indices)])
        elif isinstance(indices, list):
            if len(indices) == 0:
                return np.empty((0, self.shape[1], self.shape[2]), dtype=self.dtype)
            indices_normalized = [self._normalize_single_index(i) for i in indices]
            frame_list = [np.asarray(dcimg_file[i]) for i in indices_normalized]
            data = np.stack(frame_list, axis=0)
        else:
            start = indices.start or 0
            stop = indices.stop or self.shape[0]
            step = indices.step or 1
            if step == 0:
                raise ValueError("slice step cannot be zero")
            if step > 0 and start >= stop:
                return np.empty((0, self.shape[1], self.shape[2]), dtype=self.dtype)
            if step < 0 and start <= stop:
                return np.empty((0, self.shape[1], self.shape[2]), dtype=self.dtype)
            data = np.asarray(dcimg_file[start:stop:step])

        if data.ndim == 2:
            data = data[None, :, :]
        return data.astype(self.dtype, copy=False)

    def _compute_at_indices(self, indices: Union[list, int, slice]) -> np.ndarray:
        with self._open_dcimg() as dcimg_file:
            return self._read_from_handle(dcimg_file, indices)
