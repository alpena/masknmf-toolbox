import torch
import numpy as np
from typing import *
import math

from typing import List


def compute_highpass_filter_kernel(gaussian_sigma: List[float]) -> torch.Tensor:
    """
    Computes a high-pass filter kernel using a Gaussian filter.

    Args:
        gaussian_sigma (list[int]): Standard deviations for the Gaussian kernel.

    Returns:
        torch.Tensor: High-pass filter kernel.
    """
    if len(gaussian_sigma) != 2:
        raise ValueError("gaussian_sigma must have length 2")

    if any(s <= 0 for s in gaussian_sigma):
        raise ValueError("gaussian_sigma must contain positive values")

    sigma_h, sigma_w = gaussian_sigma

    radius_h = int(3 * sigma_h)
    radius_w = int(3 * sigma_w)

    coords_h = torch.arange(-radius_h, radius_h + 1, dtype=torch.float32)
    coords_w = torch.arange(-radius_w, radius_w + 1, dtype=torch.float32)

    g_h = torch.exp(-0.5 * (coords_h ** 2) / (sigma_h ** 2))
    g_w = torch.exp(-0.5 * (coords_w ** 2) / (sigma_w ** 2))

    kernel = g_h[:, None] @ g_w[None, :]
    kernel /= kernel.sum()

    kernel = -kernel
    kernel[radius_h, radius_w] += 1.0

    return kernel


def gaussian_kernel(kernel_size: int = 3, sigma: float = 1.0) -> torch.tensor:
    """Generates a 2D Gaussian kernel."""
    x = torch.arange(kernel_size) - kernel_size // 2
    y = torch.arange(kernel_size) - kernel_size // 2
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize to sum to 1
    return kernel


def image_filter(frames: torch.tensor, kernel: torch.tensor) -> torch.tensor:
    """
    Generic Image filter function; given a kernel an image stack, convolve every image with the kernel.

    Args:
        frames (torch.Tensor): Shape (num_frames, fov_dim1, fov_dim2)
        kernel (torch.Tensor): Shape (kH, kW)

    Returns:
        torch.Tensor: Convolved frames with shape (num_frames, fov_dim1, fov_dim2)
    """

    num_frames, fov_dim1, fov_dim2 = frames.shape
    kH, kW = kernel.shape

    # Reshape frames to fit conv2d input format: (batch=frames, channels=1, height, width)
    frames = frames.unsqueeze(1)  # Shape: (num_frames, 1, fov_dim1, fov_dim2)

    # Reshape kernel to fit conv2d weight format: (out_channels=1, in_channels=1, kH, kW)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, kH, kW)

    # Apply convolution (padding='same' ensures output size matches input size)
    convolved_frames = torch.nn.functional.conv2d(
        frames, kernel, padding="same"
    )  # Shape: (num_frames, 1, fov_dim1, fov_dim2)

    # Remove channel dimension
    return convolved_frames.squeeze(1)  # Shape: (num_frames, fov_dim1, fov_dim2)
