"""
lab_utilities.py

Minimal utilities for saving/loading image data as NumPy arrays.

Design goals:
- Always save raw data (no implicit scaling)
- Simple, predictable API
- Zero ambiguity for students
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data



# -----------------------------------------------------------------------------
# Data directory
# -----------------------------------------------------------------------------

DATA_DIR = Path("data")


def ensure_data_dir() -> Path:
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


# -----------------------------------------------------------------------------
# Filename helpers
# -----------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_filename(name: str, add_timestamp: bool = True) -> Path:
    ensure_data_dir()

    if add_timestamp:
        fname = f"{name}_{timestamp()}.npy"
    else:
        fname = f"{name}.npy"

    return DATA_DIR / fname


# -----------------------------------------------------------------------------
# Save / Load
# -----------------------------------------------------------------------------

def save_image(
    image: np.ndarray,
    name: str = "image",
    add_timestamp: bool = True,
) -> Path:
    """
    Save image data as a NumPy array (.npy).

    Parameters
    ----------
    image : np.ndarray
        Image data (2D or 3D)
    name : str
        Base filename
    add_timestamp : bool
        Prevent overwriting by appending timestamp
    """
    path = make_filename(name, add_timestamp)

    np.save(path, image)

    print(f"[lab_utilities] Saved → {path}")
    return path


def load_image(path: str | Path) -> np.ndarray:
    """
    Load image from file.

    Supported formats:
    - .npy (preferred, preserves raw data)
    - .png (for visualization / external data)

    Returns
    -------
    np.ndarray
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        image = np.load(path)

    elif suffix == ".png":
        image = plt.imread(path)

        # Convert RGBA → grayscale if needed (common for matplotlib PNGs)
        if image.ndim == 3:
            # Drop alpha if present
            image = image[..., :3]

            # Convert to grayscale using luminance weighting
            image = (
                0.299 * image[..., 0]
                + 0.587 * image[..., 1]
                + 0.114 * image[..., 2]
            )

        print("[lab_utilities] Warning: PNG loaded (data may be scaled/quantized)")

    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    print(f"[lab_utilities] Loaded ← {path}")
    return image

import numpy as np


def load_stripes(
    orientation: str = "vertical",
    bright_width: int = 2,
    dark_width: int = 2,
    shape: tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Generate a binary stripe pattern.

    The image consists of alternating bright (1.0) and dark (0.0) regions
    with a fixed spatial period defined by bright_width + dark_width.
    The pattern is periodic along one axis and uniform along the other.

    :param orientation: Direction of the stripes. Must be 'vertical' or 'horizontal'.
                        'vertical' means variation along columns (x-axis),
                        'horizontal' means variation along rows (y-axis).
    :type orientation: str

    :param bright_width: Width of bright regions (value = 1.0) in pixels.
                         Must be a positive integer.
    :type bright_width: int

    :param dark_width: Width of dark regions (value = 0.0) in pixels.
                       Must be a positive integer.
    :type dark_width: int

    :param shape: Output image shape as (rows, cols).
    :type shape: tuple[int, int]

    :return: 2D array of shape ``shape`` containing values 0.0 and 1.0.
    :rtype: numpy.ndarray

    :raises ValueError: If orientation is invalid or widths are not positive.

    .. note::
        The stripe pattern is generated using a modulo operation with period

        T = bright_width + dark_width

        This produces a periodic structure that leads to discrete peaks
        in Fourier space.
    """

    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    if bright_width <= 0 or dark_width <= 0:
        raise ValueError("bright_width and dark_width must be positive integers")

    rows, cols = shape
    period = bright_width + dark_width

    if orientation == "vertical":
        coords = np.arange(cols)
        stripe_1d = (coords % period) < bright_width
        img = np.tile(stripe_1d.astype(float), (rows, 1))
    else:
        coords = np.arange(rows)
        stripe_1d = (coords % period) < bright_width
        img = np.tile(stripe_1d[:, None].astype(float), (1, cols))

    return img

def load_grace_hopper() -> np.ndarray:
    """
    Load a standard grayscale test image using matplotlib sample data.

    The image is converted to grayscale and normalized
    to the range [0, 1].

    :return: 2D grayscale image normalized to [0, 1].
    :rtype: numpy.ndarray
    """
    with get_sample_data('grace_hopper.jpg') as file:
        img = plt.imread(file)

    # Convert to grayscale if RGB
    if img.ndim == 3:
        img = img[..., :3]  # drop alpha if present
        img = np.mean(img, axis=2)

    img = img.astype(np.float64)
    img /= img.max()

    return img


# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------

def list_data_files():
    """List all saved .npy files."""
    ensure_data_dir()
    return sorted(DATA_DIR.glob("*.npy"))


def load_latest(name: str | None = None) -> np.ndarray:
    """
    Load the most recent file, optionally filtered by name.
    """
    ensure_data_dir()

    files = list_data_files()

    if name is not None:
        files = [f for f in files if name in f.name]

    if not files:
        raise FileNotFoundError("No matching files found.")

    latest = max(files, key=lambda f: f.stat().st_mtime)

    return load_image(latest)


def fourier_transform_image(image: np.ndarray) -> np.ndarray:
    """
    Compute the centered 2D Fourier transform of an image.

    If the input is a real, nonnegative image, it is interpreted as an
    intensity image and the square root is taken before transforming.
    Otherwise, the image is transformed directly.

    The transform uses unitary normalization so that the total power,
    sum(abs(field)**2), is preserved.

    :param image: 2D image (real or complex)
    :type image: np.ndarray

    :return: Complex-valued, centered Fourier transform
    :rtype: np.ndarray
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")

    if np.iscomplexobj(image):
        field = image
    else:
        if np.all(image >= 0):
            field = np.sqrt(image)
        else:
            field = image

    return np.fft.fftshift(
        np.fft.fft2(
            np.fft.ifftshift(field),
            norm="ortho",
        )
    )


def azimuthal_average(
    image: np.ndarray,
    num_bins: int = 100,
    center: tuple[float, float] | None = None,
    normalize_radius: bool = True,
    cumulative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the azimuthal average of an image about its center.

    The average is computed in radial bins out to the largest inscribed
    circle. By default, the radius is normalized so that the outermost
    radius is 1.

    :param image: 2D real-valued image (typically intensity)
    :type image: np.ndarray

    :param num_bins: Number of radial bins
    :type num_bins: int

    :param center: Center in (row, column) pixel coordinates. If None,
                   uses the geometric center.
    :type center: tuple[float, float] or None

    :param normalize_radius: Normalize radius to [0, 1]
    :type normalize_radius: bool

    :param cumulative: If True, return encircled energy. If False,
                       return azimuthal average.
    :type cumulative: bool

    :return: (radii, values)
             radii = bin centers
             values = azimuthal average or encircled energy
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")

    if np.iscomplexobj(image):
        raise ValueError("azimuthal_average expects a real-valued image.")

    nrows, ncols = image.shape

    if center is None:
        cy = (nrows - 1) / 2.0
        cx = (ncols - 1) / 2.0
    else:
        cy, cx = center

    y, x = np.indices(image.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    r_max = min(cy, cx, nrows - 1 - cy, ncols - 1 - cx)
    if r_max <= 0:
        raise ValueError("Center too close to edge for inscribed circle.")

    mask = r <= r_max
    r = r[mask]
    values = image[mask]

    if normalize_radius:
        r = r / r_max
        r_edges = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        r_edges = np.linspace(0.0, r_max, num_bins + 1)

    bin_indices = np.digitize(r, r_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    bin_sums = np.bincount(bin_indices, weights=values, minlength=num_bins)
    bin_counts = np.bincount(bin_indices, minlength=num_bins)

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    if cumulative:
        cumulative_sum = np.cumsum(bin_sums)
        total = cumulative_sum[-1]
        if total == 0:
            radial_values = np.zeros_like(cumulative_sum, dtype=float)
        else:
            radial_values = cumulative_sum / total
    else:
        radial_values = np.full(num_bins, np.nan, dtype=float)
        valid = bin_counts > 0
        radial_values[valid] = bin_sums[valid] / bin_counts[valid]

    return r_centers, radial_values