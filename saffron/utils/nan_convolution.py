import numpy as np
from scipy.ndimage import uniform_filter


def validate_min_valid_fraction(min_valid_fraction):
    """Validate the minimum valid fraction."""
    if not isinstance(min_valid_fraction, (int, float)):
        raise TypeError(
            "min_valid_fraction must be a float in [0, 1]."
        )

    if min_valid_fraction < 0.0 or min_valid_fraction > 1.0:
        raise ValueError(
            f"min_valid_fraction must be in [0, 1], got {min_valid_fraction}."
        )


def validate_mode(mode):
    """Validate convolution mode."""
    if mode == "cercle":
        raise ValueError("cercle mode is deprecated")

    if mode != "box":
        raise ValueError(
            f"mode {mode!r} is not implemented or is misspelled."
        )


def validate_4d_array(name, array):
    """Validate that an input is a 4D NumPy array of float64."""
    array = np.asarray(array, dtype=np.float64)

    if array.ndim != 4:
        raise ValueError(
            f"{name} must be a 4D array, got ndim={array.ndim}."
        )

    return array


def validate_same_shape(data, sigma):
    """Validate that sigma has the same shape as data."""
    sigma = np.asarray(sigma, dtype=np.float64)

    if sigma.shape != data.shape:
        raise ValueError(
            f"sigma must have the same shape as data. "
            f"Got data.shape={data.shape}, sigma.shape={sigma.shape}."
        )

    return sigma


def normalize_filter_size(size, ndim):
    """Normalize a filter size to a tuple of length ndim."""
    if np.isscalar(size):
        normalized = (int(size),) * ndim
    else:
        normalized = tuple(int(value) for value in size)

    if len(normalized) != ndim:
        raise ValueError(
            f"Filter size must have length {ndim}, got {len(normalized)}."
        )

    for value in normalized:
        if value < 1:
            raise ValueError(
                f"All filter extents must be >= 1, got {normalized}."
            )

    return normalized


def validate_kernel_fits_window(size, data_shape):
    """Check whether a kernel fits within the data shape."""
    for axis_index, kernel_extent in enumerate(size):
        if kernel_extent > data_shape[axis_index]:
            raise ValueError(
                "Kernel extent exceeds data size on one axis. "
                f"kernel={size}, data_shape={data_shape}, axis={axis_index}"
            )


def compute_boundary_slices(size, data_shape):
    """Compute the strictly interior region unaffected by reflected padding."""
    slices = []

    for axis_index in range(len(size)):
        kernel_extent = size[axis_index]
        axis_length = data_shape[axis_index]

        left_trim = kernel_extent // 2
        right_trim = (kernel_extent - 1) // 2

        start = left_trim
        stop = axis_length - right_trim

        if start >= stop:
            slices.append(slice(0, 0))
        else:
            slices.append(slice(start, stop))

    return tuple(slices)


def compute_valid_mask(data, sigma=None):
    """
    Build the validity mask.

    If sigma is provided, a pixel is valid only if both data and sigma are finite.
    """
    data_valid = np.isfinite(data)

    if sigma is None:
        return data_valid

    sigma_valid = np.isfinite(sigma)
    return data_valid & sigma_valid


def compute_local_sum(array, size, mode):
    """
    Compute the local sum using uniform_filter.

    uniform_filter returns a local mean, so we multiply by kernel volume.
    """
    kernel_volume = float(np.prod(size))
    local_mean = uniform_filter(array, size=size, mode=mode)
    local_sum = local_mean * kernel_volume
    return local_sum


def compute_local_valid_count(valid_mask, size, mode):
    """Compute the number of valid pixels in each kernel."""
    valid_mask_float = valid_mask.astype(np.float64)
    return compute_local_sum(valid_mask_float, size=size, mode=mode)


def compute_accepted_mask(local_valid_count, kernel_volume, min_valid_fraction):
    """Compute which output positions satisfy the valid-fraction rule."""
    local_valid_fraction = local_valid_count / kernel_volume

    accepted = (
        (local_valid_count > 0.0) &
        (local_valid_fraction >= min_valid_fraction)
    )

    return accepted


def apply_boundary_mask(filtered_data, size):
    """Mask out boundary regions affected by reflect padding."""
    interior_slices = compute_boundary_slices(
        size=size,
        data_shape=filtered_data.shape,
    )

    strictly_valid = np.full(filtered_data.shape, False, dtype=bool)
    strictly_valid[interior_slices] = True

    filtered_data[~strictly_valid] = np.nan
    return filtered_data


def compute_filtered_data(data, valid_mask, size, mode, min_valid_fraction):
    """
    Compute the NaN-aware filtered data.

    Output is the mean over valid values only.
    """
    filled_data = np.where(valid_mask, data, 0.0)

    kernel_volume = float(np.prod(size))
    local_sum_data = compute_local_sum(filled_data, size=size, mode=mode)
    local_valid_count = compute_local_valid_count(
        valid_mask=valid_mask,
        size=size,
        mode=mode,
    )

    accepted = compute_accepted_mask(
        local_valid_count=local_valid_count,
        kernel_volume=kernel_volume,
        min_valid_fraction=min_valid_fraction,
    )

    filtered_data = np.full(data.shape, np.nan, dtype=np.float64)
    filtered_data[accepted] = (
        local_sum_data[accepted] / local_valid_count[accepted]
    )

    return filtered_data, local_valid_count, accepted


def compute_filtered_sigma(
    sigma,
    valid_mask,
    size,
    mode,
    accepted,
    local_valid_count,
):
    """
    Propagate sigma for the mean over valid samples.

    sigma_mean = sqrt(sum(sigma_i^2)) / N_valid
    """
    filled_sigma2 = np.where(valid_mask, sigma**2, 0.0)
    local_sum_sigma2 = compute_local_sum(
        filled_sigma2,
        size=size,
        mode=mode,
    )

    filtered_sigma = np.full(sigma.shape, np.nan, dtype=np.float64)
    filtered_sigma[accepted] = (
        np.sqrt(local_sum_sigma2[accepted]) / local_valid_count[accepted]
    )

    return filtered_sigma


def nan_fractional_uniform_filter_with_sigma(
    data,
    size,
    sigma=None,
    min_valid_fraction=1.0,
    mode="reflect",
    drop_reflect_boundaries=True,
):
    """
    Apply a NaN-aware uniform box filter with optional sigma propagation.

    Parameters
    ----------
    data : ndarray
        Input array of any dimension.
    size : int or iterable of int
        Uniform filter size.
    sigma : ndarray or None, optional
        Per-pixel 1-sigma uncertainty. Must match data shape if provided.
    min_valid_fraction : float, optional
        Minimum required valid fraction in each kernel.
    mode : str, optional
        Boundary handling mode for uniform_filter.
    drop_reflect_boundaries : bool, optional
        If True, boundary regions affected by reflected padding are set to NaN.

    Returns
    -------
    filtered_data : ndarray
    filtered_sigma : ndarray or None
    """
    validate_min_valid_fraction(min_valid_fraction)

    data = np.asarray(data, dtype=np.float64)
    size = normalize_filter_size(size=size, ndim=data.ndim)

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape != data.shape:
            raise ValueError(
                f"sigma must have the same shape as data. "
                f"Got data.shape={data.shape}, sigma.shape={sigma.shape}."
            )

    valid_mask = compute_valid_mask(data=data, sigma=sigma)

    filtered_data, local_valid_count, accepted = compute_filtered_data(
        data=data,
        valid_mask=valid_mask,
        size=size,
        mode=mode,
        min_valid_fraction=min_valid_fraction,
    )

    filtered_sigma = None
    if sigma is not None:
        filtered_sigma = compute_filtered_sigma(
            sigma=sigma,
            valid_mask=valid_mask,
            size=size,
            mode=mode,
            accepted=accepted,
            local_valid_count=local_valid_count,
        )

    if drop_reflect_boundaries:
        filtered_data = apply_boundary_mask(filtered_data, size=size)
        if filtered_sigma is not None:
            filtered_sigma = apply_boundary_mask(filtered_sigma, size=size)

    return filtered_data, filtered_sigma


def convolve_4D_nan_aware(
    window,
    mode,
    convolution_extent_list,
    sigma=None,
    min_valid_fraction=1.0,
    verbose=0,
    drop_reflect_boundaries=True,
):
    """
    NaN-aware 4D box convolution with optional sigma propagation.

    Parameters
    ----------
    window : ndarray
        4D input data array.
    mode : str
        Only "box" is supported.
    convolution_extent_list : iterable
        One 4D kernel size per output slice.
    sigma : ndarray or None, optional
        Per-pixel 1-sigma uncertainty, same shape as window.
    min_valid_fraction : float, optional
        Minimum acceptable valid fraction in each kernel.
    verbose : int, optional
        Verbosity level.
    drop_reflect_boundaries : bool, optional
        If True, boundary regions influenced by reflect padding are set to NaN.

    Returns
    -------
    conv_data : ndarray
        Shape (n_kernels, *window.shape)
    conv_sigma : ndarray or None
        Same shape as conv_data if sigma is provided, otherwise None.
    """
    validate_mode(mode)
    validate_min_valid_fraction(min_valid_fraction)

    window = validate_4d_array("window", window)

    if sigma is not None:
        sigma = validate_same_shape(window, sigma)

    convolution_extent_list = np.asarray(convolution_extent_list, dtype=int)

    if convolution_extent_list.ndim != 2:
        raise ValueError(
            "convolution_extent_list must be a 2D array-like object "
            "with one kernel per row."
        )

    if convolution_extent_list.shape[1] != window.ndim:
        raise ValueError(
            f"Each kernel must have length {window.ndim}, "
            f"got shape {convolution_extent_list.shape}."
        )

    n_kernels = convolution_extent_list.shape[0]

    conv_data = np.full(
        (n_kernels, *window.shape),
        np.nan,
        dtype=np.float64,
    )

    conv_sigma = None
    if sigma is not None:
        conv_sigma = np.full(
            (n_kernels, *window.shape),
            np.nan,
            dtype=np.float64,
        )

    if verbose >= 1:
        print(f"Convolving using mode={mode!r}")
        print(f"min_valid_fraction={min_valid_fraction}")

    for kernel_index in range(n_kernels):
        raw_size = convolution_extent_list[kernel_index]
        size = normalize_filter_size(size=raw_size, ndim=window.ndim)
        validate_kernel_fits_window(size=size, data_shape=window.shape)

        if verbose >= 2:
            print(f"[{kernel_index + 1}/{n_kernels}] size={size}")

        if all(kernel_extent == 1 for kernel_extent in size):
            conv_data[kernel_index] = window.copy()
            if conv_sigma is not None and sigma is not None:
                conv_sigma[kernel_index] = sigma.copy()
            continue

        filtered_data, filtered_sigma = nan_fractional_uniform_filter_with_sigma(
            data=window,
            size=size,
            sigma=sigma,
            min_valid_fraction=min_valid_fraction,
            mode="reflect",
            drop_reflect_boundaries=drop_reflect_boundaries,
        )

        conv_data[kernel_index] = filtered_data

        if conv_sigma is not None:
            conv_sigma[kernel_index] = filtered_sigma

    return conv_data, conv_sigma
