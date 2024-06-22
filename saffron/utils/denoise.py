# Script that contains denoising algorithms and its requirements
import os
from astropy.io import fits
from watroo import denoise, B3spline
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import (
    ImageNormalize,
    AsymmetricPercentileInterval,
    SqrtStretch,
)


def denoise_data(row_data, denoise_sigma):
    """
    Apply denoising to a 3D data array containing NaN values.

    Parameters
    ----------
    row_data : np.ndarray
        3D data array to be denoised, containing NaN values.
    denoise_sigma : float
        Weight parameter used by the denoising algorithm.

    Returns
    -------
    np.ndarray
        The denoised 3D data array, with NaN values preserved.

    Notes
    -----
    This function replaces NaN values in the input data array by their mean
    values along the spectral axis, and then applies a denoising algorithm
    to the resulting array. The denoising is performed on a truncated version
    of the array, obtained by removing rows and columns that only contain NaN
    values. The final denoised array has the same shape as the input array,
    with NaN values in the same positions.

    This function assumes that the input array has dimensions (spectral_axis,
    x_axis, y_axis), where the spectral axis is the first dimension, and that
    the denoising algorithm is based on Anscombe variance stabilization.
    """
    # if np.any(np.isnan(row_data)):
    # data = np.nan_to_num(row_data)
    data = row_data.copy()
    mean_spec = np.nanmean(data, axis=(1, 2))  # finding the mean spectrum

    # Building a data cube that's a repetition of the same spectrum everywhere
    size = data.shape
    mean_rep = np.repeat(mean_spec, np.prod(size[1:]))
    mean_rep = mean_rep.reshape(size)

    # Replace the nans in data by their equivalent mean value
    data[np.isnan(data)] = mean_rep[np.isnan(data)]

    # there is still some NaNs in the spectrum as some pixels (in the boarders are all NaNs so the mean yeilds also NaNs)
    # finding where the NaNs are located in data
    indeces = np.where(np.logical_not(np.isnan(data[:, 0, 0])))
    min_i = np.min(indeces)
    max_i = np.max(indeces)
    trunc_data = data[min_i : max_i + 1].copy()

    # denoising
    denoised_3d_truncated = denoise(
        trunc_data, scaling_function=B3spline, weights=denoise_sigma, anscombe=True
    )

    # resizing the results to the actual data size
    denoised_3d = row_data.copy()
    denoised_3d[min_i : max_i + 1] = denoised_3d_truncated

    # reput the nans
    denoised_3d[np.isnan(row_data)] = np.nan

    return denoised_3d


def denoise_raster(windows, denoise_sigma):

    if type(windows) != list:
        hdu_list = fits.open(windows)
        # Filters valid extensions
        windows = [hdu for hdu in hdu_list if hdu.is_image and hdu.name != "WCSDVARR"]

    denoised_windows = []
    for window in windows:
        denoised_windows.append(window.data * 0)
        denoised_windows[-1][0] = denoise_data(window.data[0], denoise_sigma)

    return denoised_windows
