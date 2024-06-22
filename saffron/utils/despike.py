import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from typing import Union


def _sdev_loc(image, size):
    if type(size) is int:
        size = (size,) * image.ndim
    # print(f"max image {np.max(image)}")
    mean2 = _blur(image, size) ** 2
    vari = _blur(image**2, size)
    vari -= mean2
    vari[vari <= 0] = 10**-20
    return np.sqrt(vari)


def _blur(image, size, output=None):

    if type(size) is int:
        size = (size,) * image.ndim

    if output is None:
        output = np.empty_like(image)

    uniform_filter(image, size=size, output=output, mode="reflect")

    return output


def sigma_clip(data3D, size, low=3, high=3, iterations=1):
    output = np.copy(data3D)

    if type(size) is int:
        size = (size,) * data3D.ndim

    for iteration in range(iterations):
        med = median_filter(output, size)
        std_dev = _sdev_loc(output, size)
        diff = output - med

        condition = (diff > high * std_dev) | (diff < -low * std_dev)
        # if np.any(condition): print(f"found {condition[condition==True].size}/{condition.size} spikes")
        # plt.figure();plt.pcolormesh(med,vmax=-0.1,vmin=0.1);plt.colorbar()

        output[:] = np.where(condition, med, output)

    return output


def despike(
    raw_data: np.ndarray,
    clipping_sigma: float = 2.5,
    clipping_med_size: Union[list[int], int] = [6, 3, 3],
    clipping_iterations: int = 3,
):
    """
    Removes spikes (outliers) in a 3D dataset by replacing them with the local median of a fixed size.

    Parameters:
    ----------
    raw_data : numpy.ndarray
        The input 3D data array with shape (n_frames, n_rows, n_cols).
    clipping_sigma : float, optional
        The clipping threshold in units of sigma. Default is 2.5.
    clipping_med_size : list of less than 5 integers or one int, optional
        The size of the local median filter window along each dimension. Default is [6,3,3].
    clipping_iterations : int, optional
        The number of iterations of the sigma-clipping algorithm to apply. Default is 3.

    Returns:
    -------
    numpy.ndarray
        The despike 3D data array with the same shape as the input `raw_data`.
    """

    data = raw_data[0].copy()
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

    # trancate the area where all pixels are NaNs
    trunc_data = data[min_i : max_i + 1].copy()

    # Despiking data
    clipped_data = np.zeros_like(trunc_data)
    clipped_data = sigma_clip(
        trunc_data,
        size=clipping_med_size,
        high=clipping_sigma,
        low=clipping_sigma,
        iterations=clipping_iterations,
    )
    data = raw_data.copy()
    data[0, min_i : max_i + 1] = clipped_data.copy()
    return data
