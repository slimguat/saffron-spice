import numpy as np
from scipy.ndimage import median_filter, uniform_filter
from typing import Union
from multiprocessing import Pool
from .utils import gen_shmm
import os
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
    if data3D.shape[0] == 1:
        work_data = np.copy(data3D)[0]
        work_size = size[1:]
    elif data3D.shape[3] == 1:
        work_data = np.copy(data3D)[:, :, :, 0]
        work_size = size[:3]
    else:
        work_data = np.copy(data3D)
        work_size = size
    
    t = create_timer()
    for iteration in range(iterations):
        med = median_filter(work_data, work_size)
        std_dev = _sdev_loc(work_data, work_size)
        diff = work_data - med

        condition = (diff > high * std_dev) | (diff < -low * std_dev)
        # plt.figure();plt.pcolormesh(med,vmax=-0.1,vmin=0.1);plt.colorbar()
        work_data[:] = np.where(condition, med, work_data)

    if data3D.shape[0] == 1:
        output[0] = work_data
    elif data3D.shape[3] == 1:
        output[:, :, :, 0] = work_data
    else:
        output = work_data
        
    return output

import time

def create_timer():
    """
    Creates a timer function that calculates the time difference between successive calls.

    Returns:
    -------
    callable:
        A function that, when called, returns the time difference since the last call.
    """
    last_called = [time.time()]  # Use a mutable object to hold the last call time

    def timer():
        now = time.time()
        elapsed = now - last_called[0]
        last_called[0] = now  # Update the last call time
        return elapsed

    return timer

def despike_depr(
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

def despike_4D(
    raw_data: np.ndarray,
    clipping_sigma: float = 2.5,
    clipping_med_size: Union[list[int], int] = [3, 6, 3, 3],
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
    
    data = raw_data.copy()
    mean_spec = np.nanmean(data, axis=(0,2, 3))  # finding the mean spectrum

    # Removing the nans by replacing them with the mean spectrum pixel values
    nan_mask = np.isnan(data)
    nan_indices = nan_mask.nonzero()
    data[nan_mask] = mean_spec[nan_indices[1]]
    
    # there is still some NaNs in the spectrum as some pixels (in the boarders are all NaNs so the mean yeilds also NaNs)
    # finding where the NaNs are located in data
    indeces = np.where(np.logical_not(np.isnan(data[:,:, :, :])))[1]
    min_i = (np.min(indeces) if indeces[0].size > 0 else 0)
    max_i = (np.max(indeces) if indeces[0].size > 0 else data.shape[1] - 1)

    # trancate the area where all pixels are NaNs
    trunc_data = data[:,min_i : max_i + 1].copy()    
    
    clipped_data = np.zeros_like(trunc_data)
    despiked_data = raw_data.copy()
    # Despiking data
    clipped_data = np.zeros_like(trunc_data)
    clipped_data = sigma_clip(
        trunc_data,
        size=clipping_med_size,
        high=clipping_sigma,
        low=clipping_sigma,
        iterations=clipping_iterations,
    )
    despiked_data[:, min_i : max_i + 1] = clipped_data.copy()
    despiked_data[np.isnan(raw_data)] = np.nan
    # print("removing the spikes values")
    # despiked_data[despiked_data!=raw_data] = np.nan
    
    return despiked_data


def process_indices_depr(args):
    """
    Process a batch of (t, y, x) indices for despiking.

    Parameters:
    ----------
    indices : list of tuples
        List of (t, y, x) indices to process.
    raw_data : np.ndarray
        4D data array with shape (n_times, n_wavelengths, n_rows, n_cols).
    clipping_sigma : float
        The clipping threshold in units of sigma.
    clipping_med_size : int or list of int
        The size of the local median filter window along each dimension.
    clipping_iterations : int
        The number of iterations of the sigma-clipping algorithm to apply.

    Returns:
    -------
    list of tuples:
        Processed (t, y, x, spectrum) tuples.
    """
    indices, shmm, clipping_sigma, clipping_med_size, clipping_iterations = args
    results = []
    shmm_,raw_data = gen_shmm(create=False, **shmm)
    for t, y, x in indices:
        spectrum = raw_data[t, :, y, x]

        if np.isnan(spectrum).all():
            continue  # Skip entirely NaN spectra

        # Replace NaNs with the local mean
        mean_spec = np.nanmean(spectrum)
        spectrum[np.isnan(spectrum)] = mean_spec

        # Apply sigma clipping to remove spikes
        clipped_spectrum = sigma_clip(
            spectrum,
            size=clipping_med_size[0] if isinstance(clipping_med_size, list) else clipping_med_size,
            high=clipping_sigma,
            low=clipping_sigma,
            iterations=clipping_iterations,
        )

        results.append((t, y, x, clipped_spectrum))

    return results


def despike_4d_depr(
    raw_data: np.ndarray,
    clipping_sigma: float = 2.5,
    clipping_med_size: Union[list[int], int] = [6, 3, 3],
    clipping_iterations: int = 3,
    num_jobs: int = 4,
):
    """
    Removes spikes (outliers) in a 4D dataset by dividing the workload into `num_jobs` batches and processing in parallel.

    Parameters:
    ----------
    raw_data : np.ndarray
        The input 4D data array with shape (n_times, n_wavelengths, n_rows, n_cols).
    clipping_sigma : float, optional
        The clipping threshold in units of sigma. Default is 2.5.
    clipping_med_size : list of less than 5 integers or one int, optional
        The size of the local median filter window along each dimension. Default is [6,3,3].
    clipping_iterations : int, optional
        The number of iterations of the sigma-clipping algorithm to apply. Default is 3.
    num_jobs : int, optional
        The number of parallel jobs to run. Default is None (uses all available CPUs).

    Returns:
    -------
    np.ndarray
        The despiked 4D data array with the same shape as the input `raw_data`.
    """
    if num_jobs is None or num_jobs < 1:
        raise ValueError("num_jobs must be greater than or equal to 1.")

    # Validate data shape
    if len(raw_data.shape) != 4:
        raise ValueError("Input data must be a 4D array with shape (time, wavelength, height, width).")

    raw_data = raw_data.copy()  # Copy input data to avoid modifying it directly
    raw_data2 = raw_data.copy()
    result_data = raw_data.copy()*np.nan
    n_time, _, n_y, n_x = raw_data.shape

    # Generate all (t, y, x) indices
    t_indices, y_indices, x_indices = np.meshgrid(
        np.arange(n_time), np.arange(n_y), np.arange(n_x), indexing='ij'
    )
    all_indices = list(zip(t_indices.ravel(), y_indices.ravel(), x_indices.ravel()))

    # Calculate batch size to divide indices into `num_jobs` batches
    batch_size = len(all_indices) // num_jobs
    batches = [all_indices[i * batch_size:(i + 1) * batch_size] for i in range(num_jobs)]

    # Ensure the last batch gets any remaining indices
    if len(all_indices) % num_jobs != 0:
        batches[-1].extend(all_indices[num_jobs * batch_size:])

    
    Shmm,data = gen_shmm(create=True, ndarray=raw_data)
    shmm_dict = {'name': Shmm.name, "dtype": raw_data.dtype, "shape": raw_data.shape}
    data[:] = raw_data
     
    # Prepare arguments for parallel processing
    args = [
        (batch, shmm_dict, clipping_sigma, clipping_med_size, clipping_iterations)
        for batch in batches
    ]
    # Initialize multiprocessing pool
    with Pool(processes=num_jobs) as pool:
        results = pool.map(process_indices, args)
    
    # results = []
    # for arg in args:
    #     results.append(process_indices(arg))
    
    # Combine results and update the raw_data
    for result_batch in results:
        for t, y, x, clipped_spectrum in result_batch:
            # mask_= np.isnan(raw_data2[t, :, y, x])
            result_data[t, :, y, x] = clipped_spectrum  # Update the spectrum in the data cube
            # result_data[t, :, y, x][mask_] = np.nan
    
    result_data[np.isnan(raw_data2)] = np.nan
    return result_data


