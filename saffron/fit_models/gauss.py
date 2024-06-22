from numba import jit
import numpy as np


@jit(nopython=True)
def Gauss(x: np.ndarray, I_max: float, x0: float, sigma: float, B: float) -> np.ndarray:
    """Function that returns the gaussian of an x array
            $Gauss(x) = I_max e^{-\frac{(x-x_0)^2}{2\sigma}}$
    Args:
        x (np.ndarray): x array.
        I_max (float): the maximum value.
        x0 (float): the position of I_max.
        sigma (float): the standard diviation.
        B (float): elevation value for the gaussian.
    Return:
        Gauss (np.ndarray): calculated value of x with the same length
    """

    exponent = -((x - x0) ** 2) / (2 * sigma**2)
    res = I_max * np.exp(exponent) + B
    return res


@jit(nopython=True)
def multiGauss(x: np.ndarray, params_list: np.ndarray, B: float):
    """Function that returns the sum of gaussians of an x array
            $Gauss(x) = I_max e^{-\frac{(x-x_0)^2}{2\sigma}}$

    Args:
        x (np.ndarray): x array.
        params_list (np.ndarray): a table of parameters (nx3) with numbre of gaussians, 3 parameters needed for every gaussian.
        B (float): elevation value for the gaussian.
    """
    res = x.copy() * 0 + B
    for params in params_list:
        I_max, x0, sigma = params
        exponent = -((x - x0) ** 2) / (2 * sigma**2)
        res += I_max * np.exp(exponent)

    return res


def flat_inArg_multiGauss(x, *array):
    array = np.array(array)
    return flat_multiGauss(x, array)


@jit(nopython=True, inline="always", error_model="numpy")
def flat_multiGauss(x, array):
    assert len(array.shape) == 1  # array parameter should be a one 1D array
    assert (
        array.shape[0] % 3 == 1
    )  # and array.shape[0]>3 # array should contain 3xn gaussian params + 1 background pparam
    i = 0
    len_array = array[:-1].shape[0]
    res = x.copy() * 0 + array[-1]

    while True:
        if i == len_array:
            break
        I_max, x0, sigma = array[i : i + 3]
        exponent = -((x - x0) ** 2) / (2 * sigma**2)
        res += I_max * np.exp(exponent)
        i += 3
    return res


@jit(nopython=True, inline="always", error_model="numpy")
def gauss_Fe18_C3(
    x: np.array or float,
    IFe: float,
    WFe: float,
    IC: float,
    VC: float,
    WC: float,
    B: float,
) -> np.array or float:
    """
    - Funcction specialized in returning the calculated intensity on a locked CIII to FeXVIII to prevent the FeXVIII that's too dimme in SPICE to climb the carbon peak

    - PS: Intensities in the arguments must match IC,IFe,B to get a logical output

    Args:
        x (np.arrayorfloat): wavelength position to calculate (Angstrums)
        IC (float): Intensity of carbon at the peak
        VC (float): The wavelength of carbon at the peak (Angstrum)
        WC (float): The FWHM of carbon at the peak (Angstrum)
        IFe (float): Intensity of iron at the peak
        WFe (float): The FWHM of iron at the peak (Angstrum)
        B (float):  The Background intensity elevation contribution

    Returns:
        np.array or float: output intensit(y/ies) at the x poistion( /s)
    """
    # The number 2.19 is the theoritical poition shift of FeXVIII
    # to C III, could be different in SPICE depend on how accurate
    # is the calibration slope (the value is right if the slope is
    # = 1)
    expoC = -((x - VC) ** 2) / (2 * WC**2)
    expoFe = -((x - VC + 2.19) ** 2) / (2 * WFe**2)
    result = B + IC * np.exp(expoC) + IFe * np.exp(expoFe)
    return result


@jit(nopython=True, inline="always", error_model="numpy")
def gauss_LyB_Fe10(
    x: np.array or float,
    ILy: float,
    VLy: float,
    WLy: float,
    IFe: float,
    WFe: float,
    B: float,
) -> np.array or float:
    """
    - Funcction specialized in returning the calculated intensity on a locked LyB to FeX to prevent the FeX that's too dimme in SPICE to climb the carbon peak

    - PS: Intensities in the arguments must match IC,IFe,B to get a logical output

    Args:
        x (np.arrayorfloat): wavelength position to calculate (Angstrums)
        ILy (float): Intensity of Lymann Beta at the peak
        VLy (float): The wavelength of Lymann Beta at the peak (Angstrum)
        WLy (float): The FWHM of Lymann Beta at the peak (Angstrum)
        IFe (float): Intensity of iron at the peak
        WFe (float): The FWHM of iron at the peak (Angstrum)
        B (float):  The Background intensity elevation contribution

    Returns:
        np.array or float: output intensit(y/ies) at the x poistion( /s)
    """
    # The number 2.32 is the theoritical poition shift of FeX
    # to LyB, could be different in SPICE depend on how accurate
    # is the calibration slope (the value is right if the
    # slope is = 1)
    expoLy = -((x - VLy) ** 2) / (2 * WLy**2)
    expoFe = -((x - VLy + 2.32) ** 2) / (2 * WFe**2)
    result = B + ILy * np.exp(expoLy) + IFe * np.exp(expoFe)
    return result
