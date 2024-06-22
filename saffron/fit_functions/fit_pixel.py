import numpy as np
from typing import Callable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import datetime
from ..utils import clean_nans
from ..utils.lock_tools import gen_unlocked_params


def fit_pixel(*args, **kwargs):
    # this part will adapt to select the old and the new fast format in the future
    # for now it works with the old algorithm
    return fit_pixel_multi(*args, **kwargs)


def fit_pixel_multi(
    x: np.ndarray,
    y: np.ndarray,
    ini_params: np.ndarray,
    quentities,
    fit_func: Callable,
    bounds: np.ndarray = [np.nan],
    I_bounds: np.ndarray = None,
    x_extent: np.array = None,
    plotit: bool = False,
    weights: str = None,
    verbose=0,
    describe_verbose=False,
    **kwargs,
):
    """Function that Fits a pixel data into a gaussian.

    Args:
        x(np.ndarray): spectrum axis.
        pixel_data (np.ndarray): pixel intensity values as function of wavelength.
        Gauss_ini_params (np.ndarray, shape:4,optional): a list of initiale parameters if not initialized.
        bounds (np.array,np.array): The bounds of the parameters
        plotit (bool, optional): In case you want to plot it. Defaults to False.
        weights (str, optional): string ["I": for a linear weight depend on the value of intensity]. Defaults to None.
    Return:
        coeff (np.ndarray): Fitted parameters.
        var_matrix (np.ndarray): Variation matrix that represent calculated fitting error.
    """
    assert len(ini_params.shape) == 1
    assert x.shape == y.shape
    _s = ini_params.shape[0]

    if callable(bounds):
        bounds = bounds(par=ini_params, que=quentities)
    elif bounds is None or not (False in (np.isnan(bounds))):
        bounds = np.zeros((2, _s))
        for i in range(_s):
            if quentities[i] == "B":
                bounds[:, i] = [-5, 5]
            if quentities[i] == "I":
                if type(I_bounds) != type(None):
                    bounds[:, i] = I_bounds
                else:
                    bounds[:, i] = [0, 1.0e3]
            if quentities[i] == "x":
                if type(x_extent) != type(None):
                    bounds[:, i] = [ini_params[i] - x_extent, ini_params[i] + x_extent]
                else:
                    bounds[:, i] = [ini_params[i] - 1, ini_params[i] + 1]
            if quentities[i] == "s":
                bounds[:, i] = [0.28, 0.8]
        if verbose >= 2:
            print(f"bounds were internally set:\n{bounds}")

    _s = ini_params.shape[0]
    _x, _y, w = clean_nans(x, y, weights)
    if np.any(np.isnan(w)):
        print(f"x     : {_x}\ny     : {_y}\nsigma : {w}")

    if _y.shape[0] <= _s:
        if verbose >= 1:
            print(
                "after cleaning data the number of parameters is greater than data points\nPrecleaning size  = {}\nPostcleaning size = {}nans size         = {}".format(
                    y.shape[0], _y.shape[0], (y[np.isnan(y)]).shape[0]
                )
            )
        return (np.ones((_s)) * np.nan, np.ones((_s, _s)) * np.nan)
    try:
        res = curve_fit(fit_func, _x, _y, p0=ini_params, bounds=bounds, sigma=w)
    except RuntimeError:
        plotit = True if np.random.random() * 1 >= np.inf else False
        if verbose >= 1:
            print(
                "couldn't find the minimum"
                + ("Plotting the result in ./tmp" if plotit else "")
            )

        if verbose >= 2:
            print(
                f"x     : {_x}\ny     : {_y}\ninipar: {ini_params}\nsigma : {w}\nbounds:\n{bounds}"
            )

        res = (np.ones((_s)) * np.nan, np.ones((_s, _s)) * np.nan)

    except:
        if verbose >= 1:
            print("this value is not feasable")
        if verbose >= 2:
            print(
                f"x     : {_x}\ny     : {_y}\nIniPar: {ini_params}\nsigma : {w}\nbounds:\n{bounds}"
            )

        res = (np.ones((_s)) * np.nan, np.ones((_s, _s)) * np.nan)
        plotit = False

    if plotit or verbose >= 3 or verbose == -3:
        fig, axis = plt.subplots(1, 1, figsize=(12, 8))
        axis.plot(
            _x,
            _y,
            # where="mid"
        )
        spectrum_title = "spectrum"
        if "plot_title_prefix" in kwargs.keys():
            spectrum_title = spectrum_title + "\n" + kwargs["plot_title_prefix"]
        axis.set_title(spectrum_title)
        if verbose >= 2:
            print(f"fit_func:\n{fit_func}")

        axis.plot(_x, fit_func(_x, *ini_params), ":", label="initial params")
        xlabel = "wavelength $(\AA)$\n"
        for i in range(len(quentities)):
            if quentities[i] == "I":
                value = f"I: {res[0][i]:.3g}$\pm${np.sqrt(res[1][i,i]):.2g}-"
            if quentities[i] == "B":
                value = f"B: {res[0][i]:.3g}$\pm${np.sqrt(res[1][i,i]):.2g}-"
            if quentities[i] == "x":
                value = f"x: {res[0][i]:.5g}$\pm${np.sqrt(res[1][i,i]):.2g}-"
            if quentities[i] == "s":
                value = f"s: {res[0][i]:.2g}$\pm${np.sqrt(res[1][i,i]):.2g}\n"
            xlabel += value
        axis.semilogy(_x, fit_func(_x, *res[0]), label="fitted params:\n".format())
        if kwargs["lock_protocols"]:
            last_par, locked_quentities = gen_unlocked_params(
                res[0], quentities, kwargs["lock_protocols"]
            )
            locked_quentities = np.array(locked_quentities)
            lbd = last_par[locked_quentities == "x"]
            for i, l in enumerate(lbd):
                plt.axvline(
                    l, label=f"{l:05.1}", ls=":", color=np.random.random(size=(3))
                )
        axis.set_xlabel(xlabel)
        axis.legend(fontsize=12)

        plt.tight_layout()

        plt.savefig(
            "./tmp/{}.jpg".format(
                kwargs["plot_title_prefix"],
                # datetime.datetime.now().strftime("%y%M%d%h%m%S")
            )
        )

        if verbose >= 1:
            print("ini_params", ini_params, "\n", "bounds", bounds)

    return res
