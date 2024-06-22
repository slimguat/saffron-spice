import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

from astropy.io import fits as fits_reader
from astropy.io.fits import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU, ImageHDU
from astropy.wcs import WCS

from ..line_catalog.catalog import LineCatalog
from ..fit_functions import fit_pixel_multi
from ..fit_models import flat_inArg_multiGauss
from ..utils import get_celestial, quickview, get_specaxis


from pathlib import PosixPath, WindowsPath, Path
from typing import Union, List, Dict, Any, Callable, Tuple
import os


from astropy.wcs import FITSFixedWarning
import warnings


class GenInits:
    def __init__(
        self,
        hdulOrPath: Union[str, PosixPath, WindowsPath, HDUList],
        conv_errors: Dict[str, float] = {"I": 0.1, "x": 10**-4, "s": 0.1, "B": 100},
        wvl_interval: Dict[str, slice] = {
            #  "SW": slice(7,-7), "LW": slice(5, -5)},
            "SW": slice(3, -3),
            "LW": slice(3, -3),
        },
        line_catalogue: LineCatalog = None,
        verbose=0,
    ) -> None:
        """Description: This class is used to generate initial parameters for fitting spectral data using Gaussian functions.
        the lines are retrieved automatically from the line catalog and the initial parameters are generated based on the spectral data.

        Args:
            hdulOrPath (Union[str, PosixPath, WindowsPath, HDUList]): Fits HDUList object or path to FITS file.
            conv_errors (dict, optional): Dictionary of convolution errors for different parameters.
                Defaults: {"I": 0.1, "x": 10**-4, "s": 0.1, "B": 100}
            wvl_interval (dict[slice], optional): Wavelength intervals for data processing.
                Defaults: {"low": [7, -7], "high": [5, -5]}
            verbose (int, optional): Verbosity level for printing and plotting. Defaults to 0.
        Returns:
            None
        """
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        self.conv_errors = conv_errors
        self.wvl_interval = wvl_interval
        self.verbose = verbose
        # check if hdulOrPath is a string path or pathlib path or an astropy HDUList
        if type(hdulOrPath) in (str, PosixPath, WindowsPath):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self.hdul = fits_reader.open(hdulOrPath)
            self._path = Path(hdulOrPath)
        else:
            self.hdul = hdulOrPath
            self._path = None

        self.hdulOrPath = hdulOrPath
        self.raster = fits_reader.open(hdulOrPath)

        self.lon, self.lat = get_celestial(self.hdul)
        if line_catalogue is None:
            self.catalog = LineCatalog(verbose=verbose)
        elif isinstance(line_catalogue, LineCatalog):
            self.catalog = line_catalogue
        elif isinstance(line_catalogue, [str, PosixPath, WindowsPath]):
            self.catalog = LineCatalog(verbose=verbose, file_location=line_catalogue)
        else:
            raise ValueError(
                "line_catalogue must be a LineCatalog object or a path to a line catalog file \n else None to use the internal catalogue"
            )
        self.default_lines = self.catalog.get_catalog_lines()

        # these are the parameters to passe
        unq = self.get_extnames(self.hdul)
        self.init_params = [None for i in range(len(unq))]

        self.quentities = [None for i in range(len(unq))]
        self.convolution_threshold = [None for i in range(len(unq))]
        self.windows_lines = {}
        self.global_shift = {
            "SW": 0,
            "LW": 0,
        }  # this is the global shift that is going to be used to shift the lines to the right position compared to the catalog
        pd.options.mode.chained_assignment = None

        # Ignore FITSFixedWarning

    def _gen_inits_window(
        self, hdul_index: int, verbose: int = None, ax=None
    ) -> Dict[str, Any]:
        """
        Description: Generate initial parameters for fitting spectral data using Gaussian functions.
        """
        if verbose is None:
            vb = self.verbose
        else:
            vb = verbose
        if vb >= 3 or vb == -4:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        # initializing the variables and spectral axis
        if True:
            hdu = self.hdul[hdul_index]
            kw = hdu.header["EXTNAME"]
            specaxis = get_specaxis(hdu)
            window_lines = []
            init_param_theory = []
            quentity = []
            cov_th = []
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                specdata = np.nanmean(hdu.data, axis=(0, 2, 3)).astype(float)
            WV = "SW" if np.nanmean(specaxis) < 800 else "LW"
            s = self.wvl_interval[WV]
            specdata = np.array(
                [
                    (
                        specdata[i]
                        if (
                            (s.start if s.start > 0 else len(specdata) - s.start)
                            if s.start is not None
                            else 0
                        )
                        <= i
                        < (
                            (s.stop if s.stop > 0 else len(specdata) - s.stop)
                            if s.stop is not None
                            else len(specdata)
                        )
                        else np.nan
                    )
                    for i in range(len(specdata))
                ]
            )

            if len(specdata[(~np.isnan(specdata))]) <= 2:
                raise (
                    ValueError(
                        "The spectrum is nearly or fully empty wvl_interval is not correct"
                    )
                )
        # starting the search for existing lines in the default windows in the wavelength range provided by the hdu
        if True:
            expension_factor = 0.1
            n_expensions = 20
            for i in range(n_expensions):
                expansion = expension_factor * i
                for ind in range(len(self.default_lines)):
                    line = self.default_lines.iloc[ind]
                    min_wvl = (
                        specaxis[~np.isnan(specdata)][0]
                        - expansion
                        - self.global_shift[WV]
                    )
                    max_wvl = (
                        specaxis[~np.isnan(specdata)][-1]
                        + expansion
                        - self.global_shift[WV]
                    )
                    if min_wvl <= line["wvl"] <= max_wvl:
                        window_lines.append(line)
                        line_index = np.nanargmin(np.abs(specaxis - line["wvl"]))
                        window_lines[-1][
                            "index"
                        ] = line_index  # adding the index of the line in the spectrum
                if len(window_lines) == 0:
                    if vb >= -1:
                        print(
                            f"no lines found in the window {kw} with the global shift {self.global_shift[WV]}"
                        )
                        if i < n_expensions:
                            print(
                                f"expanding the boundaries by from by {expansion+expension_factor} Angstroms"
                            )
                else:
                    break

        # initiating basic numerical values for each parameter on the theoretical position of the line
        if True:
            self.windows_lines[kw] = window_lines
            for line in window_lines:
                init_param_theory.extend(
                    [0.05, line["wvl"] + self.global_shift[WV], 0.35]
                )
                quentity.extend(["I", "x", "s"])
                cov_th.extend(
                    [
                        self.conv_errors["I"],
                        self.conv_errors["x"],
                        self.conv_errors["s"],
                    ]
                )
            if len(window_lines) == 0:
                if vb >= -1:
                    print(f"WARNING: no lines found in the window {kw}")
                window_lines.append({"name": "no_line", "wvl": 0})
                init_param_theory = [np.nan, np.nan, np.nan]
                cov_th.extend(
                    [
                        self.conv_errors["I"],
                        self.conv_errors["x"],
                        self.conv_errors["s"],
                    ]
                )
                quentity.extend(["I", "x", "s"])

            init_param_theory.append(max(0.00, np.nanmean(specdata)))
            init_param_theory = np.array(init_param_theory)

            cov_th.append(self.conv_errors["B"])
            cov_th = np.array(cov_th)
            quentity.append("B")

        # verbose actions
        if True:
            if vb >= 3 or vb < -2:
                ax.step(specaxis, specdata, label=f"original spectrum")
                ax.step(
                    specaxis,
                    flat_inArg_multiGauss(specaxis, *init_param_theory),
                    label=f"predefined params{' NANs' if np.isnan(init_param_theory).any()else ''} $\Delta(\lambda_{{{WV}}})={self.global_shift[WV]:03.2f}$",
                )
            if vb >= 2:
                print(f"_______________________________________\n{kw}:")
                for i, line in enumerate(window_lines):
                    print(
                        f"\t{line['name']:<8} at {line['wvl']:06.1f} in index {line['index']}"
                    )
                print(f"_______________________________________")
            # finding the closest line to the max spectral line and adjust init_params_theory accordingly
            init_param_maxAdjusted, specaxis, specdata = self.find_by_default_window(
                specaxis,
                specdata,
                init_param_theory,
                window_lines,
                verbose=vb,
                catalog=self.catalog,
            )
            if vb >= 3 or vb < -2:
                ax.step(
                    specaxis,
                    flat_inArg_multiGauss(specaxis, *init_param_maxAdjusted),
                    label="default windows adjustement{}".format(
                        " NANs" if np.isnan(init_param_maxAdjusted).any() else ""
                    ),
                )

            # plt.legend()
            # return

        # fitting with one position of the fit and generating an all in all locked init_param_maxAdjusted

        if True:
            if len(init_param_maxAdjusted) > 4:  # fitting with one position of the fit
                # Generating an all in all locked init_param_maxAdjusted
                lock_state = [["free"]]
                line_ref = window_lines[0]
                for i in range(1, len(window_lines)):
                    lock_state.append(
                        ["lock", 0, window_lines[i]["wvl"] - line_ref["wvl"]]
                    )
                # if vb >= 2:
                #     print(f'{kw} lock state')
                #     for i in range(len(lock_state)):
                #         print(window_lines[i]['name'],lock_state[i])

                init_param_locked, lock_quentities = self.gen_lock_params(
                    init_param_maxAdjusted, lock_state, verbose=vb
                )
                func = self.gen_lock_func(init_param_locked, lock_state)
                lock_func = func[list(func.keys())[0]]
                init_param_locked, var = fit_pixel_multi(
                    specaxis,
                    specdata,
                    init_param_locked,
                    quentities=lock_quentities,
                    fit_func=lock_func,
                    plot_title_prefix="preadjust",
                )
                if vb >= 3 or vb < -2:
                    ax.step(
                        specaxis,
                        lock_func(specaxis, *init_param_locked),
                        label="Locked fit{}".format(
                            " NANs" if np.isnan(init_param_locked).any() else ""
                        ),
                    )
                if not np.isnan(init_param_locked).any():
                    init_param_locked = self.get_unlock_params(
                        init_param_locked, lock_state
                    )
                else:
                    if vb >= 0:
                        print(
                            "haven't found the right params after locking using the max adjusted params"
                        )
                    init_param_locked = init_param_maxAdjusted.copy()
            else:
                init_param_locked = init_param_maxAdjusted.copy()

        # fitting the spectrum with the unlocked parameters
        if True:
            init_param_unlocked, var = fit_pixel_multi(
                specaxis,
                specdata,
                np.array(init_param_locked),
                quentities=quentity,
                fit_func=flat_inArg_multiGauss,
                plot_title_prefix="preadjust",
            )
            if not np.isnan(init_param_unlocked).any():
                pass
            else:
                if vb >= 0:
                    print(
                        "haven't found the right params after unlocking using the locked params"
                    )
                init_param_unlocked = init_param_locked.copy()

        # vb actions
        if True:
            dtime = str(datetime.datetime.now())[:19].replace(":", "-")
            if vb >= 3 or vb < -2:
                array_quentity = np.array(quentity)
                for i, l in enumerate(init_param_unlocked[array_quentity == "x"]):
                    ax.axvline(l, label=f"{window_lines[i]['name']},{l:6.1f}", ls=":")
                ax.step(
                    specaxis,
                    flat_inArg_multiGauss(specaxis, *init_param_unlocked),
                    label="Unlocked fit{}".format(
                        " NANs" if np.isnan(init_param_unlocked).any() else ""
                    ),
                )
                ax.legend(fontsize=8, framealpha=0.3)

                if np.abs(vb) >= 4:
                    os.makedirs("./tmp", exist_ok=True)
                    ax.get_figure().savefig(f"./tmp/{dtime}_window{hdul_index}.jpg")

        # saving the parameters
        if True:
            # offset = init_param_maxAdjusted[1::3] - init_param_theory[1::3]
            offset = init_param_unlocked[1::3] - init_param_theory[1::3]
            if self.global_shift[WV] == 0 and not np.isnan(np.nanmean(offset)):
                self.global_shift[WV] = np.nanmean(offset)
            self.init_params[hdul_index] = init_param_unlocked
            self.quentities[hdul_index] = quentity
            self.convolution_threshold[hdul_index] = cov_th

    def gen_inits(self, verbose: int = None):
        if verbose is None:
            vb = self.verbose
        else:
            vb = verbose
        unq = self.get_extnames(self.hdul)
        if vb >= 3 or vb <= -3:
            c = 3
            r = len(unq) // c + (1 if len(unq) % c != 0 else 0)
            fig, axis = plt.subplots(r, c, figsize=(c * 5, r * 5))
            axis = axis.flatten()
        for i in range(len(unq)):
            self._gen_inits_window(
                i,
                verbose=vb if np.abs(vb) <= 3 else 3 * (vb / np.abs(vb)),
                ax=None if abs(vb) < 3 else axis[i],
            )
        if vb >= 4 or vb <= -4:
            os.makedirs("./tmp", exist_ok=True)
            dtime = str(datetime.datetime.now())[:19].replace(":", "-")
            axis[0].get_figure().savefig(f"./tmp/{dtime}.jpg")

            ((fig1, ax1), (fig2, ax2)) = quickview(self.hdul)
            for i, params in enumerate(self.init_params):
                specaxis = get_specaxis(self.hdul[unq[i]])
                ax2[i].step(specaxis, flat_inArg_multiGauss(specaxis, *params))
            fig1.savefig(f"./tmp/{dtime}_window_all.jpg")
            fig2.savefig(f"./tmp/{dtime}_spectrum_all.jpg")

    @staticmethod
    def gen_lock_params(
        init_params: List[float],
        lock_state: List[List[Union[str, float]]],
        verbose: int = 0,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate locked parameters and corresponding quantities based on lock state.

        Args:
            init_params (List[float]): List of initial parameters for the fit.
            lock_state (List[List[Union[str, float]]]): Lock state for each parameter.
                Format: ["lockall", *list_of_distances] or [["free"], ["block", wvl], ["lock", with_line, distance]]
            verbose (int, optional): Verbosity level for printing. Defaults to 0.

        Returns:
            Tuple[np.ndarray, List[str]]: Tuple containing locked parameters and corresponding quantities.
        """
        lock_params = []
        lock_quentities = []
        if lock_state[0] == "lockall":
            # if lock_state[1:]!=(len(init_params)//3)-1: raise ValueError("the lockall parameters doesn't match the number of lines")
            lock_state2 = ["free"]
            for i in range(1, len(init_params) // 3):
                lock_state2.append("lock")
            lock_state = lock_state2
            if verbose >= 2:
                print("new lock_state:\n\t", lock_state)
        if type(lock_state[0]) == list:
            lock_state2 = []
            for state in lock_state:
                lock_state2.append(state[0])
            lock_state = lock_state2

        for i, state in enumerate(lock_state):
            if state == "free":
                lock_params.extend(init_params[i * 3 : (i + 1) * 3])
                lock_quentities.extend(["I", "x", "s"])
            elif state in ["lock", "block"]:
                lock_quentities.extend(["I", "s"])
                lock_params.extend(init_params[i * 3 : (i + 1) * 3 : 2])
            else:
                raise ValueError(f"the value of the state ({state}) is not recognized")
        lock_params.append(init_params[-1])
        lock_quentities.append("B")
        return np.array(lock_params), lock_quentities

    @staticmethod
    def gen_lock_func(
        params: List[float], lock_state: List[List[Union[str, float]]], verbose: int = 0
    ) -> Callable:
        """
        Generate a locked function based on lock state.

        Args:
            params (List[float]): List of parameters for the fit.
            lock_state (List[List[Union[str, float]]]): Lock state for each parameter.
                Format: [["free"], ["block", wvl], ["lock", with_line, distance]]
            verbose (int, optional): Verbosity level for printing. Defaults to 0.

        Returns:
            Callable: Generated locked function.
        """
        actual_params = "\n"
        params_i = 0
        for i_line, state in enumerate(lock_state):
            if state[0] == "free":
                str_prm = f"params[{int(params_i)}],params[{int(params_i+1)}],params[{int(params_i+2)}],\n"
                params_i += 3

            elif state[0] == "block":
                str_prm = (
                    f"params[{int(params_i)}],{state[1]},params[{int(params_i+1)}],\n"
                )
                params_i += 2

            elif state[0] == "lock":
                if lock_state[state[1]][0] in ["block", "lock"]:
                    raise ValueError("the state is linked to locked or a blocked value")
                i_ref_line = state[1]
                i_ref = 0
                for i in range(i_ref_line + 0):
                    # print(i,i_ref)
                    if lock_state[i][0] == "free":
                        i_ref += 3
                    if lock_state[i][0] in ["lock", "block"]:
                        i_ref += 2
                i_ref += 1
                str_prm = f"params[{int(params_i)}],params[{i_ref}]+{state[2]},params[{int(params_i+1)}],"
                params_i += 2

            else:
                raise ValueError(f"lock_state {state[0]} unknown")
            actual_params += str_prm

        actual_params += f"params[{int(len(params)-1)}]"
        func_ext_name = ""
        for state in lock_state:
            func_ext_name += f"_{state[0]}"

        str_function = f"def flat_inArg_multiGauss{func_ext_name}(x,*params):return(flat_inArg_multiGauss(x,{actual_params}))"

        if verbose >= 2:
            print("string function\n", str_function)
        loc = {}
        exec(str_function, globals(), loc)
        return loc

    @staticmethod
    def get_unlock_params(
        lock_params: List[float],
        lock_state: List[List[Union[str, float]]],
        verbose: int = 0,
    ) -> np.ndarray:
        """
        Generate unlocked parameters based on lock state.

        Args:
            lock_params (List[float]): List of locked parameters.
            lock_state (List[List[Union[str, float]]]): Lock state for each parameter.
                Format: [["free"], ["block", wvl], ["lock", with_line, distance]]
            verbose (int, optional): Verbosity level for printing. Defaults to 0.

        Returns:
            np.ndarray: Array of unlocked parameters.
        """
        unlock_param = []
        params_i = 0
        for state in lock_state:
            if state[0] == "free":
                unlock_param.extend(lock_params[params_i : params_i + 3])
                params_i += 3
            elif state[0] == "lock":
                i_ref_line = state[1]
                i_ref = 0
                for i in range(i_ref_line + 0):
                    # print(i,i_ref)
                    if lock_state[i][0] == "free":
                        i_ref += 3
                    if lock_state[i][0] in ["lock", "block"]:
                        i_ref += 2
                i_ref += 1

                unlock_param.append(lock_params[params_i])
                unlock_param.append(lock_params[i_ref] + state[2])
                unlock_param.append(lock_params[params_i + 1])
                params_i += 2
            elif state[0] == "block":
                unlock_param.append(lock_params[params_i])
                unlock_param.append(state[2])
                unlock_param.append(lock_params[params_i + 1])
                params_i += 2
            else:
                raise ValueError(f"the value of the state ({state}) is not recognized")
        unlock_param.append(lock_params[-1])
        return np.array(unlock_param)

    @staticmethod
    def find_by_default_window(
        specaxis: np.ndarray,
        specdata: np.ndarray,
        init_params: np.ndarray,
        window_lines: List[Dict[str, Union[str, float]]],
        catalog: LineCatalog = None,
        verbose: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the closest line to the default position and adjust init_params accordingly.

        Args:
            specaxis (np.ndarray): Spectral axis.
            specdata (np.ndarray): Spectral data.
            init_params (np.ndarray): Initial parameters for the Gaussian fit.
            window_lines (List[Dict[str, Union[str, float]]]): List of lines within the window.
            catalog (LineCatalog, optional) : LineCatalog object containing the line catalog. Defaults to None.
            verbose (int, optional): Verbosity level for printing. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Adjusted init_params, specaxis, and specdata.
        """
        # finding max positions
        # defining line names with their default windows
        lines_names = [i["name"] for i in window_lines]
        lines_names_sorted = sorted((lines_names.copy()))
        if True:
            if catalog is None:
                catalog = LineCatalog(verbose=verbose)
            default_lines = catalog.get_catalog_lines()
            def_win = catalog.get_catalog_windows()
            default_windows = def_win["lines"]
            max_I_names = def_win["max_line"]

        for i in range(len(default_windows)):
            if lines_names_sorted == sorted(default_windows[i]):
                I_max_line_name = max_I_names[i]
                I_max_line_index = lines_names.index(I_max_line_name)
                init_params2 = init_params.copy()
                l = len(init_params) - 1
                mod_specdata = specdata.copy()
                line_index = np.nanargmax(mod_specdata)
                if "ne_viii_1" in lines_names_sorted and len(lines_names_sorted) <= 5:
                    ne_viii_wvl = 770.42
                    line_num = np.nanargmin(np.abs(init_params[1::3] - ne_viii_wvl))
                    line_index = np.nanargmin(
                        np.abs(init_params[line_num * 3 + 1] - specaxis)
                    )
                    pass
                x_max_I = specaxis[line_index]
                target_wvl = x_max_I
                current_wvl = init_params[3 * I_max_line_index + 1]
                if abs(target_wvl - current_wvl) > 1.5:
                    if verbose >= -1:
                        print(
                            f" --Warning--\nThe current line is far from the target line\n\ttarget ={target_wvl}{I_max_line_name}\n\tcurrent:{current_wvl}\n\t\lines{window_lines}\ntrying to confine the boundaries"
                        )

                        print(
                            f"between {current_wvl-1:06.1f} Angstroms and {current_wvl+1:06.1f} Angstroms"
                        )
                    delta_neigbours = 1
                    mod_specdata_neigbors = mod_specdata.copy() * np.nan
                    mod_specdata_neigbors[
                        np.argmin(
                            np.abs(specaxis - (current_wvl - delta_neigbours))
                        ) : np.argmin(
                            np.abs(specaxis - (current_wvl + delta_neigbours))
                        )
                    ] = mod_specdata[
                        np.nanargmin(
                            np.abs(specaxis - (current_wvl - delta_neigbours))
                        ) : np.nanargmin(
                            np.abs(specaxis - (current_wvl + delta_neigbours))
                        )
                    ]
                    line_index = np.nanargmax(mod_specdata_neigbors)
                    # if "ne_viii_1" in lines_names_sorted and len(lines_names_sorted)<=5:
                    #     ne_viii_wvl = 770.42
                    #     line_num = np.nanargmin(np.abs(init_params[1::3]-ne_viii_wvl))
                    #     line_index = np.nanargmin(
                    #         np.abs(init_params[line_num*3+1]-specaxis))
                    #     pass
                    x_max_I = specaxis[line_index]
                    target_wvl = x_max_I

                current_wvl = init_params[3 * I_max_line_index + 1]

                # Yoopy found the exact position of the line
                init_params2[1:l:3] += target_wvl - current_wvl
                for j in range(0, len(init_params) // 3):
                    I_max_line_index2 = np.nanargmin(
                        np.abs(specaxis - init_params2[j * 3 + 1])
                    )
                    new_I_max = np.nanmax(
                        specdata[
                            max(0, I_max_line_index2 - 2) : min(
                                len(mod_specdata), I_max_line_index2 + 2
                            )
                        ]
                    )
                    init_params2[j * 3] = new_I_max - init_params2[-1]
                init_params2[np.isnan(init_params2)] = 0
                if True:
                    maxB = np.nanmean(
                        sorted(specdata[(~np.isnan(specdata))])[
                            : max(5, int(0.05 * len(specdata)))
                        ]
                    )
                    init_params2[-1] = maxB

                    for ind in range(0, len(init_params) // 3):
                        wavelength = init_params2[ind * 3 + 1]
                        index = np.nanargmin(np.abs(specaxis - wavelength))
                        init_params2[ind * 3] = (
                            init_params2[j * 3]
                            if np.isnan(specdata[index])
                            else specdata[index]
                        )
                        # print(f"line {ind} with index {index:02d} at {wavelength} has the intensity {init_params2[ind*3]}")
                    init_params2[0:l:3] = init_params2[0:l:3] - init_params2[-1]
                    # print(init_params2[0:l:3])
                init_params2[0:l:3][init_params2[0:l:3] <= 0] = 0.05 - init_params2[-1]
                for i in range(len(init_params2[::3])):
                    if init_params2[i * 3] <= 0:
                        init_params2[i * 3] = 0
                if np.any((init_params2 < 0)):
                    raise ValueError(
                        "The initial parameters are not correct\n{init_params2}"
                    )
                return init_params2, specaxis, specdata

        if verbose > -1:
            print(
                f"The window you have chosen is not in the catalog\n\tLOG!\n\t\t:\n\t\tspecaxis    {specaxis    }\n\t\tspecdata    {specdata    }\n\t\tinit_params {init_params }\n\t\twindow_lines{window_lines}"
            )
        return (init_params * np.nan, specaxis, specdata)

    @staticmethod
    def get_extnames(hdul: HDUList) -> List[str]:
        """
        Get a list of unique extension names from an HDUList, excluding specific extension names.

        Args:
            hdul (HDUList): An astropy HDUList object.

        Returns:
            List[str]: A list of unique extension names.
        """
        unq = [
            hdu.header["EXTNAME"]
            for hdu in hdul
            if hdu.header["EXTNAME"]
            not in ["VARIABLE_KEYWORDS", "WCSDVARR", "WCSDVARR"]
        ]
        return unq

    @property
    def path(self):
        if self._path is not None:
            return Path(self.hdulOrPath)
        else:
            raise ValueError("The path is not defined")
