import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

from astropy.io import fits as fits_reader
from astropy.io.fits import HDUList
from astropy.io.fits.hdu.image import PrimaryHDU, ImageHDU
from astropy.wcs import WCS
import astropy.units as u

from ..line_catalog.catalog import LineCatalog
from ..fit_functions import fit_pixel_model
# from ..fit_models import flat_inArg_multiGauss
from ..fit_models.Model import ModelFactory
from ..utils import get_celestial, quickview, get_specaxis, get_extnames

from pathlib import PosixPath, WindowsPath, Path
from typing import Union, List, Dict, Any, Callable, Tuple, Iterable
import os

from astropy.wcs import FITSFixedWarning
import warnings


class GenInits:
    def __init__(
        self,
        hdulOrPath   : Union[str, PosixPath, WindowsPath, HDUList],
        conv_errors  : Dict[str, float] = {"I": 0.1, "x": 10**-4, "s": 0.1, "B": 100},
        wvl_interval : float = 0.4, 
        line_catalogue: LineCatalog | str | PosixPath | WindowsPath  = None,
        verbose=0,
    ) -> None:
        """Description: This class is used to generate initial parameters for fitting spectral data using Gaussian functions.
        the lines are retrieved automatically from the line catalog and the initial parameters are generated based on the spectral data.

        Args:
            hdulOrPath (Union[str, PosixPath, WindowsPath, HDUList]): Fits HDUList object or path to FITS file.
            conv_errors (dict, optional): Dictionary of convolution errors for different parameters.
                Defaults: {"I": 0.1, "x": 10**-4, "s": 0.1, "B": 100}
            wvl_interval : minimum ratio of the number of non-nan pixel compared to the max number along the wavelength ax to accept as initdata to use to generate the initial parameters.
                Defaults: 0.4
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

        self.lon, self.lat = get_celestial(self.hdul)
        if line_catalogue is None:
            self.catalog = LineCatalog(verbose=verbose)
        elif isinstance(line_catalogue, LineCatalog):
            self.catalog = line_catalogue
        elif isinstance(line_catalogue, (str, PosixPath, WindowsPath)):
            self.catalog = LineCatalog(verbose=verbose, file_location=line_catalogue)
        else:
            raise ValueError(
                "line_catalogue must be a LineCatalog object or a path to a line catalog JSON file \nelse None to use the internal catalogue"
            )
        self.default_lines = self.catalog.get_catalog_lines()

        # These are the parameters to passe
        unq = get_extnames(self.hdul)
        # self.init_params = [None for i in range(len(unq))]

        # self.quentities = [None for i in range(len(unq))]
        self.convolution_threshold = [None for i in range(len(unq))]
        self.windows_lines = {}
        self.global_shift = {
            "SW": 0,
            "LW": 0,
        }  # this is the global shift that is going to be used to shift the lines to the right position compared to the catalog
        pd.options.mode.chained_assignment = None
        self.Models = [None for i in range(len(unq))] 
        # Ignore FITSFixedWarning

    def gen_inits_window(
        self, hdul_index: int, 
        verbose: int = None, 
        ax=None,
        Model_args = {'jit_activated':False},
        extend_wvl_search = 0.5*u.Angstrom
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
            # s = self.wvl_interval[WV]
            s = _get_lims_per_percentile(hdu.data,self.wvl_interval)
            s = slice(s[0],s[1])
            
            specdata2 = np.empty(len(specdata)) *np.nan
            specdata2[s] = specdata[s]
            specdata = specdata2.copy() 
            if len(specdata[(~np.isnan(specdata))]) <= 2:
                raise (
                    ValueError(
                        "The spectrum is nearly or fully empty wvl_interval is not correct"
                    )
                )
        
        # finding the lines in the interval of the spectral window 
        if True:
            extend_pxl_search = (
                extend_wvl_search.to(u.Angstrom).value*len(specaxis)/(
                    np.max(specaxis)-np.min(specaxis)
                    )
                )
            # print("pixel search extension",extend_pxl_search)
            expension_factor = 0.1
            n_expensions = 10
            for i in range(n_expensions):
                expansion = expension_factor * i
                for ind in range(len(self.default_lines)):
                    line = self.default_lines.iloc[ind]
                    min_wvl = (
                        specaxis[~np.isnan(specdata)][0]
                        - expansion
                        - self.global_shift[WV]
                        - extend_wvl_search.to(u.Angstrom).value
                    )
                    #TODO Delete this line
                    # strict_min_wvl = min_wvl + extend_pxl_search.to(u.Angstrom).value
                    max_wvl = (
                        specaxis[~np.isnan(specdata)][-1]
                        + expansion
                        - self.global_shift[WV]
                        + extend_wvl_search.to(u.Angstrom).value
                    )
                    #TODO Delete this line
                    # strict_max_wvl = max_wvl - extend_pxl_search.to(u.Angstrom).value
                    if min_wvl <= line["wvl"] <= max_wvl:
                        window_lines.append(line)
                        line_index = np.nanargmin(np.abs(specaxis - line["wvl"]))
                        window_lines[-1][
                            "index"
                        ] = line_index  # adding the index of the line in the spectrum
                        # TODO Delete this line
                        # if strict_min_wvl <= line["wvl"] <= strict_max_wvl:
                        #     window_lines[-1]["strict"] = True
                        
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
            Free_params_Model = ModelFactory(**Model_args)
            for line in window_lines:
                init_param_theory.extend(
                    [0.05, line["wvl"] + self.global_shift[WV], 0.35]
                )
                
                # quentity.extend(["I", "x", "s"])
                cov_th.extend(
                    [
                        self.conv_errors["I"],
                        self.conv_errors["x"],
                        self.conv_errors["s"],
                    ]
                )
                Free_params_Model.add_gaussian(
                    0.05, 
                    line["wvl"] , 
                    0.35,
                    name = [line["name"],line["wvl"]] )
                
            if len(window_lines) == 0:
                #if no line in window_line now it raises an error
                raise ValueError(f"\033[93mWARNING: no lines found in the window {kw}\033[0m")

            init_param_theory.append(max(0.00, np.nanmean(specdata)))
            init_param_theory = np.array(init_param_theory)
            
            cov_th.append(self.conv_errors["B"])
            cov_th = np.array(cov_th)
            # quentity.append("B")
            
            Free_params_Model.add_polynome(max(0.00, np.nanmean(specdata)),name = "Bg",lims =[specaxis[0],specaxis[-1]+(specaxis[-1]-specaxis[-2])])
            wvl_bounds = 0.1 if hdul_index != 0 else 1
            Free_params_Model.set_bounds(
                {"I": [0, np.nanmax(specdata)*1.5], 
                 "x": [["ref-add", -wvl_bounds], ["ref-add", wvl_bounds]], 
                 "s": [0.20, 0.6], "B": [-10, 10]
                 }
            )
            
            
        # verbose actions
        if True:
            if vb >= 3 or vb < -2:
                ax.step(specaxis, np.nanmean(hdu.data, axis=(0, 2, 3)).astype(float), label=f"original spectrum",color='black',ls=":")
                ax.step(specaxis, specdata2, color='black')
                ax.step(
                    specaxis,
                    Free_params_Model.callables['function'](specaxis, *init_param_theory),
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
        if True:    
            init_param_maxAdjusted, specaxis, specdata = self.find_by_default_window(
                specaxis,
                specdata,
                Free_params_Model,
                verbose=vb,
                catalog=self.catalog,
            )
            maxAdjusted_params_Model = Free_params_Model.copy()
            if not np.any(np.isnan(init_param_maxAdjusted)):
                maxAdjusted_params_Model.set_unlock_params(init_param_maxAdjusted)
            if vb >= 3 or vb < -2:
                ax.step(
                    specaxis,
                    Free_params_Model.callables['function'](specaxis, *init_param_maxAdjusted),
                    label="default windows adjustement{}".format(
                        " NANs" if np.isnan(init_param_maxAdjusted).any() else ""
                    ),
                )

        # fitting with one position of the fit and generating an all in all locked init_param_maxAdjusted
        
        if True:
            #TODO: check this later
            if len(init_param_maxAdjusted) > 4:  # fitting with one position of the fit
                all_locked_params_Model = Free_params_Model.copy()
                all_locked_params_Model.set_unlock_params(init_param_maxAdjusted)
                for gauss_i in range(1,len(window_lines)):
                    diff = Free_params_Model.functions['gaussian'][0]['x']-Free_params_Model.functions['gaussian'][gauss_i]['x']
                    all_locked_params_Model.lock(
                        {'model_type':'gaussian','element_index':gauss_i,'parameter':'x'},
                        {'model_type':'gaussian','element_index':0,'parameter':'x'},
                        {'operation':'add','value':-diff}
                    )
                    all_locked_params_Model.lock(
                        {'model_type':'gaussian','element_index':gauss_i,'parameter':'s'},
                        {'model_type':'gaussian','element_index':0,'parameter':'s'},
                        {'operation':'add','value':0}
                    )
                
                fit_func = all_locked_params_Model.callables['function']
                
                init_param_locked, var = fit_pixel_model(
                    specaxis,
                    specdata,
                    all_locked_params_Model,
                    plot_title_prefix="preadjust",
                )
                all_locked_params_Model.set_lock_params(init_param_locked)
                
                if vb >= 3 or vb < -2:
                    ax.step(
                        specaxis,
                        fit_func(specaxis, *init_param_locked),
                        label="Locked fit{}".format(
                            " NANs" if np.isnan(init_param_locked).any() else ""
                        ),
                    )
                if not np.isnan(init_param_locked).any():
                    init_param_locked = all_locked_params_Model.get_unlock_params(init_param_locked)
                else:
                    if vb >= 0:
                        print(
                            "haven't found the right params after locking using the max adjusted params"
                        )
                    init_param_locked = init_param_maxAdjusted.copy()
            else:
                all_locked_params_Model = Free_params_Model.copy()
                init_param_locked = init_param_maxAdjusted.copy()
                all_locked_params_Model.set_unlock_params(init_param_locked)

        # fitting the spectrum with the unlocked parameters
        if True:
            all_unlocked_Model = Free_params_Model.copy()
            all_unlocked_Model.set_unlock_params(init_param_maxAdjusted.copy())
            
            init_param_unlocked, var = fit_pixel_model(
                specaxis,
                specdata,
                all_unlocked_Model,
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
            all_unlocked_Model.set_unlock_params(init_param_unlocked)
        
        # vb actions
        if True:
            dtime = str(datetime.datetime.now())[:19].replace(":", "-")
            if vb >= 3 or vb < -2:
                array_quentity = all_unlocked_Model.get_unlock_quentities()
                
                for i, l in enumerate(init_param_unlocked[array_quentity == "x"]):
                    ax.axvline(l, label=f"{window_lines[i]['name']},{l:6.1f}", ls=":")
                try:
                    all_unlocked_Model.callables['function'](specaxis, *init_param_unlocked)
                except:
                    all_unlocked_Model.reset_callables(regenerate_file=True)
                ax.step(
                    specaxis,
                    all_unlocked_Model.callables['function'](specaxis, *init_param_unlocked),
                    label="Unlocked fit{}".format(
                        " NANs" if np.isnan(init_param_unlocked).any() else ""
                    ),
                )
                # except Exception as e:
                #     print(e)
                #     print("problem with the plot")
                #     print(all_unlocked_Model)
                #     print(init_param_unlocked)
                #     raise e

                ax.legend(fontsize=8, framealpha=0.3)

                if np.abs(vb) >= 4:
                    os.makedirs("./tmp", exist_ok=True)
                    try:
                        ax.get_figure().savefig(f"./tmp/{dtime}_window{hdul_index}.jpg")
                    except:
                        print("problem saving figure")
        # saving the parameters
        if True:
            # offset = init_param_maxAdjusted[1::3] - init_param_theory[1::3]
            offset = init_param_unlocked[1::3] - init_param_theory[1::3]
            if self.global_shift[WV] == 0 and not np.isnan(np.nanmean(offset)):
                self.global_shift[WV] = np.nanmean(offset)
            
            all_unlocked_Model.set_bounds(
                kwargs = {
                    "I": [0, 1000], 
                    "x": [["ref-add", -1], ["ref-add", 1]], 
                    "s": [0.20, 0.6], 
                    "B": [-10, 10]
                    }
            )
            toDelete_model = ModelFactory(jit_activated=True)
            all_unlocked_Model._function_string_template = toDelete_model._function_string_template
            all_unlocked_Model._jacobian_string_template = toDelete_model._jacobian_string_template
            all_unlocked_Model.gen_fit_function()
            self.Models[hdul_index] = all_unlocked_Model
            
            # self.init_params[hdul_index] = init_param_unlocked
            # self.quentities[hdul_index] = quentity
            self.convolution_threshold[hdul_index] = cov_th

    def gen_inits(self, extend_wvl_search=0.5*u.Angstrom,verbose: int = None):
        if verbose is None:
            vb = self.verbose
        else:
            vb = verbose
        unq = get_extnames(self.hdul)
        if vb >= 3 or vb <= -3:
            c = 3
            r = len(unq) // c + (1 if len(unq) % c != 0 else 0)
            fig, axis = plt.subplots(r, c, figsize=(c * 5, r * 5))
            axis = axis.flatten()
        try: list(extend_wvl_search)
        except: extend_wvl_search = [extend_wvl_search]*len(unq)
        
        for i in range(len(unq)):
            self.gen_inits_window(
                i,
                extend_wvl_search = extend_wvl_search[i],
                verbose=vb if np.abs(vb) <= 3 else 3 * (vb / np.abs(vb)),
                ax=None if abs(vb) < 3 else axis[i],
            )
        if vb >= 4 or vb <= -4:
            os.makedirs("./tmp", exist_ok=True)
            dtime = str(datetime.datetime.now())[:19].replace(":", "-")
            try: axis[0].get_figure().savefig(f"./tmp/{dtime}.jpg")
            except Exception as e: 
                print(e)
                print('problem saving figure')
            self.plot_overview(verbose=vb)

    def plot_overview(self,verbose: int = None):
        if verbose is None:
            vb = self.verbose
        else:
            vb = verbose
        unq = get_extnames(self.hdul)
            
        os.makedirs("./tmp", exist_ok=True)
        dtime = str(datetime.datetime.now())[:19].replace(":", "-")

        ((fig1, ax1), (fig2, ax2)) = quickview(self.hdul)
        ax2 = ax2.flatten()
        for i, params in enumerate([model.get_unlock_params() for model in self.Models]):
            specaxis = get_specaxis(self.hdul[unq[i]])
            ax2[i].step(specaxis, self.Models[i].callables['function'](specaxis, *params))
            array_quentity = self.Models[i].get_unlock_quentities()
            for j, l in enumerate(params[array_quentity == "x"]):
                ax2[i].axvline(l, ls=":")
        try: 
            if vb >= 4 or vb <= -4:
                fig1.savefig(f"./tmp/{dtime}_window_all.jpg")
                fig2.savefig(f"./tmp/{dtime}_spectrum_all.jpg")
        except Exception as e:
            print(e) 
            print('problem saving figure')

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
        # init_params: np.ndarray,
        # window_lines: List[Dict[str, Union[str, float]]],
        Model: ModelFactory ,
        catalog: LineCatalog= None,
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
        if catalog is None:
                catalog = LineCatalog(verbose=verbose)
        # finding max positions
        # defining line names with their default windows
        #BEAWARE the sub** are inverted with no sub** to only anlysze the lines that are strictly inside the window
        sub_Model = Model.copy()
        sub_functions = Model.functions.copy()
        sub_window_lines = Model.functions_names['gaussian']
        no_nan_specaxis = specaxis[(~np.isnan(specdata)) & ~np.isnan(specaxis)]
        min_val = np.nanmin(no_nan_specaxis)
        max_val = np.nanmax(no_nan_specaxis)
        is_strict = np.array([True if min_val <= float(wvl)  <= max_val else False for wvl in np.array(sub_window_lines)[:,1]])
        functions = sub_functions.copy()
        functions['gaussian'] = [sub_functions['gaussian'][i] for i in range(len(sub_functions['gaussian'])) if is_strict[i]]
        functions['gaussian'] = {i:functions['gaussian'][i] for i in range(len(functions['gaussian']))}
        window_lines = Model.functions_names['gaussian']
        
        window_lines = [sub_window_lines[i] for i in range(len(sub_window_lines)) if is_strict[i]]
        Model = ModelFactory(jit_activated=False,functions=functions,functions_names=window_lines)
        init_params = Model.get_unlock_params()
        lines_IDs = catalog.line2ID(list_names=window_lines)
        if verbose >= 1:
            print(f"lines strictly in the window {window_lines}")
            print(f'all lines in the window {sub_window_lines}')
        
        if True:
            def_win = catalog.get_catalog_windows()
            default_windows = def_win["lines"]
            max_I_names = def_win["max_line"]

        for i in range(len(default_windows)):
            if set(lines_IDs) == set(default_windows[i]):
                I_max_line_name = max_I_names[i]
                I_max_line_index = lines_IDs.index(I_max_line_name)
                init_params2 = init_params.copy()
                l = len(init_params) - 1
                mod_specdata = specdata.copy()
                line_index = np.nanargmax(mod_specdata)
                
                x_max_I     = specaxis[line_index]
                target_wvl  = x_max_I
                current_wvl = init_params[3 * I_max_line_index + 1]
                if abs(target_wvl - current_wvl) > 1.5:
                    if verbose >= -1:
                        print(
                            f" --Warning--\nThe current line is far from the target line\n\ttarget ={target_wvl}{I_max_line_name}\n\tcurrent:{current_wvl}\n\t\\lines{window_lines}\ntrying to confine the boundaries"
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
                
                # Joining the strict with the no strict lines
                for sub_ind,line in enumerate(sub_window_lines):
                    if line in window_lines:
                        ind = window_lines.index(line)
                        sub_Model.functions['gaussian'][sub_ind]['I'] = init_params2[ind*3]
                        sub_Model.functions['gaussian'][sub_ind]['x'] = init_params2[ind*3+1]
                        sub_Model.functions['gaussian'][sub_ind]['s'] = init_params2[ind*3+2]
                    else:
                        sub_Model.functions["gaussian"][sub_ind]['I'] =  0
                        sub_Model.functions["gaussian"][sub_ind]['x'] += target_wvl - current_wvl
                        sub_Model.functions["gaussian"][sub_ind]['s'] =  0.35
                init_params2 = sub_Model.get_unlock_params()
                return init_params2, specaxis, specdata

        if verbose > -1:
            print(
                f"The window you have chosen is not in the catalog\n\tLOG\n\t\tinit_params {init_params }\n\t\twindow_lines{window_lines}"
            )
        
        return (sub_Model.get_unlock_params(), specaxis, specdata)


def _get_lims_per_percentile(data,percentile,verbose=0):
  lbd_stat_N = np.empty(data.shape[1])
  for i in range(len(lbd_stat_N)):
      lbd_stat_N[i] = np.where(~np.isnan(data[:,i,:,:]))[0].shape[0]
  lbd_stat_N /= np.nanmax(lbd_stat_N)
  
  min_perc  = np.argmin((lbd_stat_N[:len(lbd_stat_N)//2]-percentile)**2)
  max_perc  = -np.argmin((lbd_stat_N[:len(lbd_stat_N)//2:-1]-percentile)**2)-1
  
  if verbose>=3 or verbose<=-3:
    plt.figure()
    plt.plot(lbd_stat_N)
    plt.axhline(percentile)
    ax = plt.gca()
    twinx = ax.twinx()
    twinx .plot(np.nanmean(data, axis=(0,2,3)))
    ax.axvline(min_perc                    ,
      color='r',ls="--")
    ax.axvline(len(lbd_stat_N) - min_perc-1,
      color='r',ls="--")
  return min_perc,len(lbd_stat_N) - min_perc
