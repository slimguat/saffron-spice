import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

# import pickle
import roman
import inspect
from time import sleep
import multiprocessing as mp
from multiprocessing import Process, Lock
from pathlib import PosixPath, WindowsPath, Path
import datetime
import pickle
from rich.progress import Progress
from rich.console import Console
from colorama import init, Fore, Style

init()

from typing import Union, List, Dict, Any, Callable, Tuple, Optional, Iterable
from astropy.visualization import (
    SqrtStretch,
    AsymmetricPercentileInterval,
    ImageNormalize,
)
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.wcs import WCS
from ndcube import NDCollection

# from spice_uncertainties import spice_error
from sospice import spice_error

# from sunraster.instr.spice import read_spice_l2_fits

from .fit_pixel import fit_pixel as fit_pixel_multi
from ..fit_models import flat_inArg_multiGauss
from ..line_catalog.catalog import LineCatalog
from ..utils.denoise import denoise_data
from ..utils.despike import despike
from ..utils.utils import (
    gen_shmm,
    Preclean,
    Preclean,
    convolve,
    get_specaxis,
    flatten,
    find_nth_occurrence,
    ArrToCode,
)
from ..utils.lock_tools import (
    gen_locked_params,
    gen_unlocked_params,
    gen_lock_fit_func,
    LockProtocols,
)


class RasterFit:
    def __repr__(self) -> str:
        value = (
            # "init_params             "+ str(self.init_params)             +  "\n"+
            # "quentities              "+ str(self.quentities)              +  "\n"+
            "fit_func                "
            + str(self.fit_func)
            + "\n"
            + "convolution_extent_list "
            + str(self.convolution_extent_list)
            + "\n"
            + "weights                 "
            + str(self.weights)
            + "\n"
            + "denoise                 "
            + str(self.denoise)
            + "\n"
            + "preclean                "
            + str(self.preclean)
            + "\n"
            + "save_data               "
            + str(self.save_data)
            + "\n"
            + "data_filename           "
            + str(self.data_filename)
            + "\n"
            + "data_save_dir           "
            + str(self.data_save_dir)
            + "\n"
            + "Jobs                    "
            + str(self.Jobs)
            + "\n"
            + "verbose                 "
            + str(self.verbose)
            + "\n"
            + "len(windows)            "
            + str(len(self.windows))
            + "\n"
            + "len(fused_windows)      "
            + str(len(self.fused_windows))
            + "\n"
            + "window_size             "
            + str(self.window_size)
            + "\n"
            + "init_params/quentities  "
            + "\n"
            + "\n\t".join(
                [
                    str(self.init_params[i]) + "\n" + str(self.quentities[i])
                    for i in range(len(self.init_params))
                ]
            )
        )
        return value

    def __init__(
        self,
        path_or_hdul,
        init_params: list,
        quentities: list,
        fit_func: callable,
        windows_names: list = None,
        bounds: np.array = np.array([np.nan]),
        window_size: np.ndarray = np.array([[500, 510], [60, 70]]),
        convolution_function: callable = lambda lst: np.zeros_like(lst[:, 2]) + 1,
        convolution_threshold: float = np.array([0.1, 10**-4, 0.1, 100]),
        convolution_extent_list: np.array = np.array([0, 1, 2, 3, 4, 5]),
        mode: str = "box",
        weights: bool = True,
        denoise: bool = True,
        despike: bool = True,
        convolute: bool = True,
        denoise_intervals: list = [6, 2, 1, 0, 0],
        clipping_sigma: float = 2.5,
        clipping_med_size: list = [6, 3, 3],
        clipping_iterations: int = 3,
        preclean: bool = True,
        save_data: bool = True,
        data_filename: str = "NoName.fits",
        data_save_dir: str = "./fits/",
        Jobs: int = 1,
        verbose: int = 0,
    ):
        self.path_or_hdul = path_or_hdul
        self.init_params = init_params
        self.quentities = quentities
        self.fit_func = fit_func
        self.windows_names = windows_names
        self.bounds = bounds
        self.window_size = window_size
        self.convolution_function = convolution_function
        self.convolution_threshold = convolution_threshold
        self.convolution_extent_list = convolution_extent_list
        self.mode = mode
        self.weights = weights
        self.denoise = denoise
        self.despike = despike
        self.convolute = convolute
        self.denoise_intervals = denoise_intervals
        self.clipping_sigma = clipping_sigma
        self.clipping_med_size = clipping_med_size
        self.clipping_iterations = clipping_iterations
        self.preclean = preclean
        self.save_data = save_data
        self.data_filename = data_filename
        self.filenames_generator
        self.data_save_dir = data_save_dir
        self.Jobs = Jobs
        self.verbose = verbose
        self.lock = Lock()

        self.L2_path = ""
        self.raster = None
        self.load_data()
        self.headers = [
            self.raster[i].header
            for i in range(len(self.raster))
            if self.raster[i].header["EXTNAME"]
            not in ["VARIABLE_KEYWORDS", "WCSDVARR", "WCSDVARR"]
        ]

        self.windows = []
        self.gen_windows()
        self.solo_windows_toFit = np.arange(len(self.windows))
        self.fused_windows = []
        if verbose <= -2:
            warnings.filterwarnings("ignore")
        tmp_dir = Path(r".\tmp")
        if not tmp_dir.exists():
            os.mkdir(tmp_dir)

        return None

    def fuse_windows(self, *indices):
        self.fused_windows.append(
            WindowFit(
                hdu=[self.windows[i].hdu for i in indices],
                init_params=[self.windows[i].init_params for i in indices],
                quentities=[self.windows[i].quentities for i in indices],
                window_names=[self.windows[i].window_names for i in indices],
                fit_func=self.windows[0].fit_func,
                bounds=self.bounds,
                window_size=self.window_size,
                convolution_function=self.convolution_function,
                convolution_threshold=(
                    self.convolution_threshold
                    if not isinstance(self.convolution_threshold[0], Iterable)
                    else [self.convolution_threshold[i] for i in indices]
                ),
                convolution_extent_list=self.convolution_extent_list,
                mode=self.mode,
                weights=self.weights,
                denoise=self.denoise,
                despike=self.despike,
                convolute=self.convolute,
                denoise_intervals=self.denoise_intervals,
                clipping_sigma=self.clipping_sigma,
                clipping_med_size=self.clipping_med_size,
                clipping_iterations=self.clipping_iterations,
                preclean=self.preclean,
                save_data=self.save_data,
                data_filename=self.data_filename,
                data_save_dir=self.data_save_dir,
                Jobs=self.Jobs,
                verbose=self.verbose,
                lock=self.lock,
            )
        )
        x = self.solo_windows_toFit
        y = indices
        self.solo_windows_toFit = [value for value in x if value not in y]

    def gen_windows(self):
        if self.windows_names is None:
            self.windows_names = [None for i in self.raster]

        elif isinstance(self.windows_names, dict):
            self.windows_names = [
                self.windows_names[rast.header["EXTNAME"]] for rast in self.raster
            ]

        else:
            raise SyntaxError(
                f"Please revise the windows_names argument {self.windows_names}"
            )

        for i in range(len(self.raster)):
            window = WindowFit(
                hdu=self.raster[i],
                init_params=self.init_params[i],
                quentities=self.quentities[i],
                fit_func=self.fit_func,
                bounds=self.bounds,
                window_size=self.window_size,
                window_names=self.windows_names[i],
                convolution_function=self.convolution_function,
                convolution_threshold=(
                    self.convolution_threshold
                    if not isinstance(self.convolution_threshold[0], Iterable)
                    else self.convolution_threshold[i]
                ),
                convolution_extent_list=self.convolution_extent_list,
                mode=self.mode,
                weights=self.weights,
                denoise=self.denoise,
                despike=self.despike,
                convolute=self.convolute,
                denoise_intervals=self.denoise_intervals,
                clipping_sigma=self.clipping_sigma,
                clipping_med_size=self.clipping_med_size,
                clipping_iterations=self.clipping_iterations,
                preclean=self.preclean,
                save_data=self.save_data,
                data_filename=self.data_filename,
                data_save_dir=self.data_save_dir,
                Jobs=self.Jobs,
                verbose=self.verbose,
                lock=self.lock,
            )
            self.windows.append(window)

    def run_preparations(self, redo=False):
        for i in range(len(self.windows)):
            self.windows[i].run_preparations(redo=redo)
        for i in range(len(self.fused_windows)):
            self.fused_windows[i].run_preparations(redo=redo)

    def fit_raster(self, progress_follower=None):
        if progress_follower is None:
            progress_follower = ProgressFollower()
        for ind in self.solo_windows_toFit:
            self.windows[ind].fit_window(progress_follower=progress_follower)
            self.windows[ind].write_data()
        for window in self.fused_windows:
            window.fit_window(progress_follower=progress_follower)
            window.write_data()

    def write_data(self):
        for ind in self.solo_windows_toFit:
            self.windows[ind].write_data()
        for window in self.fused_windows:
            self.windows[ind].write_data()

    def check_object_arguments(self):
        if self.verbose > 0:
            print("checking adequacy of given parameters")
        if type(self.quentities[0]) != list:
            print(
                f"ERROR queities sould be a list of lists\n quentities given {self.quentities}"
            )
        for i in range(len(self.init_params)):
            if len(self.init_params[i]) != len(self.quentities[i]):
                print(
                    f"initial parameters provided are not aligned with the quentities\nindex: {i}\ninit parms: {self.init_params[i]}\nquentities: {self.quentities}"
                )
        if self.verbose > 0:
            print("passed tests: the parameters are initially right")

    def load_data(self):
        if self.verbose > 1:
            print("reading data")
        if type(self.path_or_hdul) in (str, PosixPath, WindowsPath):
            self.L2_path = self.path_or_hdul
            if self.verbose > 1:
                print(f"data is given as path:  {self.path_or_hdul  }")
            self.raster = fits.open(self.path_or_hdul)
            self.raster = [
                self.raster[i]
                for i in range(len(self.raster))
                if self.raster[i].header["EXTNAME"]
                not in ["VARIABLE_KEYWORDS", "WCSDVARR", "WCSDVARR"]
            ]
        elif isinstance(self.path_or_hdul, HDUList):
            self.raster = self.path_or_hdul
        elif isinstance(self.path_or_hdul, NDCollection):
            raise ValueError("No, No, No, No Sunraster untill another time")
        # if self.select_window is None: self.select_window = np.arange(len(self.path_or_hdul))
        else:
            raise ValueError(
                "You need to make sure that data file is a path or HDULLIST object "
            )

        raster = [
            rast
            for rast in self.raster
            if rast.header["EXTNAME"]
            not in ["VARIABLE_KEYWORDS", "WCSDVARR", "WCSDVARR"]
        ]
        self.filenames_generator()

    def filenames_generator(self):
        """
        Generate filenames using templates and replace placeholders.
        """

        if "::PARAMPLACEHOLDER" in self.data_filename:
            self.data_filename = self.data_filename.replace("::PARAMPLACEHOLDER", "{}")
        if "::SAMENAME" in self.data_filename:
            if "::SAMENAMEL2.5" in self.data_filename:
                filename = self.L2_path.stem
                filename = filename.replace("L2", "L2.5")
                self.data_filename = self.data_filename.replace(
                    "::SAMENAMEL2.5", filename
                )
            else:
                self.data_filename = self.data_filename.replace(
                    "::SAMENAME", self.L2_path.stem
                )

        now = datetime.datetime.now()
        formatted_time = now.strftime(r"%y%m%dT%H%M%S")
        if "::TIME" in self.data_filename:
            self.data_filename = self.data_filename.replace("::TIME", formatted_time)

        strConv = "".join([f"{i:02d}" for i in self.convolution_extent_list])
        if "::CONV" in self.data_filename:
            self.data_filename = self.data_filename.replace("::CONV", strConv)


class ProgressFollower:
    def __init__(self, file_path=None):
        data = np.array([0], dtype=int)
        self.shmm, self.data = gen_shmm(create=True, ndarray=data)
        prg = {
            "name": self.shmm.name,
            "dtype": self.data.dtype,
            "shape": self.data.shape,
        }
        self.file_path = file_path
        self.pickle_lock = Lock()
        if file_path is None:
            # creating a new .p file
            dir_path = Path("./tmp").resolve()
            dir_path.mkdir(parents=True, exist_ok=True)
            filename = (
                "Progress_Shmm_config_"
                + str(datetime.datetime.now())
                .replace("-", "")
                .replace(".", "")
                .replace(" ", "T")
                .replace(":", "")
                + ".p"
            )
            self.file_path = str(dir_path / filename)
            log = {
                "is_launched": prg,
                "name": [],
                "con": [],
                "window_size": [],
            }
            with self.pickle_lock:
                pickle.dump(log, open(self.file_path, "wb"))

    @property
    def is_launched(self):
        with self.pickle_lock:
            log = pickle.load(open(self.file_path, "rb"))
        shmm, data = gen_shmm(create=False, **log["is_launched"])
        return True if data[0] == 1 else False

    def append(self, name, con, window_size):
        with open(self.file_path, "rb") as file:
            log = pickle.load(file)
        log["name"].append(name if name is not None else str(len(log["name"])))
        log["con"].append(con)
        log["window_size"].append(window_size)
        with self.pickle_lock:
            pickle.dump(log, open(self.file_path, "wb"))

    def get_log(self):
        with self.pickle_lock:
            return pickle.load(open(self.file_path, "rb"))

    def launch(self):
        with self.pickle_lock:
            log = pickle.load(open(self.file_path, "rb"))
        shmm, data = gen_shmm(create=False, **log["is_launched"])
        if data[0] == 1:
            print("Progress is already tracked another one will be launched")
        data[0] = 1
        print("launching following progress")
        print("from file ", self.file_path)
        self.process = Process(
            target=self.follow_func, args=(self.file_path, self.pickle_lock)
        )
        self.process.start()

    def __enter__(self):
        # This method is called when you enter the 'with' block
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # This method is called when you exit the 'with' block
        self.__del__()

    def __del__(self):
        # Create a Path object for the file
        file_path = Path(self.file_path)
        if self.process.is_alive():
            print(
                "Progress follower process is still running. Proceeding terminating it."
            )
            self.process.terminate()
        else:
            print("Progress follower process has been terminated.")
        # Check if the file exists
        if file_path.exists():
            # Delete the file
            print("deleting: ", self.file_path)
            file_path.unlink()

    @staticmethod
    def follow_func(filename, pickle_lock):
        console = Console()
        names = []  # just for initiation
        while len(names) == 0:
            with open(filename, "rb") as file:
                log = pickle.load(file)
                names = log["name"]
                cons = log["con"]
                window_sizes = log["window_size"]

            if len(names) == 0:
                print("there was nothing to follow its progress")
                sleep(1)
            else:
                break
        reload_counter = datetime.datetime.now()
        print_counter = datetime.datetime.now()
        data_cons = []
        with Progress() as progress:
            tasks = []
            for ind in range(len(names)):
                name = names[ind]
                con = cons[ind]
                window_size = window_sizes[ind]
                shmm_con, data_con = gen_shmm(create=False, **con)
                data_cons.append(data_con)
                n_pixels = data_con[
                    0,
                    window_size[0, 0] : window_size[0, 1],
                    window_size[1, 0] : window_size[1, 1],
                ].size
                tasks.append(progress.add_task(name, total=n_pixels + 1))

            while not progress.finished:
                for ind, task in enumerate(tasks):
                    name = names[ind]
                    con = cons[ind]
                    window_size = window_sizes[ind]
                    shmm_con, data_con = gen_shmm(create=False, **con)
                    # data_con=data_cons[ind]

                    sub_data_con = data_con[
                        0,
                        window_size[0, 0] : window_size[0, 1],
                        window_size[1, 0] : window_size[1, 1],
                    ]
                    finished_pixels = sub_data_con[
                        np.logical_not(np.isnan(sub_data_con))
                    ].size
                    if (
                        np.abs(
                            (print_counter - datetime.datetime.now()).total_seconds()
                        )
                        > 10
                    ):
                        print(
                            "finished_pixels =", finished_pixels, "/", sub_data_con.size
                        )
                        print_counter = datetime.datetime.now()
                    progress.update(task, completed=finished_pixels)
                    # console.log(progress)
                    progress.refresh()
                # sleep(0.1)
                if (
                    np.abs((reload_counter - datetime.datetime.now()).total_seconds())
                    > 1
                ):
                    # if True:
                    with open(filename, "rb") as file:
                        log = pickle.load(file)
                    if len(log["name"]) > len(names):
                        for ind in range(len(names), len(log["name"])):
                            names.append(log["name"][ind])
                            cons.append(log["con"][ind])
                            window_sizes.append(log["window_size"][ind])
                            name = names[ind]
                            con = cons[ind]
                            window_size = window_sizes[ind]
                            shmm_con, data_con = gen_shmm(create=False, **con)
                            data_cons.append(data_con)
                            n_pixels = data_con[
                                0,
                                window_size[0, 0] : window_size[0, 1],
                                window_size[1, 0] : window_size[1, 1],
                            ].size
                            tasks.append(progress.add_task(name, total=n_pixels + 1))

                            progress.refresh()
                    reload_counter = datetime.datetime.now()


class WindowFit:
    def __repr__(self):
        if not isinstance(self.hdu, Iterable):
            extname = self.hdu.header["EXTNAME"]
        else:
            extname = " ".join([h.header["EXTNAME"] for h in self.hdu])

        val = (
            "Extname                "
            + extname
            + "\n"
            + "weights                 "
            + str(self.weights)
            + "\n"
            + "denoise                 "
            + str(self.denoise)
            + "\n"
            + "despike                 "
            + str(self.despike)
            + "\n"
            + "convolute               "
            + str(self.convolute)
            + "\n"
            + "has_treated             "
            + str(self.has_treated)
            + "\n"
            + "fit_func                "
            + str(self.fit_func)
            + "\n"
            + "init_params             "
            + str(self.init_params)
            + "\n"
            + "quentities              "
            + str(self.quentities)
            + "\n"
            + "window_size             "
            + str(self.window_size)
            + "\n"
            + "convolution_extent_list "
            + str(self.convolution_extent_list)
            + "\n"
            + "data_filename           "
            + str(self.data_filename)
            + "\n"
            + "data_save_dir           "
            + str(self.data_save_dir)
            + "\n"
            + "Jobs                    "
            + str(self.Jobs)
            + "\n"
        )
        return val

    def __init__(
        self,
        hdu: str or NDCollection or List[str or NDCollection],
        init_params: List[float] or List[List[float]],
        quentities: List[str] or List[List[str]],
        fit_func: callable,
        window_names: None or List[Dict] or List[List[Dict]] = None,
        bounds: np.array = np.array([np.nan]),
        window_size: np.ndarray = np.array([[500, 510], [60, 70]]),
        convolution_function: callable = lambda lst: np.zeros_like(lst[:, 2]) + 1,
        convolution_threshold: float = np.array([0.1, 10**-4, 0.1, 100]),
        convolution_extent_list: np.array = np.array([0, 1, 2, 3, 4, 5]),
        mode: str = "box",
        weights: bool = True,
        denoise: bool = True,
        despike: bool = True,
        convolute: bool = True,
        denoise_intervals: list = [6, 2, 1, 0, 0],
        clipping_sigma: float = 2.5,
        clipping_med_size: list = [6, 3, 3],
        clipping_iterations: int = 3,
        preclean: bool = True,
        save_data: bool = True,
        data_filename: str = None,
        data_save_dir: str = "./.p/",
        Jobs: int = 1,
        verbose: int = 0,
        lock=None,
        share_B: bool = False,
        dir_tmp_functions: str = None,
        import_function_list: list = [],
    ):
        if isinstance(hdu, Iterable):
            assert len(hdu) == len(init_params) == len(quentities)
            if isinstance(window_names, Iterable):
                assert len(hdu) == len(window_names)
        self.hdu = hdu
        self.init_params = init_params
        self.quentities = quentities
        self.window_names = window_names
        self.fit_func = fit_func
        self.bounds = bounds
        self.window_size = window_size
        self.convolution_function = convolution_function
        self.convolution_threshold = convolution_threshold
        self.convolution_extent_list = convolution_extent_list
        self.mode = mode
        self.weights = weights
        self.denoise = denoise
        self.despike = despike
        self.convolute = convolute
        self.denoise_intervals = denoise_intervals
        self.clipping_sigma = clipping_sigma
        self.clipping_med_size = clipping_med_size
        self.clipping_iterations = clipping_iterations
        self.preclean = preclean
        self.save_data = save_data
        self.data_filename = data_filename
        self.data_save_dir = data_save_dir
        self.Jobs = Jobs
        self.verbose = verbose
        self.lock = Lock() if lock is None else lock

        self.specaxis = None
        self.clean_data = None
        self.conv_data = None
        self.conv_sigma = None
        self.sigma = None
        self.Job_index_list = None

        self.data_par = None
        self.data_cov = None
        self.data_con = None

        self._shmm_par = None
        self._shmm_cov = None
        self._shmm_con = None
        self._shmm_war = None
        self._shmm_sgm = None
        self._par = None
        self._cov = None
        self._con = None
        self._war = None
        self._sgm = None

        self.lock_enabled = False
        self.lock_protocols = LockProtocols()

        if dir_tmp_functions is None:
            self._dir_tmp_functions = Path("./tmp_functions").resolve()
            self._dir_tmp_functions.mkdir(parents=True, exist_ok=True)
            sys.path.append(self._dir_tmp_functions)
            sys.path.append(Path("./SAFFRON").resolve())
            self.lock_protocols._dir_tmp_functions
        else:
            self._dir_tmp_functions = dir_tmp_functions
        self.import_function_list = import_function_list
        self.lock_protocols.import_function_list = self.import_function_list

        self.has_treated = {
            "preclean": False,
            "sigma": False,
            "despike": False,
            "convolve": False,
            "denoise": False,
        }
        # In case of multiple windows these windows will be stored in here
        if isinstance(self.hdu, Iterable):
            self.separate_windows: List[WindowFit] = []
            self.share_B = share_B
            self.separate_init_params = None
            self.separate_quentities = None
            self.separate_window_names = None
            self.separate_fit_func = None

        # self.run_preparations()
        # self.FIT_window()

    def run_preparations(self, redo=False):
        if isinstance(self.hdu, Iterable):
            self.polyHDU_preparation()
        else:
            self.monoHDU_preparations(redo=redo)
        self._Chianti_window_names = (
            self.get_CHIANTI_lineNames([name["name"] for name in self.window_names])
            if self.window_names is not None
            else None
        )

    def monoHDU_preparations(self, redo=False):
        self.specaxis = get_specaxis(self.hdu)
        self._preclean(redo=redo)
        self._get_sigma_data(redo=redo)
        self._despike(redo=redo)
        self._convolve(redo=redo)
        self._denoise(redo=redo)
        self._Gen_output_shared_memory()
        self._index_list()
        pass

    def polyHDU_preparation(self):
        if len(self.separate_windows) == len(self.hdu):
            if self.verbose > -1:
                print(
                    "the separate windows have been generated already if so the code may break as the variables of this instance have been already adapted [It's going to be reinitiated to old values first]"
                )
            self.init_params = self.separate_init_params
            self.quentities = self.separate_quentities
            self.window_names = self.separate_window_names
            self.fit_func = self.separate_fit_func
        else:
            self.separate_init_params = self.init_params.copy()
            self.separate_quentities = self.quentities.copy()
            self.separate_window_names = self.window_names.copy()
            self.separate_fit_func = (
                self.fit_func.copy()
                if isinstance(self.fit_func, Iterable)
                else self.fit_func
            )
            for ind, hdu in enumerate(self.hdu):
                # print(f"Generating window {ind}")

                self.separate_windows.append(
                    WindowFit(
                        hdu=self.hdu[ind],
                        init_params=self.init_params[ind],
                        quentities=self.quentities[ind],
                        fit_func=(
                            self.fit_func[ind]
                            if self.fit_func is Iterable
                            else self.fit_func
                        ),
                        window_names=(
                            self.window_names[ind]
                            if isinstance(self.window_names, Iterable)
                            else self.window_names
                        ),
                        bounds=self.bounds,
                        window_size=self.window_size,
                        convolution_function=self.convolution_function,
                        convolution_threshold=(
                            self.convolution_threshold
                            if not isinstance(self.convolution_threshold[0], Iterable)
                            else self.convolution_threshold[ind]
                        ),
                        convolution_extent_list=self.convolution_extent_list,
                        mode=self.mode,
                        preclean=self.preclean,
                        weights=self.weights,
                        denoise=self.denoise,
                        despike=self.despike,
                        convolute=self.convolute,
                        denoise_intervals=self.denoise_intervals,
                        clipping_sigma=self.clipping_sigma,
                        clipping_med_size=self.clipping_med_size,
                        clipping_iterations=self.clipping_iterations,
                        save_data=self.save_data,
                        data_filename=self.data_filename,
                        data_save_dir=self.data_save_dir,
                        Jobs=self.Jobs,
                        verbose=self.verbose,
                        lock=self.lock,
                    )
                )
                old_v = self.separate_windows[-1].verbose
                self.separate_windows[-1].verbose = -np.inf
                self.separate_windows[-1].run_preparations()
                self.separate_windows[-1].verbose = old_v

        self.convolution_threshold = np.concatenate(
            [i.convolution_threshold for i in self.separate_windows], axis=0
        )
        self.build_fused_params()
        self.build_fused_FitFunc()

        # Computing
        # new convdata
        # new specaxis
        # new sigma
        # self.has_treated = {"preclean":False,"sigma":False,"despike":False, "convolve":False, "denoise":False}

        self.specaxis = np.concatenate(
            [get_specaxis(i.hdu) for i in self.separate_windows], axis=0
        )
        self.sigma = np.concatenate([i.sigma for i in self.separate_windows], axis=1)
        self.conv_sigma = np.concatenate(
            [i.conv_sigma for i in self.separate_windows], axis=2
        )
        self.conv_data = np.concatenate(
            [i.conv_data for i in self.separate_windows], axis=2
        )
        self.Job_index_list = self.separate_windows[0].Job_index_list
        self._Gen_output_shared_memory()

    def build_fused_params(self):
        init_params = []  # new init_params
        convolution_threshold = []
        quentities = []  # new quentities
        window_names = []  # new window_names
        B_ind = []  # index of Background in init_params

        if self.share_B:
            # if np.any(
            #     np.array([win.quentities.count('B') for win in self.separate_windows]) !=1
            # ):raise Exception('There is a problem when fusing windows and sharing B there should be no multiple B values')
            for ind, window in enumerate(self.separate_windows):
                for nI in range(window.quentities.count("I")):
                    indI = find_nth_occurrence(window.quentities, "I", nI + 1)
                    init_params.extend(window.init_params[indI : indI + 3])
                    convolution_threshold.extend(
                        window.convolution_threshold[indI : indI + 3]
                    )
                    quentities.extend(window.quentities[indI : indI + 3])
                    window_names.append(
                        None
                        if not isinstance(window.window_names, Iterable)
                        else window.window_names[nI]
                    )

                B_ind.append(
                    [
                        find_nth_occurrence(window.quentities, "B", n)
                        for n in range(window.quentities.count("B"))
                    ]
                )
            init_params.append(
                1.0e-1
            )  # TODO later just take the average vlaue for each B in the system
            convolution_threshold.append(
                100
            )  # TODO later just take the average vlaue for each B in the system
            quentities.append("B")

        else:
            for ind, window in enumerate(self.separate_windows):
                init_params.extend(window.init_params)
                quentities.extend(window.quentities)
                window_names.extend(window.window_names)
                convolution_threshold.extend(window.convolution_threshold)

        self.init_params = np.array(init_params)
        self.quentities = quentities
        self.window_names = window_names
        self.convolution_threshold = np.array(convolution_threshold)

    def build_fused_FitFunc(self):
        B_ind = []
        for ind, window in enumerate(self.separate_windows):
            B_ind.append(
                [
                    find_nth_occurrence(window.quentities, "B", n + 1)
                    for n in range(window.quentities.count("B"))
                ]
            )
        fit_funcs = [i.fit_func for i in self.separate_windows]
        min_lbda = [np.nanmin(i.specaxis) for i in self.separate_windows]
        max_lbda = [np.nanmax(i.specaxis) for i in self.separate_windows]
        init_params = [i.init_params for i in self.separate_windows]
        init_params_empty = [i.init_params * np.nan for i in self.separate_windows]
        ind_originaleInFused = []

        iter = 0
        ind_fused = 0
        for ind in range(len(init_params)):

            for Bind in B_ind[ind]:
                init_params_empty[ind][Bind] = 0
            ind_originaleInFused.append(
                [
                    ind_fused,
                    ind_fused
                    + len(init_params_empty[ind])
                    - (len(B_ind[ind]) if self.share_B else 0),
                ]
            )
            ind_fused = ind_originaleInFused[-1][1]
        str_func_sharedB_TEMPLATE = """
                \nimport numpy as np
                \nfrom SAFFRON.fit_models import flat_inArg_multiGauss
                \ndef func_{}(x,*array):
                \n\tseparate_init_params = [{}]
                \n\tfit_func = [{}]
                \n\tmin_lbda = [{}]
                \n\tmax_lbda = [{}]
                \n\tind_OinF = {}
                \n\tres = np.zeros(x.shape)
                \n\tfor ind in range(len(separate_init_params)):
                \n\t    separate_init_params[ind][np.isnan(separate_init_params[ind])] = array[ind_OinF[ind][0]:ind_OinF[ind][1]]
                \n\t    separate_init_params[ind][separate_init_params[ind]==0] = array[-1]
                
                \n\t    bool_choice = np.logical_and(x>=min_lbda[ind],x<=max_lbda[ind])
                \n\t    res[bool_choice] += fit_func[ind](x,*separate_init_params[ind])[bool_choice]
                \n\treturn res
        """
        str_func_differentB_TEMPLATE = """
                \nimport numpy as np
                \nfrom SAFFRON.fit_models import flat_inArg_multiGauss
                \ndef func_{}(x,*array):
                \n\tseparate_init_params = [{}]
                \n\tfit_func = [{}]
                \n\tmin_lbda = [{}]
                \n\tmax_lbda = [{}]
                \n\tind_OinF = {}
                \n\tres = np.zeros(x.shape)
                \n\titer_array = 0
                \n\tfor ind in range(len(separate_init_params)):
                \n\t    separate_init_params[ind] = array[iter_array:iter_array+len(separate_init_params[ind])]
                \n\t    iter_array+=len(separate_init_params[ind])
                \n\t    bool_choice = np.logical_and(x>=min_lbda[ind],x<=max_lbda[ind])
                \n\t    res[bool_choice] += fit_func[ind](x,*separate_init_params[ind])[bool_choice]
                \n\treturn res
        """
        # Truning separate params into fused params inside the new function
        selected_func = (
            str_func_differentB_TEMPLATE
            if not self.share_B
            else str_func_sharedB_TEMPLATE
        )
        separate_init_str = "\n"
        for ind, inits in enumerate(init_params_empty):
            separate_init_str += "\t\t" + ArrToCode(inits) + ",\n"
            pass
        separate_init_str += "\t\t"
        time_str = datetime.datetime.now().strftime("%H%M%d%H%M%S")
        str_func = selected_func.format(
            time_str,
            separate_init_str,
            ", ".join([fit_func.__name__ for fit_func in fit_funcs]),
            ", ".join([str(val) for val in min_lbda]),
            ", ".join([str(val) for val in max_lbda]),
            ind_originaleInFused,
        )
        str_import_Template = "from tmp_functions.{0} import {0}"
        str_func = (
            "\n".join(
                [
                    str_import_Template.format(i.__name__)
                    for i in self.import_function_list
                    if i != flat_inArg_multiGauss
                ]
            )
            + str_func
        )

        # saving to file
        _func_name = "func_" + time_str
        with open(self._dir_tmp_functions / (_func_name + ".py"), mode="w") as f:
            f.writelines(str_func)

        # str_import_Template.format()

        # print(str_func)
        self.str_func = str_func
        loc = {}
        exec(str_import_Template.format(_func_name), globals(), loc)
        # exec(str_func,globals(),loc)
        self.fit_func = list(loc.items())[0][1]
        self.import_function_list.append(self.fit_func)
        self.lock_protocols.import_function_list = self.import_function_list

    def _get_sigma_data(self, redo=False):
        if self.verbose >= 0:
            print("Computing uncertainties")
        if self.has_treated["sigma"] and not redo:
            if self.verbose >= 0:
                print("already done")
            return
        elif not self.weights:
            if self.verbose >= 0:
                print("weights is set to false it's not going to be computed")
            self.sigma = np.ones(self.hdu.data.shape)
            return
        if self.verbose >= 1:
            av_constant_noise_level, sigma = spice_error(self.hdu, verbose=self.verbose)
        else:
            from ..utils.utils import suppress_output

            # with suppress_output():
            if True:
                av_constant_noise_level, sigma = spice_error(
                    self.hdu, verbose=self.verbose
                )

        self.sigma = sigma["Total"].value.astype(float)
        self.has_treated["sigma"] = True

    def _preclean(self, redo=False):
        if self.verbose >= 0:
            print("Precleaning data")
        if self.has_treated["preclean"] and not redo:
            if self.verbose >= 0:
                print("already done")
            return
        elif not self.preclean:
            if self.verbose >= 0:
                print("preclean is set to false it's not going to be computed")
            if self.clean_data is None:
                self.clean_data = (self.hdu.data).astype(float).copy()
            return
        self.clean_data = Preclean((self.hdu.data).astype(float).copy())
        self.has_treated["preclean"] = True

    def _despike(self, redo=False):
        if self.verbose >= 0:
            print("Despiking data")
        if self.has_treated["despike"] and not redo:
            if self.verbose >= 0:
                print("already done")
            return
        elif not self.despike:
            if self.verbose >= 0:
                print("despike is set to false it's not going to be computed")
            if self.clean_data is None:
                self.clean_data = (self.hdu.data).astype(float).copy()
            return
        self.clean_data = despike(
            raw_data=self.clean_data,
            clipping_sigma=self.clipping_sigma,
            clipping_med_size=self.clipping_med_size,
            clipping_iterations=self.clipping_iterations,
        )
        self.has_treated["despike"] = True

    def _convolve(self, redo=False):
        if self.verbose >= 0:
            print("Convolving data")
        if self.has_treated["convolve"] and not redo:
            if self.verbose >= 0:
                print("already done")
            return
        elif not self.convolute:
            if self.verbose >= 0:
                print("convolute is set to false it's not going to be computed")

            self.conv_data = convolve(
                window=self.clean_data,
                mode="box",
                lon_pixel_size=self.hdu.header["CDELT1"],
                lat_pixel_size=self.hdu.header["CDELT2"],
                convolution_extent_list=np.array([0]),
                convolution_function=self.convolution_function,
            )
            self.conv_sigma = convolve(
                window=self.sigma**2,
                mode="box",
                lon_pixel_size=self.hdu.header["CDELT1"],
                lat_pixel_size=self.hdu.header["CDELT2"],
                convolution_extent_list=np.array([0]),
                convolution_function=self.convolution_function,
            )

            self.conv_sigma = np.sqrt(self.conv_sigma)

        self.conv_data = convolve(
            window=self.clean_data,
            mode=self.mode,
            lon_pixel_size=self.hdu.header["CDELT1"],
            lat_pixel_size=self.hdu.header["CDELT2"],
            convolution_extent_list=self.convolution_extent_list,
            convolution_function=self.convolution_function,
        )
        self.conv_sigma = convolve(
            window=self.sigma**2,
            mode=self.mode,
            lon_pixel_size=self.hdu.header["CDELT1"],
            lat_pixel_size=self.hdu.header["CDELT2"],
            convolution_extent_list=self.convolution_extent_list,
            convolution_function=self.convolution_function,
        )
        self.conv_sigma = np.sqrt(self.conv_sigma)

        self.has_treated["convolve"] = True

    def _denoise(self, redo=False):
        if self.verbose >= 0:
            print("Denoising data")
        if self.has_treated["denoise"] and not redo:
            if self.verbose >= 0:
                print("already done")
            return
        elif not self.denoise:
            if self.verbose >= 0:
                print("denoise is set to false it's not going to be computed")
            return
        if type(self.denoise) != type(None):
            if self.verbose >= 1:
                print(f"Generating denoised maps with denoise sigma => {self.denoise}")
            for i in range(self.convolution_extent_list.shape[0]):
                Dnois_conv_data = denoise_data(
                    self.conv_data[i, 0], denoise_sigma=self.denoise_intervals
                )
                self.conv_data[i, 0] = Dnois_conv_data.copy()

        self.has_treated["denoise"] = True

    def _index_list(self):
        ws = self.window_size.copy()
        if ws[0, 1] == None:
            ws[0, 1] = self.data_par.shape[2]
        if ws[1, 1] == None:
            ws[1, 1] = self.data_par.shape[3]
        njobs = (ws[0, 1] - ws[0, 0]) * (ws[1, 1] - ws[1, 0]) // 100
        # njobs = njobs if njobs>self.Jobs else self.Jobs
        # njobs   = self.Jobs * 3
        verbose = self.verbose

        index_list = np.zeros(
            ((ws[0, 1] - ws[0, 0]) * (ws[1, 1] - ws[1, 0]), 2), dtype=int
        )
        inc = 0
        for i in range(ws[0, 0], ws[0, 1]):
            for j in range(ws[1, 0], ws[1, 1]):
                index_list[inc] = [i, j]
                inc += 1
        Job_index_list = []
        nPerJob = len(index_list) // njobs
        reste = len(index_list) % njobs
        if verbose >= 2:
            print("N pixels per job", nPerJob)
        if verbose >= 2:
            print("reste for first job", reste)
        for i in range(njobs):  # distributing pixels over jobs
            Job_index_list.append(
                index_list[
                    i * nPerJob
                    + (reste if i != 0 else 0) : min(
                        (i + 1) * nPerJob + reste, len(index_list)
                    )
                ]
            )
        self.Job_index_list = Job_index_list

    def _Gen_output_shared_memory(self):
        if self.verbose >= 0:
            print("Generating input/output shared memory")
        dshape = np.array(
            [
                self.conv_data.shape[1],
                self.conv_data.shape[3],
                self.conv_data.shape[4],
            ]
        )

        self.data_par = np.zeros((self.init_params.shape[0], *dshape)) * np.nan
        self.data_cov = np.zeros((self.init_params.shape[0], *dshape)) * np.nan
        self.data_con = np.zeros(dshape) * np.nan

        self._shmm_par, self.data_par = gen_shmm(create=True, ndarray=self.data_par)
        self._shmm_cov, self.data_cov = gen_shmm(create=True, ndarray=self.data_cov)
        self._shmm_con, self.data_con = gen_shmm(create=True, ndarray=self.data_con)

        self._par = {
            "name": self._shmm_par.name,
            "dtype": self.data_par.dtype,
            "shape": self.data_par.shape,
        }
        self._cov = {
            "name": self._shmm_cov.name,
            "dtype": self.data_cov.dtype,
            "shape": self.data_cov.shape,
        }
        self._con = {
            "name": self._shmm_con.name,
            "dtype": self.data_con.dtype,
            "shape": self.data_con.shape,
        }

        self._shmm_war, self.conv_data = gen_shmm(create=True, ndarray=self.conv_data)
        self._war = {
            "name": self._shmm_war.name,
            "dtype": self.conv_data.dtype,
            "shape": self.conv_data.shape,
        }
        if self.weights is not None:
            self._shmm_sgm, self.conv_sigma = gen_shmm(
                create=True, ndarray=self.conv_sigma
            )
            self._sgm = {
                "name": self._shmm_sgm.name,
                "dtype": self.conv_sigma.dtype,
                "shape": self.conv_sigma.shape,
            }

        if self.weights is not None:
            self._shmm_sgm_backup, self.conv_sigma_backup = gen_shmm(
                create=True, ndarray=self.conv_sigma
            )
            self._sgm_backup = {
                "name": self._shmm_sgm_backup.name,
                "dtype": self.conv_sigma_backup.dtype,
                "shape": self.conv_sigma_backup.shape,
            }

    def write_data(self):
        if isinstance(self.hdu, Iterable):
            hdu = flatten(self.hdu)[0]
        else:
            hdu = self.hdu
        if self.data_filename is None:
            if self.verbose >= 0:
                print(
                    r'You haven\'t specified so It\'s going to be set as "solo-spice-L2.5_{specific_param_val}.fits" '
                )
            self.data_filename = (
                hdu.header["FILENAME"].replace("L2", "L2.5")[:-5]
                + str(datetime.datetime.now())
                .replace("-", "")
                .replace(":", "")
                .replace(" ", "")[:-4]
                + "_{}.fits"
            )

        self.data_filename = (
            self.data_filename[1:]
            if self.data_filename[0] == "/"
            else self.data_filename
        )

        wcs = WCS(hdu.header)
        wcs_par = wcs.dropaxis(2)
        I_BTYPE = "Spectral Radiance"
        I_BUNIT = "W/m2/sr/nm"
        v_BTYPE = "Wavelength"
        v_BUNIT = "Angstrom"
        w_BTYPE = "Line Width"
        w_BUNIT = "Angstrom"

        hdul_list = []

        iter = 0
        bg_filenames = []
        while True:
            iter += 1
            ind = find_nth_occurrence(self.quentities, "B", iter)
            if ind == -1:
                break
            else:
                # print('number of Bvalues',self.quentities.count('B'))
                # bg_filename = self.data_filename.format("Bg"+str(self.quentities[:0*3].count('B')))
                bg_filename = self.data_filename.format(
                    "Bg" + str(iter - 1) + "-" + str(int(np.random.random() * 100))
                )
                header = wcs_par.to_header()
                data = self.data_par[ind, 0]
                sigma = np.sqrt(self.data_cov[ind, 0])
                header["BTYPE"] = I_BTYPE
                header["BUNIT"] = I_BUNIT
                header["L2_NAME"] = hdu.header["FILENAME"]

                header0 = header.copy()
                header1 = header.copy()

                header0["MEASRMNT"] = "bg"
                header1["MEASRMNT"] = "bg_err"

                hdu0 = fits.PrimaryHDU(data=data, header=header0)
                hdu1 = fits.ImageHDU(data=sigma, header=header1)
                hdul = HDUList([hdu0, hdu1])
                hdul_list.append([hdul.copy(), bg_filename])
                bg_filenames.append(bg_filename)

        for i in range(self.quentities.count("I")):
            ind = find_nth_occurrence(self.quentities, "I", i + 1)
            if self.window_names is not None:
                name, wvl = self._Chianti_window_names[i]

            headers = [
                wcs_par.to_header(),
                wcs_par.to_header(),
                wcs_par.to_header(),
            ]
            B_count = 0
            for j in range(3):
                headers[j]["OBSERVATORY"] = "Solar Orbiter"
                headers[j]["INSTRUMENT"] = "SPICE"
                headers[j]["WAVELENGTH"] = (
                    wvl if self.window_names is not None else "unknown"
                )
                headers[j]["ION"] = name if self.window_names is not None else "unknown"
                headers[j]["LINE_ID"] = (
                    (f"{wvl:08.2f}-{name}").replace(" ", "_")
                    if self.window_names is not None
                    else "unknown"
                )
                headers[j]["WAVEUNIT"] = "Angstrom"
                for k, con in enumerate(self.convolution_extent_list):
                    headers[j][f"con{k}"] = con

            headers[0]["BTYPE"] = I_BTYPE
            headers[0]["BUNIT"] = I_BUNIT
            headers[1]["BTYPE"] = v_BTYPE
            headers[1]["BUNIT"] = v_BUNIT
            headers[2]["BTYPE"] = w_BTYPE
            headers[2]["BUNIT"] = w_BUNIT

            header00 = headers[0].copy()
            header01 = headers[0].copy()
            header10 = headers[1].copy()
            header11 = headers[1].copy()
            header20 = headers[2].copy()
            header21 = headers[2].copy()

            header00["MEASRMNT"] = "int"
            header01["MEASRMNT"] = "int_err"
            header10["MEASRMNT"] = "wav"
            header11["MEASRMNT"] = "wav_err"
            header20["MEASRMNT"] = "wid"
            header21["MEASRMNT"] = "wid_err"

            data0 = self.data_par[ind, 0]
            sigma0 = np.sqrt(self.data_cov[ind, 0])
            data1 = self.data_par[ind + 1, 0]
            sigma1 = np.sqrt(self.data_cov[ind + 1, 0])
            data2 = self.data_par[ind + 2, 0]
            sigma2 = np.sqrt(self.data_cov[ind + 2, 0])

            if False:  # No need to put parameters in different files
                hdu00 = fits.PrimaryHDU(data=data0, header=header00)
                hdu01 = fits.ImageHDU(data=sigma0, header=header01)
                hdu10 = fits.PrimaryHDU(data=data1, header=header10)
                hdu11 = fits.ImageHDU(data=sigma1, header=header11)
                hdu20 = fits.PrimaryHDU(data=data2, header=header20)
                hdu21 = fits.ImageHDU(data=sigma2, header=header21)
                hdul0 = HDUList([hdu00, hdu01])
                hdul1 = HDUList([hdu10, hdu11])
                hdul2 = HDUList([hdu20, hdu21])
                I_filename = self.data_filename.format(
                    header00["LINE_ID"] + "-" + header00["MEASRMNT"]
                )
                v_filename = self.data_filename.format(
                    header10["LINE_ID"] + "-" + header10["MEASRMNT"]
                )
                w_filename = self.data_filename.format(
                    header20["LINE_ID"] + "-" + header20["MEASRMNT"]
                )
                hdul_list.append([hdul0.copy(), I_filename])
                hdul_list.append([hdul1.copy(), v_filename])
                hdul_list.append([hdul2.copy(), w_filename])
            else:  # now all parameters of a given line are inside 1 fits file are in the
                hdu00 = fits.PrimaryHDU(data=data0, header=header00)
                hdu10 = fits.ImageHDU(data=data1, header=header10)
                hdu20 = fits.ImageHDU(data=data2, header=header20)
                hdu01 = fits.ImageHDU(data=sigma0, header=header01)
                hdu11 = fits.ImageHDU(data=sigma1, header=header11)
                hdu21 = fits.ImageHDU(data=sigma2, header=header21)

                hdul = HDUList([hdu00, hdu10, hdu20, hdu01, hdu11, hdu21])
                l_filename = self.data_filename.format(header00["LINE_ID"])
                hdul_list.append([hdul.copy(), l_filename])

        data_save_dir = (
            Path(self.data_save_dir).resolve()
            if self.data_save_dir is not None
            else Path("./")
        )
        data_save_dir.mkdir(exist_ok=True, parents=True)
        for col in hdul_list:
            print(f"saving_to {data_save_dir/col[1]}")
            if not (data_save_dir / col[1]).parent.exists():
                print("parent folder doesn't exists... Proceeding creating it")
                (data_save_dir / col[1]).parent.mkdir(exist_ok=True, parents=True)
            if np.all(np.isnan(col[0][0].data)):
                print(Fore.red + "Data is full of NaNs not saving it")
                print(Style.RESET_ALL)
            else:
                col[0].writeto(data_save_dir / col[1], overwrite=True)

    def fit_window(self, progress_follower=None):
        warnings.filterwarnings(("ignore" if self.verbose <= -2 else "always"))
        if progress_follower is None:
            progress_follower = ProgressFollower()

        progress_follower.append(name=None, con=self._con, window_size=self.window_size)
        try:
            if not progress_follower.is_launched:
                progress_follower.launch()
        except:
            pass
        if self.verbose >= -1:
            print("par", self._par)
        if self.verbose >= -1:
            print("cov", self._cov)
        if self.verbose >= -1:
            print("con", self._con)
        Processes = []
        fit_func2, _ = gen_lock_fit_func(
            self.init_params, self.quentities, self.lock_protocols, self.fit_func
        )
        _now = datetime.datetime.now()
        for i in range(len(self.Job_index_list)):  # preparing processes:
            keywords = {
                "x": self.specaxis,
                "list_indeces": self.Job_index_list[i],
                "war": self._war,
                "par": self._par,
                "cov": self._cov,
                "con": self._con,
                "wgt": self._sgm,
                "ini_params": self.init_params,
                "quentities": self.quentities,
                "fit_func": fit_func2,
                "bounds": self.bounds,
                "convolution_threshold": self.convolution_threshold,
                "convolution_extent_list": self.convolution_extent_list,
                "verbose": self.verbose,
                "lock": self.lock,
                "lock_protocols": self.lock_protocols,
            }
            if False:
                if i == 0 and self.verbose >= -1:
                    print("Multiprocessing deactivated for debugging purposes")
                self.task_fit_pixel(**keywords)
            else:
                Processes.append(Process(target=self.task_fit_pixel, kwargs=keywords))
                Processes[-1].start()
                if self.verbose >= 1:
                    print(
                        "Live Processes: ",
                        np.sum([1 for p in Processes if p.is_alive()]),
                    )
                    print(
                        f"Starting process job: {i+1:02d}/{len(self.Job_index_list):.4g} on raster fits\nJob list contains: {len(self.Job_index_list[i])} pixels"
                    )
                    # print("remaining_pixels= ",nan_size,'/',data.size)
                while True:
                    # print("live processes: ",np.sum([1 for p in Processes if p.is_alive()]))
                    if np.sum([1 for p in Processes if p.is_alive()]) >= self.Jobs:
                        if self.verbose >= 1:
                            data = self.data_con[
                                0,
                                self.window_size[0, 0] : self.window_size[0, 1],
                                self.window_size[1, 0] : self.window_size[1, 1],
                            ].copy()
                            nan_size = data[(np.isnan(data))].size
                        # sleep(0.5)
                    else:
                        for j, p in enumerate(Processes):
                            if not p.is_alive():
                                # print("exitcode", p.exitcode != 0)
                                p.close()
                                Processes.pop(j)
                                pass
                        break

        while np.sum([1 for p in Processes if p.is_alive()]) != 0:
            if (
                self.verbose >= 0
                and np.abs((_now - datetime.datetime.now()).total_seconds()) >= 5
            ):
                print(
                    "Live Processes: ", np.sum([1 for p in Processes if p.is_alive()])
                )
                _now = datetime.datetime.now()

            data = self.data_con[
                0,
                self.window_size[0, 0] : self.window_size[0, 1],
                self.window_size[1, 0] : self.window_size[1, 1],
            ].copy()
            nan_size = data[(np.isnan(data))].size
            if (
                self.verbose >= 0
                and np.abs((_now - datetime.datetime.now()).total_seconds()) >= 5
            ):
                print("remaining_pixels= ", nan_size, "/", data.size)
                _now = datetime.datetime.now()

            # sleep(2)
        for process in Processes:
            process.join()

    @staticmethod
    def task_fit_pixel(
        x: np.ndarray,
        list_indeces: np.ndarray,
        war: dict,
        par: dict,
        cov: dict,
        con: dict,
        ini_params: np.ndarray,
        quentities: list[str],
        fit_func: callable,
        wgt: None = None,
        wgt_backup: None = None,
        bounds: np.ndarray = [np.nan],
        convolution_threshold=None,
        convolution_extent_list=None,
        verbose=0,
        lock=None,
        lock_protocols=LockProtocols(),
        **kwargs,
    ):

        shmm_war, data_war = gen_shmm(create=False, **war)
        shmm_par, data_par = gen_shmm(create=False, **par)
        shmm_cov, data_cov = gen_shmm(create=False, **cov)
        shmm_con, data_con = gen_shmm(create=False, **con)
        if type(wgt) != type(None):
            shmm_wgt, data_wgt = gen_shmm(create=False, **wgt)

        if len(convolution_threshold) == data_par.shape[0]:
            conv_thresh = convolution_threshold
        else:
            conv_thresh = np.zeros(data_par.shape[0])

            conv_thresh[-1] = convolution_threshold[-1]
            for i_q in range(data_par.shape[0] // 3):
                conv_thresh[i_q + 0] = convolution_threshold[0]
                conv_thresh[i_q + 1] = convolution_threshold[1]
                conv_thresh[i_q + 2] = convolution_threshold[2]

        for index in list_indeces:
            i_y, i_x = index
            if verbose >= 2:
                print(f"fitting pixel [{i_y},{i_x}]")
            i_ad = -1
            best_cov = np.zeros((*ini_params.shape,)) * np.nan
            best_par = np.zeros((*ini_params.shape,)) * np.nan
            if verbose > 2:
                print(f"y data: {data_war[i_ad,0,:,i_y,i_x]}")

            while (
                True
            ):  # this will break only when the convolution threshold is met or reached max allowed convolutions
                i_ad += 1  #                 |
                if i_ad == len(convolution_extent_list):
                    break  # <————————————————'
                if (
                    data_war[i_ad, 0, :, i_y, i_x][
                        np.logical_not(np.isnan(data_war[i_ad, 0, :, i_y, i_x]))
                    ].shape[0]
                    >= ini_params.shape[0]
                ):
                    locked_ini_params, locked_quentities, _, _ = gen_locked_params(
                        ini_params, quentities, lock_protocols
                    )
                    locked_last_par, locked_last_cov = fit_pixel_multi(
                        x=x,
                        y=data_war[i_ad, 0, :, i_y, i_x].copy(),
                        ini_params=(locked_ini_params),  # ini_params,
                        quentities=locked_quentities,  # quentities,
                        fit_func=fit_func,
                        bounds=bounds,
                        weights=(
                            None
                            if type(wgt) == type(None)
                            else data_wgt[i_ad, 0, :, i_y, i_x]
                        ),
                        verbose=verbose,
                        plot_title_prefix=f"{i_y:04d},{i_x:03d},{i_ad:03d}",
                        lock_protocols=lock_protocols,
                    )
                    last_par, unlocked_quentities, last_cov = gen_unlocked_params(
                        locked_last_par,
                        locked_quentities,
                        lock_protocols,
                        np.diag(locked_last_cov),
                    )

                else:
                    _s = ini_params.shape[0]
                    last_par, last_cov = (
                        np.ones((_s)) * np.nan,
                        np.ones((_s)) * np.nan,
                    )

                if np.isnan(last_par).all():
                    best_con = convolution_extent_list[i_ad]

                else:
                    if (np.isnan(best_par)).all():
                        best_cov = last_cov
                        best_par = last_par
                        best_con = convolution_extent_list[i_ad]

                    # NEW TO BE VERIFIED
                    else:

                        all_good = True
                        for i in range(len(best_par) // 3):
                            if not ((np.sqrt((last_cov))) / last_par < conv_thresh)[
                                i * 3 : (i + 1) * 3
                            ].all():
                                if ((np.sqrt((best_cov))) / best_par < conv_thresh)[
                                    i * 3 : (i + 1) * 3
                                ].all():
                                    best_cov[i * 3 : (i + 1) * 3] = last_cov[
                                        i * 3 : (i + 1) * 3
                                    ]
                                    best_par[i * 3 : (i + 1) * 3] = last_par[
                                        i * 3 : (i + 1) * 3
                                    ]

                                    best_cov[-1] = last_cov[-1]
                                    best_par[-1] = last_par[-1]

                                elif np.nansum(
                                    ((np.sqrt((last_cov))) / last_par / conv_thresh)[
                                        i * 3 : (i + 1) * 3
                                    ]
                                ) < np.nansum(
                                    ((np.sqrt((best_cov))) / best_par / conv_thresh)[
                                        i * 3 : (i + 1) * 3
                                    ]
                                ):
                                    best_cov[i * 3 : (i + 1) * 3] = last_cov[
                                        i * 3 : (i + 1) * 3
                                    ]
                                    best_par[i * 3 : (i + 1) * 3] = last_par[
                                        i * 3 : (i + 1) * 3
                                    ]

                                    best_cov[-1] = last_cov[-1]
                                    best_par[-1] = last_par[-1]
                                    all_good = False
                        if all_good == True:
                            break
                        best_con = convolution_extent_list[i_ad]

            if verbose >= 2:
                print(f"best_par: {best_par}\nbest_con: {best_con}")

            lock.acquire()
            data_par[:, 0, i_y, i_x] = (
                best_par  # the result UUUUUUgh finally it's here every pixel will be here
            )
            data_cov[:, 0, i_y, i_x] = (
                best_cov  # the result UUUUUUgh finally it's here every pixel will be here
            )
            data_con[0, i_y, i_x] = (
                best_con  # the result UUUUUUgh finally it's here every pixel will be here
            )
            lock.release()

    @staticmethod
    def get_CHIANTI_lineNames(names):
        cat = LineCatalog()
        _CHIANTI_lines = []
        wvl = cat.get_line_wvl(names)
        for i in range(len(names)):
            if names[i] == "no_line":
                CHName = "no_line"
            else:
                CHName = names[i].split("_")[0:2]
                CHName[1] = str(roman.fromRoman(CHName[1].upper()))
                CHName = "_".join(CHName)
            _CHIANTI_lines.append([CHName, wvl[i]])
        return _CHIANTI_lines
