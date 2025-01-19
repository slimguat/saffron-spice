import warnings
import numpy as np
import os
import sys
import re
import platform
import subprocess

# import pickle
from time import sleep
from multiprocessing import Process, Lock
from pathlib import PosixPath, WindowsPath, Path
import datetime
import pickle
from rich.progress import Progress
from rich.console import Console
from colorama import init, Fore, Style
init(autoreset=True)


from typing import Union, List, Dict, Any, Callable, Tuple, Optional, Iterable
from astropy.io import fits
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.wcs import WCS
from ndcube import NDCollection

# from spice_uncertainties import spice_error
from sospice import spice_error

# from sunraster.instr.spice import read_spice_l2_fits

from .fit_pixel import fit_pixel as fit_pixel_multi
from ..utils.denoise import denoise_data
from ..utils.despike import despike_4D
from ..utils.utils import (
    gen_shmm,
    Preclean,
    Preclean,
    # convolve,
    convolve_4D,
    get_specaxis,
    flatten,
    # default_convolution_function,
    get_extnames,
    get_data_raster,
    colored_text
)
from ..utils.fits_clone import HDUClone, HDUListClone

from ..fit_models import ModelFactory
class RasterFit:
    def __repr__(self) -> str:
        value = (
            # "init_params             "+ str(self.init_params)             +  "\n"+
            # "quentities              "+ str(self.quentities)              +  "\n"+
            "fit functions                "
            + str(self.models)
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
                    str(self.models[i].get_lock_params()) + "\n" + str(self.models[i].get_lock_quentities())
                    for i in range(len(self.models))
                ]
            )
            
            # + "\n\t".join(
            #     [
            #         str(self.init_params[i]) + "\n" + str(self.quentities[i])
            #         for i in range(len(self.init_params))
            #     ]
            # )
        )
        return value

    def __init__(
        self,
        path_or_hdul,
        models,
        # init_params: list,
        # quentities: list,
        # fit_func: callable,
        # windows_names: list = None,
        # bounds: np.array = np.array([np.nan]),
        window_size: np.ndarray = np.array([[500, 510], [60, 70]]),
        time_size: Iterable = [0, None],
        # convolution_function: callable = default_convolution_function,
        convolution_threshold: float = np.array([0.1, 10**-4, 0.1, 100]),
        convolution_extent_list: np.array = np.array([0, 1, 2, 3, 4, 5]),
        t_convolution_index: int = 0,
        mode: str = "box",
        weights: bool = True,
        denoise: bool = True,
        despike: bool = True,
        convolute: bool = True,
        denoise_intervals: list = [6, 2, 1, 0, 0],
        clipping_sigma: float = 2.5,
        clipping_med_size: list = [3, 6, 3, 3],
        clipping_iterations: int = 3,
        preclean: bool = True,
        save_data: bool = True,
        data_filename: str = "NoName.fits",
        data_save_dir: str = "./fits/",
        Jobs: int = 1,
        verbose: int = 0,
    ):
        self.path_or_hdul = path_or_hdul
        self.models = models
        self.window_size = window_size
        self.time_size = time_size
        # self.convolution_function = convolution_function
        self.convolution_threshold = convolution_threshold
        self.convolution_extent_list = convolution_extent_list
        self.t_convolution_index = t_convolution_index
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
        

        self.L2_path = ""
        self.raster = None
        self.load_data()
        self.headers = [
            self.raster[i].header
            for i in range(len(self.raster))
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
                model= [self.windows[i].model for i in indices],
                window_size=self.window_size,
                time_size = self.time_size,
                # convolution_function=self.convolution_function,
                convolution_threshold=(
                    self.convolution_threshold
                    if not isinstance(self.convolution_threshold[0], Iterable)
                    else [self.convolution_threshold[i] for i in indices]
                ),
                convolution_extent_list=self.convolution_extent_list,
                t_convolution_index = self.t_convolution_index,
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
            )
        )
        x = self.solo_windows_toFit
        y = indices
        self.solo_windows_toFit = [value for value in x if value not in y]
        # if in the other windows the data has been treated copy them into the new fused windows
        
        # First: make sure that all the windows have the same treatment status 
        all_is_treated_same = all(
            [
                all([
                    self.windows[i].has_treated[key]==self.windows[indices[0]].has_treated[key]
                    for key in self.windows[i].has_treated
                    ])
                for i in indices
            ]
        )
        if all_is_treated_same:
            #Second: role over preclean,sigma, despike, convolve,denoise, 
            has_treated = self.windows[indices[0]].has_treated
            
            #Preclean and despike search 
            if self.windows[indices[0]].clean_data is not None:
                if self.verbose >= 1:
                    print("fusing preclean and despike found in the source windows. Using them in the fused window")
                self.fused_windows[-1].clean_data = np.concatenate(
                    [self.windows[i].clean_data for i in indices],
                    axis=1) 
                self.fused_windows[-1].has_treated["preclean"] = has_treated["preclean"]
                self.fused_windows[-1].has_treated["despike" ] = has_treated["despike"]
                
            
            #Sigma search  
            if self.windows[indices[0]].sigma is not None:
                if self.verbose >= 1:
                    print("fusing sigma found in the source windows. Using them in the fused window")
                self.fused_windows[-1].sigma = np.concatenate(
                    [self.windows[i].sigma for i in indices],
                    axis=1)
                self.fused_windows[-1].has_treated["sigma"] = has_treated["sigma"]
            
            #Convolve search
            if self.windows[indices[0]].conv_data is not None:
                if self.verbose >= 1:
                    print("fusing convolve and denoise found in the source windows. Using them in the fused window")
                self.fused_windows[-1].conv_data = np.concatenate(
                    [self.windows[i].conv_data for i in indices],
                    axis=2)
                self.fused_windows[-1].conv_sigma = np.concatenate(
                    [self.windows[i].conv_sigma for i in indices],
                    axis=2)
                
                self.fused_windows[-1].has_treated["convolve"] = has_treated["convolve"]
                self.fused_windows[-1].has_treated["denoise" ] = has_treated["denoise" ]
            
            #Shared memory initiation 
            if has_treated["shared_memory"]:
                if self.verbose >= 1:
                    print("proceeding with shared memory generation in the fused window")
                self.fused_windows[-1].run_preparations(redo=False,without_shared_memory=False)
                
    def gen_windows(self):
        for i in range(len(self.raster)):
            window = WindowFit(
                hdu=self.raster[i],
                model = self.models[i],
                window_size=self.window_size,
                time_size=self.time_size,
                # convolution_function=self.convolution_function,
                convolution_threshold=(
                    self.convolution_threshold
                    if not isinstance(self.convolution_threshold[0], Iterable)
                    else self.convolution_threshold[i]
                ),
                convolution_extent_list=self.convolution_extent_list,
                t_convolution_index=self.t_convolution_index,
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
                
            )
            self.windows.append(window)

     #TODO: This have been abusively used in the code, it should be removed or rearranged
    
    def run_raster_preparations_parallel(self, redo=False, max_processes=None,without_shared_memory=False):
        
        from concurrent.futures import ProcessPoolExecutor
        for ind in range(len(self.windows)):
            self.windows[ind].model._callables = None
        max_processes = max_processes or os.cpu_count()
        print("\033[91mrun preparation in parallel")
        print("father process id", os.getpid(),"\033[0m")
        
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            self.windows = list(executor.map(
                run_one_window_preparations, self.windows, [redo] * len(self.windows),[without_shared_memory]*len(self.windows)
            ))
            self.fused_windows = list(executor.map(
                run_one_window_preparations, self.fused_windows, [redo] * len(self.fused_windows)   ,[without_shared_memory]*len(self.fused_windows)
            ))
    
    #TODO: new implementation of the function, to be tested
    def run_preparations(self, redo=False,max_processes=os.cpu_count(),without_shared_memory=False):
        if max_processes !=1:
            # raise "max_processes is not implemented yet"
            self.run_raster_preparations_parallel(redo=redo, max_processes=max_processes,without_shared_memory=True)
            self.run_preparations(redo=False,without_shared_memory=without_shared_memory,max_processes=1)
        else:
            for i in range(len(self.windows)):
                self.windows[i].run_preparations(redo=redo,without_shared_memory=without_shared_memory)
            for i in range(len(self.fused_windows)):
                self.fused_windows[i].run_preparations(redo=redo,without_shared_memory=without_shared_memory)

    def fit_raster(self, progress_follower=None):
        if progress_follower is None:
            progress_follower = ProgressFollower()
        for ind2 in range(len(self.fused_windows)):
            self.fused_windows[ind2].fit_window(progress_follower=progress_follower)
            self.fused_windows[ind2].write_data()

        for ind in self.solo_windows_toFit:
            self.windows[ind].fit_window(progress_follower=progress_follower)
            self.windows[ind].write_data()
        
    def write_data(self):
        for ind in self.solo_windows_toFit:
            self.windows[ind].write_data()
        for ind2 in range(len(self.fused_windows)):
            self.fused_windows[ind2].write_data()

    def load_data(self):
        if self.verbose > 1:
            print("reading data")
        if type(self.path_or_hdul) in (str, PosixPath, WindowsPath):
            self.L2_path = self.path_or_hdul
            if self.verbose > 1:
                print(f"data is given as path:  {self.path_or_hdul  }")
            self.raster = fits.open(self.path_or_hdul)
            self.raster = get_data_raster(self.raster)
        elif isinstance(self.path_or_hdul, HDUList):
            self.raster = self.path_or_hdul
            self.raster = HDUListClone.from_hdulist(self.raster)
            self.path_or_hdul = None
        elif isinstance(self.path_or_hdul, NDCollection):
            raise ValueError("No Sunraster untill another time")
        # if self.select_window is None: self.select_window = np.arange(len(self.path_or_hdul))
        else:
            raise ValueError(
                "You need to make sure that data file is a path or HDULLIST object "
            )
        self.raster = get_data_raster(self.raster) 
        self.filenames_generator()

    # NOTE No need to adapt it, it is already good and the change will be once the object is called
    def filenames_generator(self):
        """
        Generate filenames using templates and replace placeholders.
        """

        if "::PARAMPLACEHOLDER" in self.data_filename:
            self.data_filename = self.data_filename.replace("::PARAMPLACEHOLDER", "{}")
        if "::SAMENAME" in self.data_filename:
            if "::SAMENAMEL2.5" in self.data_filename:
                filename = Path(self.L2_path).stem
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
        if "::TCONV" in self.data_filename:
            self.data_filename = self.data_filename.replace("::TCONV", self.t_convolution_index)


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
                "time_size": [],
                
            }
            with self.pickle_lock:
                pickle.dump(log, open(self.file_path, "wb"))

    @property
    def is_launched(self):
        with self.pickle_lock:
            log = pickle.load(open(self.file_path, "rb"))
        shmm, data = gen_shmm(create=False, **log["is_launched"])
        return True if data[0] == 1 else False

    def append(self, name, con, window_size,time_size):
        with open(self.file_path, "rb") as file:
            log = pickle.load(file)
        log["name"].append(name if name is not None else str(len(log["name"])))
        log["con"].append(con)
        log["window_size"].append(window_size)
        log["time_size"].append(time_size)
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
                time_sizes = log["time_size"]

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
                time_size = time_sizes[ind]
                shmm_con, data_con = gen_shmm(create=False, **con)
                data_cons.append(data_con)
                n_pixels = data_con[
                    time_size[0] : time_size[1],
                    window_size[0, 0] : window_size[0, 1],
                    window_size[1, 0] : window_size[1, 1],
                ].size
                tasks.append(progress.add_task(name, total=n_pixels + 1))

            while not progress.finished:
                for ind, task in enumerate(tasks):
                    name = names[ind]
                    con = cons[ind]
                    window_size = window_sizes[ind]
                    time_size = time_sizes[ind]
                    shmm_con, data_con = gen_shmm(create=False, **con)
                    # data_con=data_cons[ind]

                    sub_data_con = data_con[
                        time_size[0] : time_size[1],
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
                        # print(
                        #     "finished_pixels =", finished_pixels, "/", sub_data_con.size
                        # )
                        print_counter = datetime.datetime.now()
                        pass
                    progress.update(task, completed=finished_pixels)
                    # Remove the task if completed
                    if progress.tasks[task].completed:
                        progress.remove_task(task)
                        console.log(f"Task '{names[ind]}' completed and removed.")
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
                            time_sizes.append(log["time_size"][ind])
                            name = names[ind]
                            con = cons[ind]
                            window_size = window_sizes[ind]
                            time_size = time_sizes[ind]
                            shmm_con, data_con = gen_shmm(create=False, **con)
                            data_cons.append(data_con)
                            n_pixels = data_con[
                                time_size[0] : time_size[1],
                                window_size[0, 0] : window_size[0, 1],
                                window_size[1, 0] : window_size[1, 1],
                            ].size
                            tasks.append(progress.add_task(name, total=n_pixels + 1))

                            progress.refresh()
                    reload_counter = datetime.datetime.now()


def _prepare_axes(num_plots, axis=None):
    """
    Prepare axes for plotting.

    Parameters:
        num_plots (int): Number of plots required.
        axis (matplotlib axis): Axis object if provided.

    Returns:
        tuple: matplotlib figure and axes.
    """
    if axis is None:
        c = int(min(5, math.ceil(np.sqrt(num_plots))))
        r = int(np.ceil(num_plots / c))
        fig, axes = plt.subplots(r, c, figsize=(c * 3, r * 3))
        try:axes = axes.flatten()
        except:pass
        [ax.remove() for ax in axes[num_plots:]]
        [ax.grid() for ax in axes[:num_plots]]
        return fig, axes[:num_plots]
    else:
        return None, axis


def plot_pixel(spectrum,x_axis=None, t=None, y=None, x=None, ax=None,plot_kwargs={}):
    """
    Plots the spectrum of a single pixel.

    Parameters:
    ----------
    spectrum : np.ndarray
        The spectrum data to plot (1D array along wavelength).
    t : int
        Time index.
    y : int
        Y-coordinate of the pixel.
    x : int
        X-coordinate of the pixel.
    ax : matplotlib axis
        Axis object to use for plotting.

    Returns:
    -------
    None
    """
    fig,ax = _prepare_axes(1, ax)
    if x_axis is None:
        x_axis = np.arange(len(spectrum))
    ax.step(x_axis,spectrum, label=f"t={t}, y={y}, x={x}",**plot_kwargs)
    ax.set_xlabel("Wavelength Index")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Spectrum at (t={t}, y={y}, x={x})")
    ax.legend()

def plot_predefined_pixels(hdu, indices, plot_pixel_function, axis=None,plot_kwargs={}):
    """
    Plots a list of predefined indices from the data.

    Parameters:
    ----------
    hdu : HDU object
        HDU containing the data array with shape (time, wavelength, height, width).
    indices : list of tuples
        List of (t, y, x) indices to plot.
    plot_pixel_function : callable
        Function to plot a single pixel. It should accept parameters (spectrum, t, y, x, ax).
    axis : matplotlib axis, optional
        Axis object if provided.

    Returns:
    -------
    tuple: (indices, axis)
    """
    if not isinstance(hdu,np.ndarray):
        data = hdu.data.copy()
        specaxis = get_specaxis(hdu)
    else:
        data = hdu.copy()
        specaxis = np.arange(data.shape[1])
    num_pixels = len(indices)

    # Validate data shape
    if len(data.shape) != 4:
        raise ValueError("Input data must be a 4D array with shape (time, wavelength, height, width).")

    # Prepare axes for plotting
    fig, axes = _prepare_axes(num_pixels, axis)

    # Plot the selected pixels
    for index, (rand_t, rand_y, rand_x) in enumerate(indices):
        spectrum = data[rand_t, :, rand_y, rand_x]  # Extract the spectrum along the wavelength axis
        plot_pixel_function(spectrum, x_axis=specaxis, t=rand_t, y=rand_y, x=rand_x, ax=axes[index],plot_kwargs=plot_kwargs)

    plt.tight_layout()
    # plt.show()
    return indices, axes

def plot_random_pixels(hdu, num_pixels, plot_pixel_function, axis=None,plot_kwargs={}):
    """
    Picks random pixels from data along axes (t, y, x) and calls `plot_predefined_pixels`.

    Parameters:
    ----------
    hdu : HDU object
        HDU containing the data array with shape (time, wavelength, height, width).
    num_pixels : int
        Number of random pixels to pick and plot.
    plot_pixel_function : callable
        Function to plot a single pixel. It should accept parameters (spectrum, t, y, x, ax).
    axis : matplotlib axis, optional
        Axis object if provided.

    Returns:
    -------
    tuple: (indices, axis)
    """
    if not isinstance(hdu,np.ndarray):
        data = hdu.data.copy()
    else:
        data = hdu.copy()
    # Validate data shape
    if len(data.shape) != 4:
        raise ValueError("Input data must be a 4D array with shape (time, wavelength, height, width).")

    n_time, _, n_y, n_x = data.shape

    # Ensure we don't pick more pixels than possible
    max_pixels = n_time * n_y * n_x
    if num_pixels > max_pixels:
        raise ValueError(f"Cannot pick {num_pixels} pixels from a total of {max_pixels} available pixels.")

    # Generate a list of all (t, y, x) combinations
    t_indices, y_indices, x_indices = np.meshgrid(
        np.arange(n_time), np.arange(n_y), np.arange(n_x), indexing='ij'
    )
    all_indices = np.stack([t_indices.ravel(), y_indices.ravel(), x_indices.ravel()], axis=1)

    # Randomly choose `num_pixels` indices without replacement
    chosen_indices = random.sample(list(all_indices), num_pixels)

    # Call `plot_predefined_pixels` with the chosen indices
    return plot_predefined_pixels(hdu, chosen_indices, plot_pixel_function, axis=axis,plot_kwargs=plot_kwargs)



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
            + str(self.model)
            + "\n"
            + "window_size   "
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
        model : ModelFactory,
        window_size: np.ndarray = np.array([[500, 510], [60, 70]]),
        time_size: Iterable = [0, None],
        convolution_threshold: np.ndarray = np.array([0.1, 10**-4, 0.1, 100]),
        convolution_extent_list: np.ndarray = np.array([0, 1, 2, 3, 4, 5]),
        t_convolution_index: int = 0,
        mode: str = "box",
        weights: bool = True,
        denoise: bool = True,
        despike: bool = True,
        convolute: bool = True,
        denoise_intervals: list = [6, 2, 1, 0, 0],
        clipping_sigma: float = 2.5,
        clipping_med_size: list = [3, 6, 3, 3],
        clipping_iterations: int = 3,
        preclean: bool = True,
        save_data: bool = True,
        data_filename: Union[str, None] = None,
        data_save_dir: str = "./.p/",
        Jobs: int = 1,
        verbose: int = 0,
    ):  
        
        if not isinstance(hdu, HDUClone) and not isinstance(hdu, Iterable):
            hdu = HDUClone.from_hdu(hdu)
        self.hdu = hdu
        self.model = model
        self.window_size = window_size
        self.time_size = time_size
        # self.convolution_function = convolution_function
        self.convolution_threshold = convolution_threshold
        self.convolution_extent_list = convolution_extent_list
        self.t_convolution_index = t_convolution_index
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
        
        self.has_treated = {
            "preclean": False,
            "sigma": False,
            "despike": False,
            "convolve": False,
            "denoise": False,
            'shared_memory':False
        }
        # In case of multiple windows these windows will be stored in here
        if isinstance(self.hdu, Iterable):
            self.separate_windows: List[WindowFit] = []
            self.separate_models = None
            # self.share_B = share_B
            # self.separate_init_params = None
            # self.separate_quentities = None
            # self.separate_window_names = None
            # self.separate_fit_func = None

        # self.run_preparations()
        # self.FIT_window()

    def run_preparations(self, redo=False,without_shared_memory=False):
        warnings.filterwarnings("ignore")
        start = datetime.datetime.now() 
        if isinstance(self.hdu, Iterable):
            self.polyHDU_preparation(without_shared_memory=without_shared_memory)
        else:
            self.monoHDU_preparations(redo=redo,without_shared_memory=without_shared_memory)
        print("\033[91m process id", os.getpid(),
              "EXTNAME", (self.hdu.header["EXTNAME"]) if not isinstance(self.hdu, Iterable) else self.hdu[0].header["EXTNAME"],
              "beg", start,
              "end", datetime.datetime.now(),"\033[0m") 
    
    def monoHDU_preparations(self, redo=False,without_shared_memory=False):
        self.specaxis = get_specaxis(self.hdu)
        self._preclean(redo=redo)
        self._get_sigma_data(redo=redo)
        self._despike(redo=redo)
        self._convolve(redo=redo)
        self._denoise(redo=redo)
        print("without shared memory inside monoHDU_preparations",without_shared_memory)
        if not without_shared_memory:
            self._Gen_output_shared_memory()
            self._index_list()
        
        pass

    def polyHDU_preparation(self,without_shared_memory=False):
        if len(self.separate_windows) == len(self.hdu):
            if self.verbose > -1:
                print(
                    "the separate windows have been generated already if so the code may break as the variables of this instance have been already adapted [It's going to be reinitiated to old values first]"
                )
            self.model = self.separate_models
        else:
            self.separate_models = self.model
            for ind, hdu in enumerate(self.hdu):
                self.separate_windows.append(
                    WindowFit(
                        hdu=self.hdu[ind],
                        model = self.model[ind],
                        # bounds=self.bounds,
                        window_size=self.window_size,
                        time_size=self.time_size,
                        # convolution_function=self.convolution_function,
                        convolution_threshold=(
                            self.convolution_threshold
                            if not isinstance(self.convolution_threshold[0], Iterable)
                            else self.convolution_threshold[ind]
                        ),
                        convolution_extent_list=self.convolution_extent_list,
                        t_convolution_index=self.t_convolution_index,
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
                        
                    )
                )
            
             
            if [self.preclean == self.has_treated["preclean"],
                self.denoise == self.has_treated["denoise"],
                self.despike == self.has_treated["despike"],
                self.convolute == self.has_treated["convolve"]].count(False) > 0: 
                # coppying the cleaned data to the fused windows
                # if self.has_treated
                old_v = [self.separate_windows[ind].verbose for ind in range(len(self.separate_windows))]
                for ind in range(len(self.separate_windows)): self.separate_windows[ind].verbose = -2
                [self.separate_windows[ind].run_preparations() for ind in range(len(self.separate_windows))]
                for ind in range(len(self.separate_windows)): self.separate_windows[ind].verbose = old_v[ind] 
                self.sigma = np.concatenate([i.sigma for i in self.separate_windows], axis=1)
                self.conv_sigma = np.concatenate(
                    [i.conv_sigma for i in self.separate_windows], axis=2
                )
                self.conv_data = np.concatenate(
                    [i.conv_data for i in self.separate_windows], axis=2
                )
        
        self.convolution_threshold = np.concatenate(
            [i.convolution_threshold for i in self.separate_windows], axis=0
        )
        self.model = np.sum(self.separate_models)
        
        self.specaxis = np.concatenate(
            [get_specaxis(i.hdu) for i in self.separate_windows], axis=0
        )
        if not without_shared_memory:
            self._Gen_output_shared_memory()
            self._index_list()
    
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
            av_constant_noise_level, sigma = spice_error(self.hdu.to_hdu(), verbose=self.verbose)
        else:
            from ..utils.utils import suppress_output

            # with suppress_output():
            if True:
                av_constant_noise_level, sigma = spice_error(
                    self.hdu.to_hdu(), verbose=self.verbose
                )
                
        self.sigma = sigma["Total"].value.astype(float)
        if np.all(np.isnan(self.sigma)):
            print("\033[91mAll sigma values are nan, going back to sigma = 1 everywhere\033[0m")
            self.sigma[:] = 1
        self.has_treated["sigma"] = True
        if self.verbose >= 1:
            print("Uncertainties computed")
        
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
        if self.verbose >= 1:
            print("Precleaning done")

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
        self.clean_data = despike_4D(
            raw_data=self.clean_data.copy(),
            clipping_sigma=self.clipping_sigma,
            clipping_med_size=self.clipping_med_size,
            clipping_iterations=self.clipping_iterations,
        )
        self.has_treated["despike"] = True
        if self.verbose >= 1:
            print("Despiking done")

    def _convolve(self, redo=False):
        #convolve in space and date
        if self.verbose >= 0:
            print("Convolving data")
        if self.has_treated["convolve"] and not redo:
            if self.verbose >= 0:
                print("already done")
            return
        elif self.convolute:
            if True:
                expanded_convolution_list = np.empty([self.convolution_extent_list.shape[0], 4], dtype=int)
                CDELT1 = self.hdu.header["CDELT1"] if isinstance(self.hdu, HDUClone) else self.hdu[0].header["CDELT1"]
                CDELT2 = self.hdu.header["CDELT2"] if isinstance(self.hdu, HDUClone) else self.hdu[0].header["CDELT2"]
                shape = self.hdu.data.shape if isinstance(self.hdu, HDUClone) else self.hdu[0].data.shape
                ratio = float(CDELT1/ CDELT2)
                for i in range(self.convolution_extent_list.shape[0]):
                    size = np.array(
                        [
                            1 + self.t_convolution_index,
                            1,
                            1 + (self.convolution_extent_list[i]),
                            1 + (self.convolution_extent_list[i])* np.nanmin([ratio,1/ratio]),
                        ],
                        dtype=int,
                    )
                    for dim in range(len(size)): 
                        if shape[dim] < size[dim]:
                            size[dim] = shape[dim]
                    expanded_convolution_list[i] = size
            self.conv_data = convolve_4D(
                window=self.clean_data,
                mode=self.mode,
                convolution_extent_list=expanded_convolution_list,
            )
            self.conv_sigma = convolve_4D(
                window=self.sigma**2,
                mode=self.mode,
                convolution_extent_list=expanded_convolution_list
            )
            self.conv_sigma = np.sqrt(self.conv_sigma)

            self.has_treated["convolve"] = True
            if self.verbose >= 1:
                print("Convolution done")
            
        else:
            if self.verbose >= 0:
                print("convolute is set to false it's not going to be computed")
            self.conv_data = np.array([self.clean_data])
            self.conv_sigma = np.array([self.sigma]) 
            # self.has_treated["convolve"] = True
            
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
            if self.conv_data[0].shape[3] == 1:
                colored_text('Warning: It appears that the data has one value along xaxis, the author of saffron didn\'t test it throughly yet so be careful of potential implications', 'yellow') 

            for i in range(self.convolution_extent_list.shape[0]):
                for j in range(self.conv_data.shape[1]):
                    if np.any(~np.isnan(self.conv_data[i, j])):
                        Dnois_conv_data = denoise_data(
                            self.conv_data[i, j], denoise_sigma=self.denoise_intervals
                        )
                        self.conv_data[i, j] = Dnois_conv_data.copy()
                        


        self.has_treated["denoise"] = True
        if self.verbose >= 1:
            print("Denoising done")
    
    def _index_list(self):
        """
        Create a list of (i, j) pixel indices to be fit,
        then split them into generator-based batches of size 100
        and store them in `self.Job_index_list`.
        """
        def generate_coord_batches(coords: np.ndarray, batch_size: int = 100):
            """
            Yields consecutive slices of `coords` as batches of size `batch_size`.
            Each yielded value is an array of shape (<=batch_size, 2).
            """
            for start_idx in range(0, len(coords), batch_size):
                yield coords[start_idx : start_idx + batch_size]
        
        # Make a local copy of window_size
        ws = self.window_size.copy()
        ts = np.array(self.time_size).copy()
        
        # Handle None boundaries
        if ws[0, 1] is None or ws[0,1]>self.data_par.shape[2]:
            ws[0, 1] = self.data_par.shape[2]
        if ws[1, 1] is None or ws[1,1]>self.data_par.shape[3]:
            ws[1, 1] = self.data_par.shape[3]
        if ts[1] is None or ts[1]>self.data_par.shape[1]:
            ts[1] = self.data_par.shape[1]
        if ws[0, 0] is None or ws[0,0]>self.data_par.shape[2]:
            ws[0, 0] = 0
        if ws[1, 0] is None or ws[1,0]>self.data_par.shape[3]:
            ws[1, 0] = 0
        if ts[0] is None or ts[0]>self.data_par.shape[1]:
            ts[0] = 0
        self.window_size = ws.copy()
        self.time_size = ts.copy()
        
        # Compute total pixels in the specified subregion
        pixel_count = (ws[0, 1] - ws[0, 0]) * (ws[1, 1] - ws[1, 0] * (ts[1] - ts[0]))
        batch_size = max(min(100, int(pixel_count/self.Jobs)), 10)  # Limit batch size to 100 pixels

        # Vectorized creation of (t,i, j) pairs
        t_vals = np.arange(ts[0], ts[1])
        i_vals = np.arange(ws[0, 0], ws[0, 1])
        j_vals = np.arange(ws[1, 0], ws[1, 1])
        coords = np.stack(np.meshgrid(t_vals,i_vals, j_vals, indexing="ij"), axis=-1).reshape(-1, 3)

        # Call the helper function and simply turn the generator into a list
        job_index_list = list(generate_coord_batches(coords, batch_size=batch_size))

        self.Job_index_list = job_index_list

        return np.array(job_index_list,dtype=object)

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

        self.data_par = np.zeros((self.model.get_unlock_params().shape[0], *dshape)) * np.nan
        self.data_cov = np.zeros((self.model.get_unlock_params().shape[0], *dshape)) * np.nan
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

        if self.verbose >= 1:
            print("Shared memory generated")
        self.has_treated['shared_memory'] = True
        
    def write_data(self):
        def get_python_environment_info():
            # Kernel info
            kernel = platform.system()
            kernel_release = platform.release()
            
            # Architecture info
            architecture = platform.machine()
            
            # Hostname
            hostname = platform.node()
            
            # Operating System info
            os_info = platform.platform()

            # CPU info
            cpu_info = "Unknown CPU"
            try:
                if kernel == "Linux":
                    cpu_info = subprocess.check_output("lscpu", shell=True).decode().strip()
                    cpu_info = [line.split(":")[1].strip() for line in cpu_info.split("\n") if line.startswith("Model name")][0]
                elif kernel == "Windows":
                    cpu_info = platform.processor()
                elif kernel == "Darwin":
                    cpu_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            except Exception as e:
                cpu_info = f"Could not retrieve CPU info: {e}"

            # Python-specific info
            python_version = sys.version.split()[0]  # Python version
            python_compiler = platform.python_compiler()  # Compiler
            python_build = platform.python_build()  # Build details
            virtual_env = os.environ.get("VIRTUAL_ENV", "None")  # Virtual environment
            
            # Key libraries and their versions (optional)
            try:
                import numpy
                numpy_version = numpy.__version__
            except ImportError:
                numpy_version = "Not installed"

            # Formatting output like PRENV2
            environment_info = f"""
                Kernel: {kernel}
                Kernel release number: {kernel_release}
                Architecture: {architecture}
                Host name: {hostname}
                OS: {os_info}
                CPU: {cpu_info}
                Python version: {python_version}
                Python compiler: {python_compiler}
                Python build: {python_build}
                Memory bits: 64, File offset bits: 64
            """
            return environment_info.strip()
        def generate_fit_id():
            # Get the current time as a numpy datetime64 with millisecond precision
            current_time = np.datetime64('now', 'ms')
            # Convert to string and format as desired
            fit_id = str(current_time).replace('-', '').replace(':', '').replace('T', 'T')  # Keep 'T' separator
            return fit_id
        
        if True: #preparing the filename 
            if isinstance(self.hdu, Iterable):
                hdu = flatten(self.hdu)[0]
            else:
                hdu = self.hdu
            if self.data_filename is None:
                if self.verbose >= 0:
                    print(
                        r'You haven\'t specified so It\'s going to be set as "solo-spice-L2.5_{specific_param_val}.fits" '
                    )
                # in case the data_filename is not defined 
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
        
        if True: #preparing base header
            # Generate a FIT_ID
            fit_id = generate_fit_id()
            # Add it to the FITS header
            FIT_ID = fit_id
            PRENV2 = get_python_environment_info()
            PRENV2 = PRENV2.replace("\n", ";")
            PRENV2 = PRENV2.replace("  ", "")
            # FILENAME = 
            L2PARENT = hdu.header["FILENAME"]
            L2WINDOW = (self.hdu.header["EXTNAME"] if isinstance(self.hdu, HDUClone) else ",".join([h.header["EXTNAME"] for h in self.hdu]))
            # print(L2WINDOW)
            L1PARENT = hdu.header["PARENT"]
            LEVEL= 'L2.5'
            HISTORY = str(hdu.header["HISTORY"])+'\nSAFFRON python library'
            HISTORY = HISTORY.replace("\n", ";")
            
            # These are L2 header keys to be transformed into the L3 data header
            all_keys=  np.array(list(hdu.header.keys()))

            pattern = r"PR(?:STEP|PROC|PVER)\d+"  
            matches = np.vectorize(lambda x: bool(re.search(pattern, x)))(all_keys)
            index_occurrences = np.where(matches)[0]
            all_keys[index_occurrences]
            largest_number = max(
                int(match.group()) for key in all_keys[index_occurrences] for match in [re.search(r'\d+', key)] if match
            )

            PRSTEPkey  = f"PRSTEP{largest_number+1}"
            PRPROCkey  = f"PRPROC{largest_number+1}"
            PRPVERkey  = f"PRPVER{largest_number+1}"

            PRSTEP = "FITTING"
            PRPROC = "SAFFRON application of the fitting algorithm"
            PRPVER = "1.3"
            WINSIZE = ','.join([str(i) for i in [*self.window_size.flatten(),*self.time_size] ])
            copy_keys = [
                # Observation Context
                    "STUDYTYP", "STUDYDES", "STUDY", "OBS_MODE", "OBS_ID", "SPIOBSID",
                    "OBS_DESC", "PURPOSE", "SOOPNAME", "SOOPTYPE", "NRASTERS", "RASTERNO",
                    "STUDY_ID", "MISOSTUD", "XSTART", "XPOSURE", "FOCUSPOS",
                # Calibration and Processing Steps
                "RADCAL",*all_keys[index_occurrences],
                # Physical and Ephemeris Information
                    "DSUN_OBS", "DSUN_AU", "RSUN_ARC", "RSUN_REF", "SOLAR_B0", "SOLAR_P0", 
                    "CAR_ROT", "HGLT_OBS", "HGLN_OBS", "CRLT_OBS", "CRLN_OBS", "OBS_VR", 
                    "EAR_TDEL", "SUN_TIME", "DATE_EAR", "DATE_SUN",
                ]
            base_header = {}
            base_comments = {}
            for key in copy_keys:
                base_header[key] = hdu.header[key]
                base_comments[key] = hdu.header.comments[key]

                base_header["L2PARENT" ] = L2PARENT; base_comments["L2PARENT" ] = "L2 parent file name" 
                base_header["L1PARENT" ] = L1PARENT; base_comments["L1PARENT" ] = "L1 parent file name"
                base_header["L2WINDOW"    ] = L2WINDOW; base_comments["L2WINDOW"    ] = "L2 extension name"
                base_header["FILENAME" ] = Path(self.data_filename).name ; base_comments["FILENAME" ] = "Data product file name"
                base_header["LEVEL"    ] = LEVEL   ; base_comments["LEVEL"    ] = "Data product level"
                base_header["HISTORY"  ] = HISTORY ; base_comments["HISTORY"  ] = "Processing history"
                base_header["PRENV2"   ] = PRENV2  ; base_comments["PRENV2"   ] = "environment information"
                base_header[PRSTEPkey  ] = PRSTEP  ; base_comments[PRSTEPkey  ] = "Processing step"
                base_header[PRPROCkey  ] = PRPROC  ; base_comments[PRPROCkey  ] = "Processing procedure"
                base_header[PRPVERkey  ] = PRPVER  ; base_comments[PRPVERkey  ] = "Processing version"
                base_header["FIT_ID"   ] = fit_id  ; base_comments["FIT_ID"   ] = "Unique ID for the window fit process"
                base_header['WINSIZE'  ] = WINSIZE ; base_comments['WINSIZE'  ] = "Window size and time size"
                
        hdul_list = []
        quentities = self.model.get_unlock_quentities()
        # Siblings = np.zeros(
        #         (np.sum([len(self.model.functions[key]) for key in (list(self.model.functions.keys())) ]),2)
        #         ,dtype=object
        #         )
        Siblings = []
        if True:        
            I_pattern = r"I"  
            matches = np.vectorize(lambda x: bool(re.search(I_pattern, x)))(quentities)
            I_index_occurrences = np.where(matches)[0]
            
            for index_order , index_occurence in enumerate(I_index_occurrences):
                try:
                    wvl = self.model.functions_names['gaussian'][index_order][1]
                except Exception as e:
                    wvl = "unknown"
                    print(f"Error: {e}")
                try:
                    name = self.model.functions_names['gaussian'][index_order][0]
                except Exception as e:
                    name = "unknown"
                    print(f"Error: {e}")
                    
                headers = [
                    wcs_par.to_header(),
                    wcs_par.to_header(),
                    wcs_par.to_header(),
                ]
                
                for j in range(3):
                    headers[j]["OBSERVATORY"] = "Solar Orbiter"
                    headers[j]["INSTRUMENT"] = "SPICE"
                    
                    headers[j]["WAVELENGTH"] = wvl
                    headers[j]["ION"] = name 
                    headers[j]["LINE_ID"] = (f"{wvl:08.2f}-{name}").replace(" ", "_")
                        
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
                tmp = [
                    header00,
                    header01,
                    header10,
                    header11,
                    header20,
                    header21,]
                
               
                data0 = self.data_par[index_occurence, 0]
                sigma0 = np.sqrt(self.data_cov[index_occurence, 0])
                data1 = self.data_par[index_occurence + 1, 0]
                sigma1 = np.sqrt(self.data_cov[index_occurence + 1, 0])
                data2 = self.data_par[index_occurence + 2, 0]
                sigma2 = np.sqrt(self.data_cov[index_occurence + 2, 0])

                hdu00 = fits.PrimaryHDU(data=data0,  header=header00 ,)
                hdu10 = fits.ImageHDU  (data=data1,  header=header10 , name="wavelength"    )
                hdu20 = fits.ImageHDU  (data=data2,  header=header20 , name="width"         )
                hdu01 = fits.ImageHDU  (data=sigma0, header=header01 , name="intensity_err" )
                hdu11 = fits.ImageHDU  (data=sigma1, header=header11 , name="wavelength_err")
                hdu21 = fits.ImageHDU  (data=sigma2, header=header21 , name="width_err"     )

                hdul = HDUList([hdu00, hdu10, hdu20, hdu01, hdu11, hdu21])
                keys=  list(base_header.keys())
                keys.sort()
                for key in keys: 
                    for ind in range(len(hdul)):
                        hdul[ind].header[key] = base_header[key]
                        hdul[ind].header.comments[key] = base_comments[key]
                        
                l_filename = self.data_filename.format(header00["LINE_ID"])
                hdul_list.append([hdul.copy(), l_filename])
                Siblings.append([l_filename,[index_occurence,index_occurence+1,index_occurence+2,]])
                
        if True:
            B_pattern = r"B(\d?)"  # Expression régulière pour trouver "B" ou "B" suivi d'un chiffre
            matches = np.vectorize(lambda x: bool(re.search(B_pattern, x)))(quentities)
            B_index_occurrences = np.where(matches)[0]
            # quentities[B_index_occurrences]
            #NOTE All the bakgrounds of a model are in one file
            # Generating a FIT_ID window ID
            bg_filename = self.data_filename.format(
                    # quentities[index_occurence] +"_"+ str(index_order) + "_" + str(int(np.random.random() * 100))
                    "Background" + "_" + fit_id
                )
            background_list = []
            bg_index_occurence = []
            for index_order , index_occurence in enumerate(B_index_occurrences):
                bg_index_occurence.append(index_occurence)
                header = wcs_par.to_header()
                data = self.data_par[index_occurence]
                sigma = np.sqrt(self.data_cov[index_occurence])
                header["BTYPE"] = I_BTYPE
                header["BUNIT"] = I_BUNIT
                for key in base_header: 
                    header[key] = base_header[key]
                    header.comments[key] = base_comments[key]
                    
                header0 = header.copy()
                header1 = header.copy()

                header0["MEASRMNT"] = "bg"
                header1["MEASRMNT"] = "bg_err"
                header0["Parameter"] = f"{quentities[index_occurence]};{index_occurence}"
                header1["Parameter"] = f"{quentities[index_occurence]};{index_occurence}"
                
                if index_order == 0:
                    hdu0 = fits.PrimaryHDU(data=data, header=header0)
                else: 
                    hdu0 = fits.ImageHDU  (data=data, header=header0, name=f"Bg;{header0['Parameter']}")
                hdu1 = fits.ImageHDU(data=sigma, header=header1, name=f"Bg_err;{header1['Paramete']}")
                background_list.extend([hdu0.copy(), hdu1.copy()])
                
            hdul = HDUList(background_list)
            hdul_list.append([hdul.copy(), bg_filename])
            # bg_filenames.append(bg_filename)
            Siblings.append([bg_filename, bg_index_occurence])
        
        if True:
            # indicate for each file what are its siblings and add them to the model hdu
            model_hdu = self.model.to_hdu()
            model_hdu.header["FIT_ID"   ] = fit_id  ; base_comments["FIT_ID"   ] = "Unique ID for the window fit process"
            
            for sibling_ind,sibling in enumerate(Siblings):
                model_hdu.header["SIB"+str(sibling_ind)] = Path(sibling[0]).name
                model_hdu.header["ORD"+str(sibling_ind)] = ",".join(([str(val) for val in sibling[1]]))
                model_hdu.header.comments["SIB"+str(sibling_ind)] = f"Sibling {str(sibling_ind)} file name"
                model_hdu.header.comments["ORD"+str(sibling_ind)] = f"Sibling {str(sibling_ind)} order in the model unlock paraameters"
                
            for ind_hdul in range(len(hdul_list)):
                hdul_list[ind_hdul][0].append(model_hdu)
        
        # Saving the files
        data_save_dir = (
            Path(self.data_save_dir).resolve()
            if self.data_save_dir is not None
            else Path("./")
        )
        data_save_dir.mkdir(exist_ok=True, parents=True)
        paths = []
        for col in hdul_list:
            print(f"saving_to {data_save_dir/col[1]}")
            if not (data_save_dir / col[1]).parent.exists():
                print("parent folder doesn't exists... Proceeding creating it")
                (data_save_dir / col[1]).parent.mkdir(exist_ok=True, parents=True)
            if np.all(np.isnan(col[0][0].data)):
                print(Fore.RED + "Data is full of NaNs not saving it")
                print(Style.RESET_ALL)
            else:
                col[0].writeto(data_save_dir / col[1], overwrite=True)
                paths.append(data_save_dir / col[1])
                col[0].close()
        return paths







        # iter = 0
        # bg_filenames = []
        # while True:
        #     iter += 1
        #     ind = find_nth_occurrence(self.quentities, "B", iter)
        #     if ind == -1:
        #         break
        #     else:
        #         # print('number of Bvalues',self.quentities.count('B'))
        #         # bg_filename = self.data_filename.format("Bg"+str(self.quentities[:0*3].count('B')))
        #         bg_filename = self.data_filename.format(
        #             "Bg" + str(iter - 1) + "-" + str(int(np.random.random() * 100))
        #         )
        #         header = wcs_par.to_header()
        #         data = self.data_par[ind, 0]
        #         sigma = np.sqrt(self.data_cov[ind, 0])
        #         header["BTYPE"] = I_BTYPE
        #         header["BUNIT"] = I_BUNIT
        #         header["L2_NAME"] = hdu.header["FILENAME"]

        #         header0 = header.copy()
        #         header1 = header.copy()

        #         header0["MEASRMNT"] = "bg"
        #         header1["MEASRMNT"] = "bg_err"

        #         hdu0 = fits.PrimaryHDU(data=data, header=header0)
        #         hdu1 = fits.ImageHDU(data=sigma, header=header1)
        #         hdul = HDUList([hdu0, hdu1])
        #         hdul_list.append([hdul.copy(), bg_filename])
        #         bg_filenames.append(bg_filename)

        # for i in range(self.quentities.count("I")):
        #     ind = find_nth_occurrence(self.quentities, "I", i + 1)
        #     if self.window_names is not None:
        #         name, wvl = self._Chianti_window_names[i]

        #     headers = [
        #         wcs_par.to_header(),
        #         wcs_par.to_header(),
        #         wcs_par.to_header(),
        #     ]
        #     B_count = 0
        #     for j in range(3):
        #         headers[j]["OBSERVATORY"] = "Solar Orbiter"
        #         headers[j]["INSTRUMENT"] = "SPICE"
        #         headers[j]["WAVELENGTH"] = (
        #             wvl if self.window_names is not None else "unknown"
        #         )
        #         headers[j]["ION"] = name if self.window_names is not None else "unknown"
        #         headers[j]["LINE_ID"] = (
        #             (f"{wvl:08.2f}-{name}").replace(" ", "_")
        #             if self.window_names is not None
        #             else "unknown"
        #         )
        #         headers[j]["WAVEUNIT"] = "Angstrom"
        #         for k, con in enumerate(self.convolution_extent_list):
        #             headers[j][f"con{k}"] = con

        #     headers[0]["BTYPE"] = I_BTYPE
        #     headers[0]["BUNIT"] = I_BUNIT
        #     headers[1]["BTYPE"] = v_BTYPE
        #     headers[1]["BUNIT"] = v_BUNIT
        #     headers[2]["BTYPE"] = w_BTYPE
        #     headers[2]["BUNIT"] = w_BUNIT

        #     header00 = headers[0].copy()
        #     header01 = headers[0].copy()
        #     header10 = headers[1].copy()
        #     header11 = headers[1].copy()
        #     header20 = headers[2].copy()
        #     header21 = headers[2].copy()

        #     header00["MEASRMNT"] = "int"
        #     header01["MEASRMNT"] = "int_err"
        #     header10["MEASRMNT"] = "wav"
        #     header11["MEASRMNT"] = "wav_err"
        #     header20["MEASRMNT"] = "wid"
        #     header21["MEASRMNT"] = "wid_err"

        #     data0 = self.data_par[ind, 0]
        #     sigma0 = np.sqrt(self.data_cov[ind, 0])
        #     data1 = self.data_par[ind + 1, 0]
        #     sigma1 = np.sqrt(self.data_cov[ind + 1, 0])
        #     data2 = self.data_par[ind + 2, 0]
        #     sigma2 = np.sqrt(self.data_cov[ind + 2, 0])

        #     if False:  # No need to put parameters in different files
        #         hdu00 = fits.PrimaryHDU(data=data0, header=header00)
        #         hdu01 = fits.ImageHDU(data=sigma0, header=header01)
        #         hdu10 = fits.PrimaryHDU(data=data1, header=header10)
        #         hdu11 = fits.ImageHDU(data=sigma1, header=header11)
        #         hdu20 = fits.PrimaryHDU(data=data2, header=header20)
        #         hdu21 = fits.ImageHDU(data=sigma2, header=header21)
        #         hdul0 = HDUList([hdu00, hdu01])
        #         hdul1 = HDUList([hdu10, hdu11])
        #         hdul2 = HDUList([hdu20, hdu21])
        #         I_filename = self.data_filename.format(
        #             header00["LINE_ID"] + "-" + header00["MEASRMNT"]
        #         )
        #         v_filename = self.data_filename.format(
        #             header10["LINE_ID"] + "-" + header10["MEASRMNT"]
        #         )
        #         w_filename = self.data_filename.format(
        #             header20["LINE_ID"] + "-" + header20["MEASRMNT"]
        #         )
        #         hdul_list.append([hdul0.copy(), I_filename])
        #         hdul_list.append([hdul1.copy(), v_filename])
        #         hdul_list.append([hdul2.copy(), w_filename])
        #     else:  # now all parameters of a given line are inside 1 fits file are in the
        #         hdu00 = fits.PrimaryHDU(data=data0, header=header00)
        #         hdu10 = fits.ImageHDU(data=data1, header=header10)
        #         hdu20 = fits.ImageHDU(data=data2, header=header20)
        #         hdu01 = fits.ImageHDU(data=sigma0, header=header01)
        #         hdu11 = fits.ImageHDU(data=sigma1, header=header11)
        #         hdu21 = fits.ImageHDU(data=sigma2, header=header21)

        #         hdul = HDUList([hdu00, hdu10, hdu20, hdu01, hdu11, hdu21])
        #         l_filename = self.data_filename.format(header00["LINE_ID"])
        #         hdul_list.append([hdul.copy(), l_filename])

        # data_save_dir = (
        #     Path(self.data_save_dir).resolve()
        #     if self.data_save_dir is not None
        #     else Path("./")
        # )
        # data_save_dir.mkdir(exist_ok=True, parents=True)
        # for col in hdul_list:
        #     print(f"saving_to {data_save_dir/col[1]}")
        #     if not (data_save_dir / col[1]).parent.exists():
        #         print("parent folder doesn't exists... Proceeding creating it")
        #         (data_save_dir / col[1]).parent.mkdir(exist_ok=True, parents=True)
        #     if np.all(np.isnan(col[0][0].data)):
        #         print(Fore.RED + "Data is full of NaNs not saving it")
        #         print(Style.RESET_ALL)
        #     else:
        #         col[0].writeto(data_save_dir / col[1], overwrite=True)
    
    def fit_window(self, progress_follower=None):
        warnings.filterwarnings(("ignore" if self.verbose <= -2 else "always"))
        if progress_follower is None:
            progress_follower = ProgressFollower()

        progress_follower.append(name=None, con=self._con, window_size=self.window_size,time_size=self.time_size)
        try:
            if not progress_follower.is_launched:
                progress_follower.launch()
        except:
            pass
        if self.verbose >= -2:
            print("par", self._par)
        if self.verbose >= -2:
            print("cov", self._cov)
        if self.verbose >= -2:
            print("con", self._con)
        Processes = []
        _now = datetime.datetime.now()
        for i in range(len(self.Job_index_list)):  # preparing processes:
            # Remove callables to render the class picklable
            self.model._callables = None
            keywords = {
                "x": self.specaxis,
                "list_indeces": self.Job_index_list[i],
                "war": self._war,
                "par": self._par,
                "cov": self._cov,
                "con": self._con,
                "wgt": self._sgm,
                "model" : self.model,
                "convolution_threshold": self.convolution_threshold,
                "convolution_extent_list": self.convolution_extent_list,
                "verbose": self.verbose,
                
            }
            if False:
                if i == 0 and self.verbose >= -1:
                    colored_text("Multiprocessing deactivated for debugging purposes","red")
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
                                self.time_size[0] : self.time_size[1],
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
        model: ModelFactory,
        wgt: None = None,
        wgt_backup: None = None,
        convolution_threshold=None,
        convolution_extent_list=None,
        verbose=0,
        lock=None,
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
        
        unlocked_ini_params = model.get_unlock_params()
        locked_ini_params = model.get_lock_params()
        locked_quentities = model.get_lock_quentities()
        fit_func = model.callables['function']
        jacobian = model.callables['jacobian']
        bounds = model.bounds
        
        for index in list_indeces:
            i_t,i_y, i_x = index
            if verbose >= 2:
                print(f"fitting pixel [{i_t},{i_y},{i_x}]")
            i_ad = -1
            best_cov = np.zeros((*model.get_unlock_params().shape,)) * np.nan
            best_par = np.zeros((*model.get_unlock_params().shape,)) * np.nan
            if verbose > 2:
                print(f"y data: {data_war[i_ad,0,:,i_y,i_x]}")

            while (
                True
            ):  # this will break only when the convolution threshold is met or reached max allowed convolutions
                i_ad += 1  #                 |
                if i_ad == len(convolution_extent_list):
                    break  # <———————————————'
                if (
                    data_war[i_ad, i_t, :, i_y, i_x][
                        np.logical_not(np.isnan(data_war[i_ad, i_t, :, i_y, i_x]))
                    ].shape[0]
                    >= model.get_unlock_params().shape[0]
                ):
                    
                    
                    locked_last_par, locked_last_cov = fit_pixel_multi(
                        x=x,
                        y=data_war[i_ad, i_t, :, i_y, i_x].copy(),
                        ini_params=(locked_ini_params),  # ini_params,
                        quentities=locked_quentities,  # quentities,
                        fit_func=fit_func,
                        jac_func=jacobian,
                        bounds=bounds,
                        weights=(
                            None
                            if type(wgt) == type(None)
                            else data_wgt[i_ad, i_t, :, i_y, i_x]
                        ),
                        verbose=verbose,
                        plot_title_prefix=f"{i_y:04d},{i_x:03d},{i_ad:03d}",
                        # lock_protocols=lock_protocols, 
                        # #TODO search what are the reprocations for deleting the lock_protocols
                    )
                    last_par = model.get_unlock_params(locked_last_par) 
                    unlocked_quentities = model.get_unlock_quentities() 
                    last_cov = np.diag(model.get_unlock_covariance(locked_last_cov))
                    # print(f"last_par: {last_par}\nlast_cov: {last_cov}")
                    if any(np.isnan(locked_last_par)):
                        print("no fitting possible")
                else:
                    _s = unlocked_ini_params.shape[0]
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

            # lock.acquire()
            data_par[:, i_t, i_y, i_x] = (
                best_par  # the result UUUUUUgh finally it's here every pixel will be here
            )
            data_cov[:, i_t, i_y, i_x] = (
                best_cov  # the result UUUUUUgh finally it's here every pixel will be here
            )
            data_con[i_t, i_y, i_x] = (
                best_con  # the result UUUUUUgh finally it's here every pixel will be here
            )
            # lock.release()


def run_one_window_preparations(window_fit_obj, redo=False,without_shared_memory=False):
    """
    Helper function that simply calls `run_preparations` on a single WindowFit
    and returns the updated object.
    """
    #print the process number in red 
    # print(f"\033[91mProcess {os.getpid()} is running window \n{window_fit_obj}\033[0m")
    window_fit_obj.run_preparations(redo=redo,without_shared_memory=without_shared_memory)
    return window_fit_obj
