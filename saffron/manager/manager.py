import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from ..fit_models import flat_inArg_multiGauss
from ..utils import getfiles

# from ..init_handler import gen_fit_inits
from ..init_handler import GenInits
from ..fit_functions.fit_raster import RasterFit
from typing import Union, List, Dict, Any, Callable, Tuple, Optional, Iterable


class Manager:
    def __init__(self, Input_JSON: Optional[str]):
        """
        Initialize an instance of the Manager class.

        Args:
            Input_JSON (Optional[str]): Path to a JSON configuration file.
        """
        if Input_JSON is not None:
            # Read configuration file
            with open(Input_JSON) as config_file:
                self.config = json.load(config_file)

            # In the future you need to add a reference JSON whene there is
            self.SELECTION_MODE = self.config["SELECTION_MODE"]
            raster_args_config = self.config["fit_raster_args"]
            self.preclean = raster_args_config["preclean"]
            self.weights = raster_args_config["weights"]
            self.denoise = raster_args_config["denoise"]
            self.despike = raster_args_config["despike"]
            self.convolute = raster_args_config["convolute"]
            self.denoise_intervals = [6, 2, 1, 0, 0]
            self.clipping_sigma = 2.5
            self.clipping_med_size = [6, 3, 3]
            self.clipping_iterations = 3
            self.mode = "box"
            self.conv_errors = raster_args_config["conv_errors"]
            self.convolution_extent_list = np.array(
                raster_args_config["convolution_extent_list"]
            )
            self.save_data = raster_args_config["save_data"]
            self.data_filename = raster_args_config["data_filename"]
            self.data_save_dir = raster_args_config["data_save_dir"]
            self.window_size = np.array(raster_args_config["window_size"])
            self.Jobs = raster_args_config["Jobs"]
            self.geninits_verbose = self.config["geninits_verbose"]
            self.fit_verbose = self.config["fit_verbose"]
            self.selected_fits = []
            self.rasters = []
            # TODO DELET
            # self.plot_filename           = raster_args_config["plot_filename"       ]
            # self.save_plot               = raster_args_config["save_plot"           ]
            # self.plot_save_dir           = raster_args_config["plot_save_dir"       ]
            # for directory in [self.plot_save_dir,self.data_save_dir]:
            # if not Path(directory).exists():
            #     print(f"the directory:{directory} doesn't exist creating it now")
            #     os.mkdir(directory)

    def build_files_list(self):
        """
        Build the list of selected FITS files based on the chosen selection mode.
        """

        if len(self.selected_fits) != 0:
            print("Warning : self.selected_fits is not empty it will be overwritten")
        if self.SELECTION_MODE == "intervale":
            intervale_config = self.config["file selection mode=> date intervale"]
            self.L2_folder = intervale_config["L2_folder"]
            self.YEAR = intervale_config["YEAR"]
            self.MONTH = intervale_config["MONTH"]
            self.DAY = intervale_config["DAY"]
            self.STD_TYP = intervale_config["STD_TYP"]
            self.STP_NUM = intervale_config["STP_NUM"]
            self.SOOP_NAM = intervale_config["SOOP_NAM"]
            self.MISOSTUD_NUM = intervale_config["MISOSTUD_NUM"]
            self.in_name = intervale_config["in_name"]
            self.getfile_verbose = self.config["getfile_verbose"]
            self.selected_fits = getfiles(
                L2_folder=self.L2_folder,
                YEAR=self.YEAR,
                MONTH=self.MONTH,
                DAY=self.DAY,
                STD_TYP=self.STD_TYP,
                STP_NUM=self.STP_NUM,
                SOOP_NAM=self.SOOP_NAM,
                MISOSTUD_NUM=self.MISOSTUD_NUM,
                in_name=self.in_name,
                verbose=self.getfile_verbose,
            )
        elif self.SELECTION_MODE == "list":
            self.selected_fits = [
                Path(file)
                for file in self.config["file selection mode=> list"]["files"]
            ]
        elif self.SELECTION_MODE == "folder":
            self.config["folder"]
            self.L2_folder = Path(self.config["file selection mode=> folder"]["folder"])
            list_files = os.listdir(self.L2_folder)
            self.selected_fits = [self.L2_folder / file for file in list_files]
        else:
            raise ValueError("selection_mode must be ['intervale','folder','list']")
        for i, file in enumerate(self.selected_fits):
            if not Path(file).exists():
                raise Exception(f"{file} dosn't exists")

    def build_rasters(
        self, wvl_interval={"SW": slice(3, -3), "LW": slice(3, -3)}, line_catalogue=None
    ):
        """
        Create Run instances for each selected FITS file and configure their parameters.
        """
        if self.selected_fits is None:
            raise ValueError("files resolved yet run self.Build_file_list()")
        self.rasters = []
        for i, file in enumerate(self.selected_fits):

            # fit_args = gen_fit_inits(
            #     file ,
            #     conv_errors=self.conv_errors,
            #     verbose=self.geninits_verbose,
            #     )
            inits = GenInits(
                file,
                conv_errors=self.conv_errors,
                verbose=self.geninits_verbose,
                line_catalogue=line_catalogue,
                wvl_interval=wvl_interval,
            )
            inits.gen_inits()

            self.window_name = inits.windows_lines
            self.init_params = inits.init_params
            self.quentities = inits.quentities
            self.convolution_threshold = inits.convolution_threshold

            self.rasters.append(
                RasterFit(
                    path_or_hdul=file,
                    init_params=self.init_params,
                    quentities=self.quentities,
                    fit_func=flat_inArg_multiGauss,
                    windows_names=self.window_name,
                    bounds=None,
                    window_size=self.window_size,
                    convolution_function=lambda lst: np.zeros_like(lst[:, 2]) + 1,
                    convolution_threshold=self.convolution_threshold,
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
                    verbose=self.fit_verbose,
                )
            )
            pass

    def fuse_windows(self, indices):
        if not isinstance(indices[0], Iterable):
            indices = [indices] * len(self.rasters)
        for ind in range(len(self.rasters)):
            self.rasters[ind].fuse_windows(*indices[ind])

    def set_lock_protocol(
        self,
        window_type: "solo" or "fuse",
        window_index,
        lock_line1_index: int,
        lock_line2_index: int,
        lock_distance: float,
    ) -> None:
        for ind in range(len(self.rasters)):
            if window_type == "solo":
                self.rasters[ind].windows[window_index].lock_protocols.add_lock(
                    lock_line1_index, lock_line2_index, lock_distance
                )
            elif window_type == "fuse":
                self.rasters[ind].fused_windows[window_index].lock_protocols.add_lock(
                    lock_line1_index, lock_line2_index, lock_distance
                )
            else:
                raise Exception(
                    f"window_type is eather 'solo' or 'fuse' your value is {window_type}"
                )

    def run_preparations(self, redo=False):
        for i in range(len(self.rasters)):
            self.rasters[i].run_preparations(redo=redo)

    # TODO DELETE
    # def build_initial_parameters(self):
    #     """
    #     Build initial parameters for the runs.
    #     """
    #     for i in range(len(self.rasters)):
    #         self.rasters[i].build_initial_parameters()
    def fit_manager(self):
        """
        Execute the fitting process for all runs.
        """
        from ..fit_functions import ProgressFollower

        progress_follower = ProgressFollower()

        for i in range(len(self.rasters)):
            self.rasters[i].fit_raster(progress_follower=progress_follower)

    def __repr__(self) -> str:
        val = (
            "SELECTION_MODE          "
            + str(self.SELECTION_MODE)
            + "\n"
            + "preclean                "
            + str(self.preclean)
            + "\n"
            + "weights                 "
            + str(self.weights)
            + "\n"
            + "denoise                 "
            + str(self.denoise)
            + "\n"
            + "despike                 "
            + str(self.denoise)
            + "\n"
            + "convolute               "
            + str(self.denoise)
            + "\n"
            + "conv_errors             "
            + str(self.conv_errors)
            + "\n"
            + "convolution_extent_list "
            + str(self.convolution_extent_list)
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
            + "window_size             "
            + str(self.window_size)
            + "\n"
            + "Jobs                    "
            + str(self.Jobs)
            + "\n"
            + "geninits_verbose        "
            + str(self.geninits_verbose)
            + "\n"
            + "fit_verbose             "
            + str(self.fit_verbose)
            + "\n"
            + "len(rasters)            "
            + str(len(self.rasters))
            + "\n"
            "selected_fits           "
            + "\n"
            + "\n".join([str(file) for file in self.selected_fits])
            + "\n"
        )
        return val
