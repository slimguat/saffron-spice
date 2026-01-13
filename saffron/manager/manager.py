import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path


import multiprocessing as mp
from multiprocessing import Process, Lock
from ..fit_models import flat_inArg_multiGauss
from ..utils import getfiles
# from ..utils.utils import default_convolution_function

# from ..init_handler import gen_fit_inits
from ..init_handler import GenInits
from ..fit_functions.fit_raster import RasterFit
from typing import Union, List, Dict, Any, Callable, Tuple, Optional, Iterable

import astropy.units as u

class Manager:
    def __init__(self, Input_JSON: Optional[str] = None):
        """
        Initialize an instance of the Manager class.

        If Input_JSON is provided, settings are read from that file.
        Otherwise default values are used.
        """
        # Set defaults
        self.SELECTION_MODE = "folder"
        self.preclean = True
        self.weights = True
        self.denoise = False
        self.despike = True
        self.convolute = False
        self.denoise_intervals = [6, 2, 1, 0, 0]
        self.clipping_sigma = 3
        self.clipping_med_size = [3, 6, 3, 3]
        self.clipping_iterations = 3
        self.mode = "box"
        self.conv_errors = {"I": 0.01,"x": 0.0001,"s": 0.01,"B": 100 }
        self.convolution_extent_list = np.array([0])
        self.t_convolution_index = 0
        self.save_data = True
        self.data_filename = "::SAMENAMEL2.5/con-::CONV_tcon-::TCONV/::SAMENAMEL2.5_::PARAMPLACEHOLDER_con.fits"
        self.data_save_dir = Path("./SAFFRON_results/")
        self.window_size = np.array([[0,None],[0,None]])
        self.time_size = np.array([0, None])
        self.Jobs = 8
        self.geninits_verbose = 3
        self.fit_verbose = 0
        self.selected_fits = []
        self.rasters = []

        # If a JSON config path is provided, override defaults
        if Input_JSON:
            with open(Input_JSON) as cfg:
                cfgd = json.load(cfg)
            self.config = cfgd
            self.SELECTION_MODE = cfgd.get("SELECTION_MODE", self.SELECTION_MODE)
            rargs = cfgd.get("fit_raster_args", {})
            self.preclean = rargs.get("preclean", self.preclean)
            self.weights = rargs.get("weights", self.weights)
            self.denoise = rargs.get("denoise", self.denoise)
            self.despike = rargs.get("despike", self.despike)
            self.convolute = rargs.get("convolute", self.convolute)
            self.conv_errors = rargs.get("conv_errors", self.conv_errors)
            self.convolution_extent_list = np.array(
                rargs.get("convolution_extent_list", self.convolution_extent_list)
            )
            self.t_convolution_index = rargs.get(
                "t_convolution_index", self.t_convolution_index
            )
            self.save_data = rargs.get("save_data", self.save_data)
            self.data_filename = rargs.get("data_filename", self.data_filename)
            self.data_save_dir = Path(rargs.get("data_save_dir", str(self.data_save_dir)))
            self.window_size = np.array(rargs.get("window_size", self.window_size))
            self.time_size = np.array(rargs.get("time_size", self.time_size))
            self.Jobs = rargs.get("Jobs", self.Jobs)
            self.geninits_verbose = cfgd.get("geninits_verbose", self.geninits_verbose)
            self.fit_verbose = cfgd.get("fit_verbose", self.fit_verbose)

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
        self, 
        wvl_interval=0.4, 
        # wvl_interval={"SW": slice(3, -3), "LW": slice(3, -3)}, 
        line_catalogue=None,
        extend_wvl_search = 0.5*u.Angstrom,
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
            inits.gen_inits(extend_wvl_search=extend_wvl_search)

            # self.window_name = inits.windows_lines
            # self.init_params = inits.init_params
            # self.quentities = inits.quentities
            self.models = inits.Models
            self.convolution_threshold = inits.convolution_threshold
            
            self.rasters.append(
                RasterFit(
                    path_or_hdul=file,
                    # init_params=self.init_params,
                    # quentities=self.quentities,
                    # fit_func=flat_inArg_multiGauss,
                    # windows_names=self.window_name,
                    # bounds=None,
                    models = self.models,
                    window_size=self.window_size,
                    time_size = self.time_size,
                    # convolution_function = default_convolution_function,
                    convolution_threshold=self.convolution_threshold,
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
                    verbose=self.fit_verbose,
                )
            )
            
            pass

    def fuse_windows(self, indices):
        if not isinstance(indices[0], Iterable):
            indices = [indices] * len(self.rasters)
        for ind in range(len(self.rasters)):
            self.rasters[ind].fuse_windows(*indices[ind])

    def lock(
        self,
        window_type: "solo" or "fuse",
        window_index: int,
        param_1: dict,
        param_2: dict,
        lock_protocol: dict,
    ) -> None:
      '''
      This function is used to lock two parameters of the model together.
      Parameters
      ----------
      window_type : str ["solo" or "fuse"]
      The type of the window that you want to lock the parameters in. It can be either "solo" or "fuse".
      window_index : int
      The index of the window that you want to lock the parameters in.
      param_1 : 
        dict must contain the following keys: "element_index", "parameter"
        A dictionary that contains the information of the first parameter that you want to lock.
      param_2 : dict 
        must contain the following keys: "element_index", "parameter"
        A dictionary that contains the information of the second parameter that you want to lock.
      lock_protocol : dict 
        must contain the following keys: "operation", "value" operation: str
        "operation" It can be either "add" or "mul".
        A dictionary that contains the information of the lock protocol.
      '''
      assert "element_index" in param_1.keys() and "parameter" in param_1.keys() 
      assert "element_index" in param_2.keys() and "parameter" in param_2.keys() 
      assert "operation" in lock_protocol.keys() and "value" in lock_protocol.keys()
      
      param_1['model_type'] = "gaussian"
      param_2['model_type'] = "gaussian"
      
      if param_1 == param_2:raise ValueError(f"Cannot lock a parameter to itself\nparam_1={param_1}\nparam_2={param_2}")
      
      for ind in range(len(self.rasters)):
          if window_type == "solo":
              self.rasters[ind].windows[window_index].model.lock(param_1, param_2, lock_protocol)
            
          elif window_type == "fuse":
              self.rasters[ind].fused_windows[window_index].model.lock(
                  param_1, param_2, lock_protocol
              )
          else:
              raise Exception(
                  f"window_type is eather 'solo' or 'fuse' your value is {window_type}"
              )

    def run_preparations(self, redo=False,max_processes=mp.cpu_count()):
        
        for i in range(len(self.rasters)):
            self.rasters[i].run_preparations(redo=redo,max_processes=max_processes)

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

