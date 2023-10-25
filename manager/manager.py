import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from ..fit_models import flat_inArg_multiGauss
from ..utils import getfiles
from ..init_handler import gen_fit_inits
from ..fit_functions.fit_raster import RasterFit
from typing import Union, List, Dict, Any, Callable, Tuple, Optional


class Manager():
    def __init__(self, Input_JSON: Optional[str] = None):
        """
        Initialize an instance of the Manager class.

        Args:
            Input_JSON (Optional[str]): Path to a JSON configuration file.
        """
        if Input_JSON is not None:
            # Read configuration file
            with open(Input_JSON) as config_file:
                self.config = json.load(config_file)
            
            #In the future you need to add a reference JSON whene there is 
            self.SELECTION_MODE          = self.config      ["SELECTION_MODE"      ]
            raster_args_confg            = self.config      ["fit_raster_args"]
            self.preadjust               = raster_args_confg['preadjust'           ]           
            self.preclean                = raster_args_confg['preclean'            ]            
            self.weights                 = raster_args_confg['weights'             ]             
            self.denoise                 = raster_args_confg['denoise'             ]             
            self.clipping_sigma          = raster_args_confg['clipping_sigma'      ]      
            self.clipping_med_size       = raster_args_confg['clipping_med_size'   ]   
            self.clipping_iterations     = raster_args_confg['clipping_iterations' ]
            self.mode                    = raster_args_confg["mode"                ]
            self.conv_errors             = raster_args_confg["conv_errors"         ]
            self.convolution_extent_list = np.array(raster_args_confg["convolution_extent_list"])
            self.save_data               = raster_args_confg["save_data"           ]
            self.save_plot               = raster_args_confg["save_plot"           ]
            self.plot_filename           = raster_args_confg["plot_filename"       ]
            self.data_filename           = raster_args_confg["data_filename"       ]
            self.plot_save_dir           = raster_args_confg["plot_save_dir"       ]
            self.data_save_dir           = raster_args_confg["data_save_dir"       ]
            self.forced_order            = raster_args_confg["forced_order"        ]
            self.quite_sun               = raster_args_confg["quite_sun"           ]
            self.window_size             = np.array(raster_args_confg["window_size"  ])
            self.select_window           = np.array(raster_args_confg["select_window"])
            self.show_ini_infos          = raster_args_confg["show_ini_infos"        ]
            self.Jobs                    = raster_args_confg["Jobs"                  ]     
            self.geninits_verbose        = self.config["geninits_verbose"            ]
            self.fit_verbose             = self.config["fit_verbose"                 ]
            self.describe_verbose        = self.config["describe_verbose"            ]
            
            self.rasters = []
            for directory in [self.plot_save_dir,self.data_save_dir]:
                if not Path(directory).exists():
                    print(f"the directory:{directory} doesn't exist creating it now")
                    os.mkdir(directory)
            
    def build_files_list(self):
        """
        Build the list of selected FITS files based on the chosen selection mode.
        """
        if self.SELECTION_MODE == "intervale": 
            intervale_config         = self.config     ["file selection mode=> date intervale"]
            self.L2_folder           = intervale_config['L2_folder'           ] 
            self.YEAR                = intervale_config['YEAR'                ]
            self.MONTH               = intervale_config['MONTH'               ]
            self.DAY                 = intervale_config['DAY'                 ]                 
            self.STD_TYP             = intervale_config['STD_TYP'             ]             
            self.STP_NUM             = intervale_config['STP_NUM'             ]             
            self.SOOP_NAM            = intervale_config['SOOP_NAM'            ]            
            self.MISOSTUD_NUM        = intervale_config['MISOSTUD_NUM'        ]        
            self.in_name             = intervale_config['in_name'            ]
            self.getfile_verbose     = self.config     ['getfile_verbose'     ]
            self.selected_fits = getfiles(
                L2_folder            = self.L2_folder       ,
                YEAR                 = self.YEAR            ,
                MONTH                = self.MONTH           ,
                DAY                  = self.DAY             ,
                STD_TYP              = self.STD_TYP         ,
                STP_NUM              = self.STP_NUM         ,
                SOOP_NAM             = self.SOOP_NAM        ,
                MISOSTUD_NUM         = self.MISOSTUD_NUM    ,
                in_name              = self.in_name         ,
                verbose              = self.getfile_verbose ,
                )
        elif self.SELECTION_MODE == "list":
            self.selected_fits       = [
                Path(file) for file in self.config["file selection mode=> list"]["files"]
                ]
        elif self.SELECTION_MODE == "folder":
            self.config["folder"]
            self.L2_folder           = Path(
                self.config["file selection mode=> folder"]["folder"]
                )
            list_files               = os.listdir(self.L2_folder)
            self.selected_fits       = [self.L2_folder/file for file in list_files]
        else: raise ValueError("selection_mode must be ['intervale','folder','list']")
        for i,file in enumerate(self.selected_fits):
            if not Path(file).exists(): raise Exception(f"{file} dosn't exists")
    def build_rasters(self):
        """
        Create Run instances for each selected FITS file and configure their parameters.
        """
        try: self.selected_fits 
        except: raise ValueError('files selected yet run self.Build_file_list() to have it')
        
        for i,file in enumerate(self.selected_fits):
            # self.RasterFit
            self.rasters.append(RasterFit(
                path_or_hdul             = file                          ,
                init_params              = ...                           ,#TODO                                         
                quentities               = ...                           ,#TODO                                         
                fit_func                 = ...                           ,#TODO    
                select_window            = self.select_window            ,     
                windows_names            = None                          ,                                
                bounds                   = None                          ,
                window_size              = self.window_size              ,                                
                convolution_function     = lambda lst:np.zeros_like(lst[:,2])+1,
                convolution_threshold    = self.conv_errors              ,                                
                convolution_extent_list  = self.convolution_extent_list  ,    
                mode                     = self.mode                     ,                         
                weights                  = self.weights                  ,                            
                denoise                  = self.denoise                  ,                            
                clipping_sigma           = self.clipping_sigma           ,                                   
                clipping_med_size        = self.clipping_med_size        ,                                      
                clipping_iterations      = self.clipping_iterations      ,                                        
                preclean                 = self.preclean                 ,                             
                save_data                = self.save_data                , 
                save_plot                = self.save_plot                ,                              
                plot_filename            = self.plot_filename            ,
                data_filename            = self.data_filename            ,             
                plot_save_dir            = self.plot_save_dir            ,                                  
                data_save_dir            = self.data_save_dir            ,                                  
                Jobs                     = self.Jobs                     ,                         
                verbose                  = self.fit_verbose              ,                            
                describe_verbose         = self.describe_verbose         ,                                     
                # forced_order             = self.forced_order             ,                                 
                # quite_sun                = self.quite_sun                ,                              
                # show_ini_infos           = self.show_ini_infos           ,                                   
                # geninits_verbose         = self.geninits_verbose         ,
            ))
            pass
    def build_initial_parameters(self):
        """
        Build initial parameters for the runs.
        """
        for i in range(len(self.rasters)):
            self.rasters[i].build_initial_parameters()
    def execute_all(self):
        """
        Execute the fitting process for all runs.
        """
        for i in range(len(self.rasters)):
            self.rasters[i].execute_fit()
    
# class Run():
#     def __init__(
#         self,
#         file_path              : str         = None,
#         preadjust              : bool        = None,
#         preclean               : bool        = None,
#         weights                : bool        = None,
#         denoise                : bool        = None,
#         clipping_sigma         : float       = None,
#         clipping_med_size      : int         = None,
#         clipping_iterations    : int         = None,
#         mode                   : str         = None,
#         conv_errors            : dict        = None,
#         convolution_extent_list: np.ndarray  = None,
#         save_data              : bool        = None,
#         save_plot              : bool        = None,
#         plot_save_dir          : str         = None,
#         data_save_dir          : str         = None,
#         plot_filename          : str         = None,
#         data_filename          : str         = None,
#         plot_kwargs            : dict        = None,
#         prefix                 : str         = None,
#         forced_order           : bool        = None,
#         quite_sun              : bool        = None,
#         window_size            : np.ndarray  = None,
#         show_ini_infos         : bool        = None,
#         Jobs                   : int         = None,
#         geninits_verbose       : int         = None,
#         verbose                : int         = None,
#         describe_verbose       : int         = None
#         ):
#         """
#         Initialize a Run instance.

#         Args:
#             file_path               (str)        : Path to the FITS file.
#             preadjust               (bool)       : Pre-adjust flag.
#             preclean                (bool)       : Pre-clean flag.
#             weights                 (bool)       : Weights flag.
#             denoise                 (bool)       : Denoise flag.
#             clipping_sigma          (float)      : Clipping sigma value.
#             clipping_med_size       (int)        : Clipping median filter size.
#             clipping_iterations     (int)        : Clipping iterations.
#             mode                    (str)        : Mode for fitting.
#             conv_errors             (dict)       : Dictionary of convolution errors.
#             convolution_extent_list (np.ndarray) : Array of convolution extent values.
#             save_data               (bool)       : Save data flag.
#             save_plot               (bool)       : Save plot flag.
#             plot_save_dir           (str)        : Plot save directory.
#             data_save_dir           (str)        : Data save directory.
#             plot_filename           (str)        : Plot filename template.
#             data_filename           (str)        : Data filename template.
#             plot_kwargs             (dict)       : Plotting keyword arguments.
#             prefix                  (str)        : Prefix for filenames.
#             forced_order            (bool)       : Forced order flag.
#             quite_sun               (bool)       : Quiet sun flag.
#             window_size             (np.ndarray) : Window size array.
#             show_ini_infos          (bool)       : Show initial information flag.
#             Jobs                    (int)        : Number of jobs.
#             geninits_verbose        (int)        : Verbose level for gen_fit_inits.
#             verbose                 (int)        : Verbose level for fitting.
#             describe_verbose        (int)        : Verbose level for describe_fit.
#         """ 
#         self.file_path               = file_path                     
#         self.preadjust               = preadjust                     
#         self.preclean                = preclean                       
#         self.weights                 = weights                         
#         self.denoise                 = denoise                         
#         self.clipping_sigma          = clipping_sigma           
#         self.clipping_med_size       = clipping_med_size     
#         self.clipping_iterations     = clipping_iterations 
#         self.mode                    = mode                               
#         self.conv_errors             = conv_errors                 
#         self.convolution_extent_list = convolution_extent_list                 
#         self.save_data               = save_data                     
#         self.save_plot               = save_plot                     
#         self.plot_save_dir           = plot_save_dir             
#         self.data_save_dir           = data_save_dir      
#         self.plot_filename           = plot_filename      
#         self.data_filename           = data_filename      
#         self.plot_kwargs             = plot_kwargs                 
#         self.prefix                  = prefix                           
#         self.forced_order            = forced_order               
#         self.quite_sun               = quite_sun                     
#         self.window_size             = window_size                 
#         self.show_ini_infos          = show_ini_infos           
#         self.Jobs                    = Jobs                   
#         self.geninits_verbose        = geninits_verbose               
#         self.verbose                 = verbose                         
#         self.describe_verbose        = describe_verbose   
#         self.init_params             = None
#         self.quentities              = None
#         self.convolution_threshold   = None
#         self.window_name             = None
        
#         self.filenames_generator()            
        
#     def build_initial_parameters(self):
#         """
#         Build initial parameters for the fitting process.
#         """
#         fit_args = gen_fit_inits(
#             self.file_path ,
#             conv_errors=self.conv_errors,
#             verbose=self.geninits_verbose,
#             )
        
#         self.window_name           = fit_args['windows_lines'        ]
#         self.init_params           = fit_args['init_params'          ]
#         self.quentities            = fit_args['quentities'           ]
#         self.convolution_threshold = fit_args['convolution_threshold']
    
#     def execute_fit(self):
#         """
#         Execute the fitting process using the configured parameters.
#         """
#         raise ('_fit_raster is deprecated change to OO version')
#         if self.init_params is None:self.build_initial_parameters()
#         # _fit_raster(
#         #     str(self.file_path)                                           ,                                                            
#         #     init_params             = self.init_params                    ,                                                      
#         #     fit_func                = flat_inArg_multiGauss               ,                                                      
#         #     quentities              = self.quentities                     ,                                                      
#         #     bounds                  = np.array([np.nan])                  , 
#         #     window_size             = self.window_size                    ,
#         #     convolution_function    = lambda lst:np.zeros_like(lst[:,2])+1,
#         #     convolution_threshold   = self.convolution_threshold          ,
#         #     convolution_extent_list = self.convolution_extent_list        ,
#         #     mode                    = self.mode                           ,
#         #     weights                 = self.weights                        ,
#         #     denoise                 = self.denoise                        ,
#         #     clipping_sigma          = self.clipping_sigma                 ,           
#         #     clipping_med_size       = self.clipping_med_size              ,              
#         #     clipping_iterations     = self.clipping_iterations            ,                
#         #     preclean                = self.preclean                       ,
#         #     preadjust               = self.preadjust                      , 
#         #     save_data               = self.save_data                      ,
#         #     save_plot               = self.save_plot                      ,           
#         #     prefix                  = self.prefix                         ,
#         #     plot_filename           = self.plot_filename                  ,
#         #     data_filename           = self.data_filename                  ,
#         #     quite_sun               = self.quite_sun                      ,
#         #     data_save_dir           = self.data_save_dir                  ,
#         #     plot_save_dir           = self.plot_save_dir                  ,
#         #     plot_kwargs             = self.plot_kwargs                    ,
#         #     show_ini_infos          = self.show_ini_infos                 ,
#         #     forced_order            = self.forced_order                   , 
#         #     Jobs                    = self.Jobs                           ,
#         #     verbose                 = self.verbose                        ,
#         #     describe_verbose        = self.describe_verbose               ,
            
#         # )
    
#     def filenames_generator(self):
#         """
#         Generate filenames using templates and replace placeholders.
#         """
        
#         if "::PARAMPLACEHOLDER" in self.data_filename:
#             self.data_filename = self.data_filename.replace("::PARAMPLACEHOLDER","{}")
#         if "::PARAMPLACEHOLDER" in self.plot_filename:
#             self.plot_filename = self.plot_filename.replace("::PARAMPLACEHOLDER","{}")
        
#         if "::SAMENAME" in self.plot_filename:
#             if "::SAMENAMEL2.5" in self.plot_filename:
#                 filename = self.file_path.stem
#                 filename = filename.replace("L2","L2.5")
#                 self.plot_filename = self.plot_filename.replace("::SAMENAMEL2.5",filename)
#             else:
#                 self.plot_filename = self.plot_filename.replace("::SAMENAME",self.file_path.stem)

#         if "::SAMENAME" in self.data_filename:
#             if "::SAMENAMEL2.5" in self.data_filename:
#                 filename = self.file_path.stem
#                 filename = filename.replace("L2","L2.5")
#                 self.data_filename = self.data_filename.replace("::SAMENAMEL2.5",filename)
#             else:
#                 self.data_filename = self.data_filename.replace("::SAMENAME",self.file_path.stem)

#         now = datetime.now()
#         formatted_time = now.strftime(r"%y%m%dT%H%M%S")
#         if "::TIME" in self.plot_filename:
#             self.plot_filename = self.plot_filename.replace("::TIME",formatted_time)
#         if "::TIME" in self.data_filename:
#             self.data_filename = self.data_filename.replace("::TIME",formatted_time)
        
#         strConv = "".join([f"{i:02d}"for i in self.convolution_extent_list ])
#         if "::CONV" in self.plot_filename:
#             self.plot_filename = self.plot_filename.replace("::CONV",strConv)
#         if "::CONV" in self.data_filename:
#             self.data_filename = self.data_filename.replace("::CONV",strConv)


