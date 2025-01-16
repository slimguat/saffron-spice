from pathlib import Path
import os
import random
import copy
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.wcs import WCS
import astropy
import pathlib
import numpy as np
from astropy.io import fits
from pathlib import Path, WindowsPath, PosixPath
import matplotlib.pyplot as plt
import os
from ..utils import normit, suppress_output, gen_axes_side2side, get_coord_mat,get_corner_HLP,get_lims,get_frame,reduce_largeMap_SmallMapFOV
from ..utils.fits_clone import HDUClone, HDUListClone
from ..utils import get_specaxis
from ..utils.utils import colored_text
from collections.abc import Iterable
from sunpy.map import GenericMap
import pandas as pd
from saffron.fit_models import ModelFactory
# from euispice_coreg.hdrshift.alignment import Alignment
# from euispice_coreg.plot.plot import PlotFunctions
# from euispice_coreg.utils.Util import AlignCommonUtil

from sunpy.net import Fido, attrs as a
from astropy.io import fits
from astropy.time import Time
import sunpy.map
import sunpy_soar

from astropy.visualization import (
    SqrtStretch,
    PowerStretch,
    LogStretch,
    AsymmetricPercentileInterval,
    ImageNormalize,
    MinMaxInterval,
    interval,
    stretch,
)
from sunpy.map import Map
from fiplcr import Line
from fiplcr import LinearComb as lc
from fiplcr import fip_map
import astropy.units as u
import math
import sunpy
from scipy.optimize import curve_fit


# TODO MOVE somewhere else
def FIP_error(ll, Errors, Datas):
    S_delHF = 0
    S_delLF = 0

    # S_LF = ll.lc_LF.value
    # S_HF = ll.lc_HF.value
    S_LF = 0
    S_HF = 0

    for i in range(len(ll.lines)):

        if i < ll.xLF.shape[1]:
            S_delLF += ll.ionsLF[i].coeff_map / ll.ionsLF[i].ph_abund * Errors[i]
            S_LF += ll.ionsLF[i].coeff_map / ll.ionsLF[i].ph_abund * Datas[i]
            # plt.figure()
            # plt.pcolormesh(S_delLF,vmin=0,vmax=150);plt.colorbar()
            # plt.title("S_delLF")
        else:
            S_delHF += (
                ll.ionsHF[i - ll.xLF.shape[1]].coeff_map
                / ll.ionsHF[i - ll.xLF.shape[1]].ph_abund
                * Errors[i]
            )
            S_HF += (
                ll.ionsHF[i - ll.xLF.shape[1]].coeff_map
                / ll.ionsHF[i - ll.xLF.shape[1]].ph_abund
                * Datas[i]
            )

    FIP_error = S_delHF / S_HF + S_delLF / S_LF
    if False:
        if False:
            norm = ImageNormalize(
                S_delHF,
                interval=AsymmetricPercentileInterval(1, 99),
                stretch=SqrtStretch(),
            )
            plt.figure()
            plt.pcolormesh(
                S_delHF,
                norm=norm,
                cmap="magma",
                # vmin=0,vmax=150
            )
            plt.colorbar()
            plt.title("S_delHF")

            plt.figure()
            norm = ImageNormalize(
                S_delLF,
                interval=AsymmetricPercentileInterval(1, 99),
                stretch=SqrtStretch(),
            )
            plt.pcolormesh(
                S_delLF,
                norm=norm,
                cmap="magma",
                # vmin=0,vmax=150
            )
            plt.colorbar()
            plt.title("S_delLF")

            plt.figure()
            norm = ImageNormalize(
                S_HF,
                interval=AsymmetricPercentileInterval(1, 99),
                stretch=SqrtStretch(),
            )
            plt.pcolormesh(
                S_HF,
                norm=norm,
                cmap="magma",
                # vmin=0,vmax=150
            )
            plt.colorbar()
            plt.title("S_HF")

            plt.figure()
            norm = ImageNormalize(
                S_LF,
                interval=AsymmetricPercentileInterval(1, 99),
                stretch=SqrtStretch(),
            )
            plt.pcolormesh(
                S_LF,
                norm=norm,
                cmap="magma",
                # vmin=0,vmax=150
            )
            plt.colorbar()
            plt.title("S_LF")

            plt.figure()
            norm = ImageNormalize(
                S_delLF / S_LF,
                interval=AsymmetricPercentileInterval(1, 99),
                stretch=SqrtStretch(),
            )
            plt.pcolormesh(
                S_delLF / S_LF,
                norm=norm,
                cmap="magma",
                # vmin=0,vmax=150
            )
            plt.colorbar()
            plt.title("S_delLF/S_LF")

            plt.figure()
            norm = ImageNormalize(
                S_delHF / S_HF,
                interval=AsymmetricPercentileInterval(1, 99),
                stretch=SqrtStretch(),
            )
            plt.pcolormesh(
                S_delHF / S_HF,
                norm=norm,
                cmap="magma",
                # vmin=0,vmax=150
            )
            plt.colorbar()
            plt.title("S_delHF/S_HF")

        plt.figure()
        norm = ImageNormalize(
            FIP_error,
            interval=AsymmetricPercentileInterval(1, 99),
            stretch=SqrtStretch(),
        )
        plt.pcolormesh(
            FIP_error,
            norm=norm,
            cmap="magma",
            # vmin=0,vmax=1
        )
        plt.colorbar()
        plt.title("FIP_error")

    return FIP_error


class SPECLine:
    def __init__(self, hdul_or_path,verbose=0,parent_raster = None):
        self._all = {
            "int": None,
            "wav": None,
            "wid": None,
            "rad": None,
            "int_err": None,
            "wav_err": None,
            "wid_err": None,
            "rad_err": None,
        }
        self.data_path = None
        self.uncorrected_wavelength = None
        self.verbose= verbose
        self.parent_raster = parent_raster
        self._prepare_data(hdul_or_path)
    @property
    def wavelength(self):
        return self.headers["int"]["WAVELENGTH"]

    @property
    def observatory(self):
        return self.headers["int"]["OBSERVATORY"]

    @property
    def instrument(self):
        return self.headers["int"]["INSTRUMENT"]

    @property
    def ion(self):
        return self.headers["int"]["ION"]

    @property
    def line_id(self):
        return self.headers['int']["LINE_ID"]

    @property
    def headers(self):
        return {
            "int": self._all["int"][1],
            "wav": self._all["wav"][1],
            "wid": self._all["wid"][1],
            "rad": self._all["rad"][1],
            "int_err": self._all["int_err"][1],
            "wav_err": self._all["wav_err"][1],
            "wid_err": self._all["wid_err"][1],
            "rad_err": self._all["rad_err"][1],
        }

    @property
    def obs_date(self):
        return np.datetime64(self.headers['int']["DATE-OBS"])

    @property
    def filename(self):
        return Path(self.data_path).name
    
    @property
    def model(self):
        return self._FIT_MODEL[0]
    
    @property
    def model_header(self):
        return self._FIT_MODEL[1]
    
    def get_map(self, param="rad",remove_dumbells=False):
        data = self[param].copy()
        if remove_dumbells:
            data[:100] = np.nan
            data[700:] = np.nan
        _map = Map(data, self.headers[param if "rad" not in param else "int"])
        if param in ["int", "rad", "wid"] or "err" in param:
            _map.plot_settings["cmap"] = "magma" if "err" not in param else "gray"
            _map.plot_settings["norm"] = normit(self[param][200:700])
        else:
            _map.plot_settings["cmap"] = "twilight_shifted"
            mean_val = np.nanmean(self[param][200:700])
            _100_kms_equivalent_doppler = 100e3 / 3e8 * mean_val
            _map.plot_settings["norm"] = normit(
                self[param],
                vmin=mean_val - _100_kms_equivalent_doppler,
                vmax=mean_val + _100_kms_equivalent_doppler,
                stretch=None,
            )
        _map.plot_settings["aspect"] = "auto"
        return _map

    def correct_doppler_gradient(self, direction="x", verbose=0, coeff=None):
        def linear_func(x, a, b):
            return a * x + b

        if direction not in ["x", "y", "xy"]:
            raise ValueError("direction should be one of these values: 'x','y','xy'")
        if self.uncorrected_wavelength is None:
            self.uncorrected_wavelength = self["wav"].copy()
        corrected_wavelength = self["wav"].copy() * np.nan
        data = self["wav"].copy()
        # print warning in red
        if data.shape[-1] <= 10 and "x" in direction and verbose > -2:
            print(
                "\033[93mWarning: Data is too small in x direction the doppler gradient estimation could be wrong\033[0m"
            )
        if data.shape[0] <= 10 and "y" in direction and verbose > -2:
            print(
                "\033[93mWarning: Data is too small in y direction the doppler gradient estimation could be wrong\033[0m"
            )
        coeffs = np.empty((len(direction),))
        errors = np.empty((len(direction),))
        if verbose > 2:
            fig, axis = plt.subplots(
                1, 1 + 2 * len(direction), figsize=(2 * (1 + 2 * len(direction)), 2)
            )
            _map = self.get_map("wav")
            norm = _map.plot_settings["norm"]
            cmap = _map.plot_settings["cmap"]
            axis[0].pcolormesh(self.uncorrected_wavelength, norm=norm, cmap=cmap)
        for ind, d in enumerate(direction):
            if d == "x":
                new_data = data.copy()
                new_data[:200] = np.nan
                new_data[700:] = np.nan
                xpix = np.arange(len(np.nanmean(new_data, axis=0)))
                mean_val = np.nanmean(new_data, axis=0)
                curve_fit_x = xpix[~np.isnan(mean_val)]
                curve_fit_y = mean_val[~np.isnan(mean_val)]
                if coeff is None:
                    coeff, var = curve_fit(
                        linear_func, curve_fit_x, curve_fit_y
                    )  # ,p0 = [curve_fit_y[0],2] )
                    data = (
                        data
                        - np.repeat(
                            coeff[0] * xpix.reshape(-1, 1), data.shape[0], axis=1
                        ).T
                    )
                    if verbose > 0:
                        print(
                            f"corrected {d} directoin\n doppler gradient is {coeff[0]}pixels/Angstrom"
                        )
                    coeffs[ind] = coeff[0]
                    errors[ind] = (var.diagonal() ** 0.5)[0]
                    if verbose > 2:
                        axis[ind + 1].pcolormesh(data, norm=norm, cmap=cmap)
                        axis[(ind + len(direction)) + 1].plot(curve_fit_x, curve_fit_y)
                        axis[(ind + len(direction)) + 1].plot(
                            curve_fit_x, linear_func(curve_fit_x, *coeff)
                        )
                else:
                    data = (
                        data
                        - np.repeat(
                            coeff * xpix.reshape(-1, 1), data.shape[0], axis=1
                        ).T
                    )
                    var = np.nan
                    coeffs[ind] = coeff
                    errors[ind] = np.nan

            elif d == "y":
                new_data = data.copy()
                new_data[:200] = np.nan
                new_data[700:] = np.nan
                ypix = np.arange(len(np.nanmean(new_data, axis=1)))
                mean_val1 = np.nanmean(new_data, axis=1)
                curve_fit_x1 = ypix[~np.isnan(mean_val1)]
                curve_fit_y1 = mean_val1[~np.isnan(mean_val1)]
                if coeff is None:

                    coeff1, var1 = curve_fit(
                        linear_func, curve_fit_x1, curve_fit_y1
                    )  # ,p0 = [curve_fit_y[0],2] )
                    data = (
                        data
                        - np.repeat(
                            coeff1[0] * ypix.reshape(1, -1), data.shape[1], axis=0
                        ).T
                    )
                    if verbose > 0:
                        print(
                            f"corrected {d} directoin\n doppler gradient is {coeff1[0]}pixels/Angstrom"
                        )
                    coeffs[ind] = coeff1[0]
                    errors[ind] = (var1.diagonal() ** 0.5)[0]

                    if verbose > 2:
                        axis[ind + 1].pcolormesh(data, norm=norm, cmap=cmap)
                        axis[(ind + len(direction)) + 1].plot(
                            curve_fit_x1, curve_fit_y1
                        )
                        axis[(ind + len(direction)) + 1].plot(
                            curve_fit_x1, linear_func(curve_fit_x1, *coeff1)
                        )
                else:
                    data = (
                        data
                        - np.repeat(
                            coeff * ypix.reshape(1, -1), data.shape[1], axis=0
                        ).T
                    )
                    var = np.nan
                    coeffs[ind] = coeff
                    errors[ind] = np.nan
        self._all["wav"][0] = data
        return coeffs, errors

    def reset_doppler(self):
        try:
            self._all["wav"][0] = self.uncorrected_wavelength.copy()
        except:
            pass

    def get_coord_mat(self, as_skycoord=False):
        data = self["int"]
        header = self.headers["int"]
        coord_matrix = get_coord_mat(Map(data, header), as_skycoord=as_skycoord)
        return coord_matrix

    def __getitem__(
        self,
        val: ["int", "wav", "wid", "rad", "int_err", "wav_err", "wid_err", "rad_err"],
    ):
        if isinstance(val, Iterable) and not isinstance(val, str):
            return [self.__getitem__(key) for key in val]
        else:
            if val == "int":
                return self._all["int"][0]
            elif val == "wav":
                return self._all["wav"][0]
            elif val == "wid":
                return self._all["wid"][0]
            elif val == "int_err":
                return self._all["int_err"][0]
            elif val == "wav_err":
                return self._all["wav_err"][0]
            elif val == "wid_err":
                return self._all["wid_err"][0]
            elif val == "rad":
                return self._all["rad"][0]
            elif val == "rad_err":
                return self._all["rad_err"][0]
            else:
                raise Exception(
                    f"{val} is not a valid keyword \nValid keywords: int,wav,wid,rad,int_err,wav_err,wid_err,rad_err"
                )

    def _prepare_data(self, hdul_or_path):
        self.charge_data(hdul_or_path)
        self.compute_params()

    def charge_data(self, hdul_or_path):
        # if isinstance(hdul_or_path,(str, PosixPath, WindowsPath, pathlib.WindowsPath,HDUList)):raise ValueError('The hdul_or_path sould be a list of 3')
        if isinstance(hdul_or_path, (str, PosixPath, WindowsPath, pathlib.WindowsPath)):
            hdul = fits.open(hdul_or_path)
            self.data_path = hdul_or_path
        elif isinstance(hdul_or_path, HDUList):
            hdul = hdul_or_path.copy()
        else:
            raise TypeError(str(hdul_or_path))
        all_background = True
        for ind,hdu in enumerate(hdul):
            if hdu.name == "FIT_MODEL":
                self._FIT_MODEL = [ModelFactory.from_hdu(hdu), hdu.header]
                continue
            if hdu.header["MEASRMNT"] == "bg":
                if self.parent_raster is not None and self.data_path is not None:
                    if Path(self.data_path).name not in self.parent_raster.backgrounds:
                        self.parent_raster.backgrounds[Path(self.data_path).name] = []
                    self.parent_raster.backgrounds[Path(self.data_path).name].append(HDUClone.from_hdu(hdu))
                    
                else:  
                    raise Exception("The background is not needed in this Object")
            else:
                all_background = False
            self._all[hdu.header["MEASRMNT"]] = [hdu.data.copy(), hdu.header]
        if all_background: raise Exception("tactical exit")
            
        if isinstance(hdul_or_path, (str, PosixPath, WindowsPath, pathlib.WindowsPath)):
            hdul.close()
        
    def compute_params(
        self,
    ):
        if any([self._all[key] is None for key in ["int", "wav", "wid"]]):
            raise Exception(
                f"Call self.charge_data first because there is no {[key for key in ['int','wav','wid'] if self._all[key] is None]}"
            )
        if (self._all["rad"] is None) or (self._all["rad_err"] is None):
            self._all["rad"] = [None,None]
            self._all["rad_err"] = [None,None]
            
            self._all["rad"][0] = self["int"] * self["wid"] * np.sqrt(np.pi)
            self._all["rad_err"][0] = (
                self["int_err"] / self["int"] + self["wid_err"] / self["wid"]
            ) * self["rad"]
            
            self._all["rad"]    [1] = self.headers["int"].copy()
            self._all["rad_err"][1] = self.headers["int_err"].copy()
            
            self._all["rad"]    [1]["MEASRMNT"] = "rad"
            self._all["rad"]    [1]["BTYPE"]    = 'Radiance'
            self._all["rad"]    [1]["BUNIT"]    = 'W/m2/sr' 
            self._all["rad_err"][1]["MEASRMNT"] = "rad_err"
            self._all["rad_err"][1]["BTYPE"]    = 'Radiance'
            self._all["rad_err"][1]["BUNIT"]    = 'W/m2/sr' 
        else:
            if self.verbose > 0:
                print("The rad and rad_err are already computed")                                            
            pass
            
    def plot(self, params="rad", axes=None, add_keywords=False):
        """_summary_

        Args:
            params (str or list, optional): all,int,wid,wav,rad. Defaults to 'all'.
            axes (list, optional): axes length = params. Defaults to None.
        """
        if params == "all":
            params = ["int", "wav", "wid", "rad"]
        if isinstance(params, str):
            params = [params]
        if axes is None:
            fig, axes = plt.subplots(1, len(params), figsize=(len(params) * 4, 4))
        if not isinstance(axes, Iterable):
            axes = [axes]
        if len(params) != 1:
            for ind, param in enumerate(params):
                self.plot(params=param, axes=axes[ind])
        else:
            if params[0] == "int":
                data = self["int"]
                norm = normit(data)
            elif params[0] == "wid":
                data = self["wid"]
                norm = normit(data, stretch=None)
            elif params[0] == "wav":
                data = self["wav"]
                norm = normit(data, stretch=None)
            elif params[0] == "rad":
                data = self["rad"]
                norm = normit(data)
            elif params[0] == "int_err":
                data = self["int_err"]
                norm = normit(data, AsymmetricPercentileInterval(5, 95))
            elif params[0] == "wav_err":
                data = self["wav_err"]
                norm = normit(data, AsymmetricPercentileInterval(5, 95))
            elif params[0] == "wid_err":
                data = self["wid_err"]
                norm = normit(data, AsymmetricPercentileInterval(5, 95))
            elif params[0] == "rad_err":
                data = self["rad_err"]
                norm = normit(data, AsymmetricPercentileInterval(5, 95))
            else:
                raise ValueError

            # map = Map(self['int'],self.headers['int'])
            lon, lat, time = get_celestial_L3(self)
            im = axes[0].pcolormesh(lon, lat, data, norm=norm, zorder=-1, cmap="magma")
            axes[0].set_title(self.line_id + "\n" + params[0])
            if add_keywords:
                pass

    def __repr__(self):
        # ANSI escape codes for bright green
        bright_green = "\033[92m"
        bright_yellow = "\033[93m"
        reset = "\033[0m"

        return (
            f"SPECLine object: ---------------------------\n"
            f"observatory,instrument={self.observatory},{self.instrument}\n"
            f"obs_date={bright_yellow}{self.obs_date}{reset},\n"
            f"ion,wavelength={bright_green}{self.ion}{reset},{bright_green}{self.wavelength}{reset},\n"
            "---------------------------------------------\n"
            ""
        )

    def write_data(self,file_name =None,overwrite=False):
        #create HDUList
        if file_name is None:
            if self.data_path is None:
                raise Exception("self.data_path is None and self.data_path is None")
            else:
                file_name = self.data_path
        
        hdu_list = [
            fits.PrimaryHDU(data=self['int'], header=self.headers['int'])
            ]
        for key in ["wav","wid","rad","int_err","wav_err","wid_err","rad_err"]:
            hdu_list.append(fits.ImageHDU(data=self[key], header=self.headers[key])) 
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(file_name, overwrite=overwrite)
    
    # define equality of lines 
    def __eq__(self, other):
        if not isinstance(other, SPECLine):
            return False
        return (
            self.observatory == other.observatory and 
            self.instrument == other.instrument and 
            self.obs_date == other.obs_date and 
            self.ion == other.ion and 
            self.wavelength == other.wavelength and 
            self.filename == other.filename
            )
class SPICEL3Raster:
    def __init__(self, list_paths=None, folder_path=None, verbose=0):
        if (list_paths is None and folder_path is None) or (
            list_paths is not None and folder_path is not None
        ):
            raise Exception(
                "you need to specify strictly one of these arguments list_paths or folder_path"
            )
        elif folder_path is not None:
            list_paths = [str(file) for file in Path(folder_path).glob("*.fits")]
            list_paths.sort()
        else:
            # nothing to do if the list is given
            pass
        self.lines = []
        self.ll             = None
        self.backgrounds    = {}
        self.FIP_err        = None
        self.FIP_header     = None    
        self.FIP_err_header = None
        self.density_header = None
        self.verbose = verbose
        
        self._prepare_data(list_paths)
        
        self.HFLines = None
        self.LFLines = None
        self.new_crvals = {"CRVAL1":None,"CRVAL2":None}
        self.old_crvals = {"CRVAL1":None,"CRVAL2":None}
        self.old_crvals['CRVAL1'] = self.lines[0].headers["int"]["CRVAL1"]
        self.old_crvals['CRVAL2'] = self.lines[0].headers["int"]["CRVAL2"]
        self.fsi174_path =  None
        self.L2_data = None
        self.L2_path = None
        self.params_matrix = None
    def _prepare_data(self, list_paths):
        for paths in list_paths:
            try:
            # if True:
                line = SPECLine(paths,verbose=self.verbose,parent_raster=self)
                self.lines.append(line)
            except Exception as e:
                # print(f"Couldn't load {paths} because of {e}")
                pass
        self.FIP_err = self.lines[0]["int"] * np.nan
        pass

    @property
    def FIP(self):
        try:
            res = self.ll.FIP_map.copy()
            if res is None:
                res = self.FIP_err.copy() + 1
            return res
        except:
            return self.FIP_err.copy() + 1
    
    def _gen_FIP_header(self,HFLines,LFLines):    
        "_______________________________________________________"
        #FIP header
        FIP_header = self.lines[0].headers["int"].copy()
        FIP_header["MEASRMNT"] = "fip"
        FIP_header["BTYPE"] = "FIP Bias"
        FIP_header["BUNIT"] = "" #it's a ratio so no unit
        del FIP_header['ION'] 
        del FIP_header['LINE_ID']  
        del FIP_header['WAVELENGTH'] 
        for i in range(len(HFLines)):
            FIP_header[f'HF_WAVE{i}'] = HFLines[i][1]
            FIP_header[f'HF_ION{i}' ] = HFLines[i][0]
            
        for i in range(len(LFLines)):
            FIP_header[f'LF_WAVE{i}'] = LFLines[i][1]
            FIP_header[f'LF_ION{i}' ] = LFLines[i][0]
        
        "_______________________________________________________"
        #FIP error header
        FIP_err_header = self.lines[0].headers["int"].copy()
        FIP_err_header["MEASRMNT"] = "fip_err"
        FIP_err_header["BTYPE"] = "FIP Bias error"
        FIP_err_header["BUNIT"] = "" #it's a ratio so no unit
        del FIP_err_header['ION'] 
        del FIP_err_header['LINE_ID']  
        del FIP_err_header['WAVELENGTH'] 
        for i in range(len(HFLines)):
            FIP_err_header[f'HF_WAVE{i}'] = HFLines[i][1]
            FIP_err_header[f'HF_ION{i}' ] = HFLines[i][0]
        
        for i in range(len(LFLines)):
            FIP_err_header[f'LF_WAVE{i}'] = LFLines[i][1]
            FIP_err_header[f'LF_ION{i}' ] = LFLines[i][0]
        
        "_______________________________________________________"
        #density header
        density_header = self.lines[0].headers["int"].copy()
        density_header["MEASRMNT"] = "den"
        density_header["BTYPE"]    = "Density"
        density_header["BUNIT"]    = "cm^-3"
        del density_header['ION']
        del density_header['LINE_ID']
        del density_header['WAVELENGTH']
        
        self.FIP_header     = FIP_header    
        self.FIP_err_header = FIP_err_header
        self.density_header = density_header

        
    def gen_compo_LCR(
        self,
        HFLines=None,
        LFLines=None,
        ll=None,
        suppressOutput=True,
        using_S_as_LF=True,
        density = 10**8.3,
    ):
        self.HFLines = HFLines
        self.LFLines = LFLines
        All_lines = list(HFLines)
        All_lines.extend(LFLines)
        if ll is None:
            self.ll = lc(
                [Line(ionid, wvl) for ionid, wvl in All_lines],
                using_S_as_LF=using_S_as_LF,
                
            )

            if suppressOutput:
                with suppress_output():
                    self.ll.compute_linear_combinations()
            else:
                self.ll.compute_linear_combinations()
        else:
            self.ll = copy.deepcopy(ll)
        # logdens = 8.3  # try  9.5
        # idens = np.argmin(abs(self.ll.density_array.value - 10**logdens))
        if isinstance(density,Iterable):
            density_map = density
        else:
            density_map = (
                density * np.ones(self.lines[0]["int"].shape, dtype=float) * u.cm**-3
            )
        self.density = density_map
        wvls = np.empty(shape=(len(self.lines),))
        for ind, line in enumerate(self.lines):
            wvls[ind] = line.wavelength

        data = []
        err = []
        for ind, ionLF in enumerate(self.ll.ionsLF):
            diff = np.abs(ionLF.wvl.value - wvls)
            argmin = np.argmin(diff)
            if diff[argmin] > 0.1:
                raise Exception(f"Haven't found an ion for: {ionLF} {wvls}")
            self.ll.ionsLF[ind].int_map = self.lines[argmin]["rad"] * u.W * u.m**-2 / 10
            data.append(self.lines[argmin]["rad"])
            err.append(self.lines[argmin]["rad_err"])
            # print(f"rad: {self.lines[argmin].hdul['int'][0].header['wavelength']},lc: {ionLF.wvl.value}")
        for ind, ionHF in enumerate(self.ll.ionsHF):
            diff = np.abs(ionHF.wvl.value - wvls)
            argmin = np.argmin(diff)
            if diff[argmin] > 0.1:
                raise Exception(f"Haven't found a for: {ionHF} {wvls}")
            self.ll.ionsHF[ind].int_map = self.lines[argmin]["rad"] * u.W * u.m**-2 / 10
            data.append(self.lines[argmin]["rad"])
            err.append(self.lines[argmin]["rad_err"])
            # print(f"rad: {self.lines[argmin].hdul['int'][0].header['wavelength']},lc: {ionHF.wvl.value}")

        fip_map(self.ll, density_map)
        if True:  # Complete calculation of error based on differentiation method
            self.FIP_err = (
                FIP_error(
                    self.ll,
                    err,
                    data,
                )
                / self.FIP
            )
        else:
            S_ind = np.argmin(np.abs(wvls - 750.22))
            self.FIP_err = self.lines[S_ind].rad_err / self.lines[S_ind].rad
        
        self._gen_FIP_header(HFLines=HFLines,LFLines=LFLines)
        
    def find_line(self, wvl):
        wvls = np.empty(shape=(len(self.lines),))
        for ind, line in enumerate(self.lines):
            wvls[ind] = line.wavelength
        diff = np.abs(wvls - wvl)
        ind = np.argmin(diff)
        if diff[ind] > 1:
            raise Exception(
                f"big difference {wvl}\n{wvls}",
            )
        return self.lines[ind]

    def show_all_wvls(self):
        wvls = np.empty(shape=(len(self.lines),))
        for ind, line in enumerate(self.lines):
            wvls[ind] = line.wavelength
        return wvls

    def get_coord_mat(self, as_skycoord=False):
        return self.lines[0].get_coord_mat(as_skycoord=as_skycoord)

    def _check_valide_ion_name(self, ion):
        try:
            assert isinstance(ion, str)
            # check if ion name is of the form element_number
            assert "_" in ion
            # separate ion by where there is _ character and generate a list of the form [element,number]
            ion = ion.split("_")
            assert len(ion) == 2
            # assert ion[0] is in the periodic table of elements
            assert ion[0] in [
                "h",
                "he",
                "li",
                "be",
                "b",
                "c",
                "n",
                "o",
                "f",
                "ne",
                "na",
                "mg",
                "al",
                "si",
                "p",
                "s",
                "cl",
                "ar",
                "k",
                "ca",
                "sc",
                "ti",
                "v",
                "cr",
                "mn",
                "fe",
                "co",
                "ni",
                "cu",
                "zn",
                "ga",
                "ge",
                "as",
                "se",
                "br",
                "kr",
                "rb",
                "sr",
                "y",
                "zr",
                "nb",
                "mo",
                "tc",
                "ru",
                "rh",
                "pd",
                "ag",
                "cd",
                "in",
                "sn",
                "sb",
                "te",
                "i",
                "xe",
                "cs",
                "ba",
                "la",
                "ce",
                "pr",
                "nd",
                "pm",
                "sm",
                "eu",
                "gd",
                "tb",
                "dy",
                "ho",
                "er",
                "tm",
                "yb",
                "lu",
                "hf",
                "ta",
                "w",
                "re",
                "os",
                "ir",
                "pt",
                "au",
                "hg",
                "tl",
                "pb",
                "bi",
                "po",
                "at",
                "rn",
                "fr",
                "ra",
                "ac",
                "th",
                "pa",
                "u",
                "np",
                "pu",
                "am",
                "cm",
                "bk",
                "cf",
                "es",
                "fm",
                "md",
                "no",
                "lr",
            ]
            # BITCH!!!! did copilot autocomplete just suggested the entire periodic table in the previous assertion, well that's dope????
            assert ion[1].isdigit()
            return True
        except:
            return False

    def search_lines(self, ion=None, wavelength=None, closest_wavelength=None):
        if ion is None and wavelength is None:
            raise Exception(
                "You need to specify one of these arguments ion or wavelength or both)"
            )
        if wavelength is not None and closest_wavelength is not None:
            raise Exception("Either specify wavelength or closest_wavelength not both")
        line_selection = self.lines

        if ion is not None:
            ion = ion.lower()
            if not (self._check_valide_ion_name(ion)):
                raise Exception(
                    'The ion name should be a string of Chianti structure ion naming "element_wvl" ex: "fe_18", "o_6"'
                )
            lines_selected_by_ions = []
            for line in line_selection:
                if line.ion == ion:
                    lines_selected_by_ions.append(line)
            line_selection = lines_selected_by_ions

        if wavelength is not None:
            lines_selected_by_wavelength = []
            for line in line_selection:
                if line.wavelength == wavelength:
                    lines_selected_by_wavelength.append(line)
            line_selection = lines_selected_by_wavelength
        elif closest_wavelength is not None:
            if len(line_selection) != 0:
                lines_selected_by_wavelength = []
                diff = np.abs(
                    np.array([line.wavelength for line in line_selection])
                    - closest_wavelength
                )
                min_diff = np.min(diff)
                index_diff = np.argmin(diff)
                line_selection = [line_selection[index_diff]]

        return line_selection

    def correct_doppler_gradient(
        self,
        direction="x",
        reference={"ion": "ne_8", "closest_wavelength": 770},
        verbose=0,
    ):
        # yellow colored warning
        if verbose > -1:
            print(
                f"\033[93mWarning: The correction of the doppler gradient for a raster is based on the line reference parameter ('the reference here is {reference}') it doesn't mean the correction is surely right\nIf you want to run gradiant correction for each line independetly call this method for the lines of your choice \033[0m"
            )
        ref_line = self.search_lines(**reference)
        if len(ref_line) == 0:
            raise Exception(f"Couldn't find the reference line {reference}")
        elif len(ref_line) > 1:
            raise Exception(
                f"Found more than one reference line {reference}\n{ref_line}"
            )
        else:
            ref_line = ref_line[0]

        coeff = ref_line.correct_doppler_gradient(direction=direction, verbose=verbose)
        for i in range(len(self.lines)):
            self.lines[i].correct_doppler_gradient(
                direction=direction, verbose=verbose, coeff=coeff[0]
            )

    def reset_doppler(self):
        for ind in range(len(self.lines)):
            self.lines[ind].reset_doppler()

    def plot(
        self,
        params="all",
        axes=None,
        lines=None,
        no_dumbells=False,
        col_row=[None, None],
    ):
        if lines is None:
            selected_lines = self.lines
        elif isinstance(lines, Iterable):
            selected_lines = []
            for element in lines:
                if isinstance(element, str):
                    sub_selected_lines = self.search_lines(ion=element)
                elif isinstance(element, float):
                    sub_selected_lines = self.search_lines(wavelength=element)
                elif isinstance(element, dict):
                    sub_selected_lines = self.search_lines(**element)
                else:
                    sub_selected_lines = self.search_lines(
                        ion=element[0], closest_wavelength=element[1]
                    )
                selected_lines.extend(sub_selected_lines)
        else:
            raise Exception(
                "lines should be a list of strings or floats or dictionaries"
            )
        if params == "all":
            params = ["int", "wav", "wid", "rad"]

        if isinstance(params, str):
            params = [params]
        if no_dumbells:
            slc = slice(200, 700)
        else:
            slc = slice(None, None)
        data = self.lines[0]["int"]
        header = self.lines[0].headers["int"]
        lon, lat = get_coord_mat(Map(data, header))
        data = data[slc]
        lon = lon[slc]
        lat = lat[slc]
        if axes is None:
            axes = []
            for ind in range(len(params)):
                if col_row[0] is None:
                    c = int(min(5, math.ceil(np.sqrt(len(selected_lines)))))
                else:
                    c = col_row[0]
                if col_row[1] is None:
                    r = len(selected_lines) // c + (
                        1 if len(selected_lines) % c != 0 else 0
                    )
                else:
                    r = col_row[1]

                inch_size = 2
                aspect = (np.max(lon) - np.min(lon)) / (np.max(lat) - np.min(lat))
                _axes = gen_axes_side2side(
                    r,
                    c,
                    figsize=(
                        c * (inch_size * aspect),
                        r * (inch_size),
                        #  r*(inch_size)
                    ),
                    wspace=0,
                    hspace=0,
                    top_pad=1 / (c * 2 * inch_size),
                    bottom_pad=1 / (c * 2 * inch_size),
                    left_pad=1 / (c * 2 * inch_size),
                    right_pad=1 / (c * 2 * inch_size),
                    aspect=aspect,
                ).flatten()
                for ax in _axes:
                    ax.tick_params(axis="both", which="major", labelsize=10)
                    ax.tick_params(axis="both", which="minor", labelsize=10)
                _axes[0].text(
                    0,
                    0.5,
                    "HLP longitude",
                    rotation=90,
                    va="center",
                    ha="left",
                    transform=_axes[0].figure.transFigure,
                )
                _axes[0].text(
                    0.5,
                    0,
                    "HLP latitude",
                    va="bottom",
                    ha="center",
                    transform=_axes[0].figure.transFigure,
                )
                for ax in _axes[::c]:
                    for tick in ax.get_yticklabels():
                        tick.set_rotation(90)
                        tick.set_ha("right")
                        tick.set_va("center")
                axes.append(_axes)

            axes = np.array(axes)

        for ind, param in enumerate(params):
            for ind2, line in enumerate(selected_lines):
                data = line[param][slc]
                map_ = line.get_map(param)
                norm = map_.plot_settings["norm"]
                cmap = map_.plot_settings["cmap"]
                axes[ind, ind2].pcolormesh(lon, lat, data, norm=norm, cmap=cmap)
                axes[ind, ind2].text(
                    0.5,
                    0.95,
                    selected_lines[ind2].line_id + " " + param,
                    transform=axes[ind, ind2].transAxes,
                    ha="center",
                    va="top",
                    bbox=dict(facecolor="white", alpha=0.5),
                )
            for ind3 in range(ind2 + 1, len(axes[ind])):
                axes[ind, ind3].remove()
        axes[0][0].figure.suptitle(
            selected_lines[0].observatory
            + " "
            + selected_lines[0].instrument
            + " "
            + str(selected_lines[0].obs_date),
            va="top",
            ha="center",
        )

        return axes

    def write_FIP_data(self,file_name ,overwrite=False):
        hdu = fits.PrimaryHDU(data=self.FIP    , header=self.FIP_header)
        hdu2 = fits.ImageHDU  (data=self.FIP_err, header=self.FIP_err_header)
        hdu3 = fits.ImageHDU  (data=self.density, header=self.density_header)
        if self.FIP_header is None:
            raise Exception("FIP is not yet computed run gen_compo_LCR")
        hdul = fits.HDUList([hdu,hdu2,hdu3])
        hdul.writeto(file_name, overwrite=overwrite)
    
    def _get_from_fido_closest_fsi174(self,raise_error=True):
        # Assuming self has an attribute or method to get the observation time
        if type(self).__name__ == SPICEL3Raster.__name__:
          obs_time = np.datetime64(np.datetime64(self.lines[0].headers['rad']['DATE-AVG']))  # Replace with actual method to get time
        else:
          obs_time = np.datetime64(self)
        time_range = a.Time(obs_time, obs_time + np.timedelta64(1,"h"))  # 1 hour range for example
        # Search for FSI data close to the self observation time
        instrument = a.Instrument('EUI')
        level = a.Level(1)
        product = a.soar.Product('EUI-FSI174-IMAGE')
        query = Fido.search(instrument & time_range & level & product)

        if len(query) == 0:
          if raise_error:
            raise ValueError("No FSI data found close to the specified time.")
          else:
            print("No FSI data found close to the specified time.")
            return None
        # Download the data
        downloaded_files = Fido.fetch(query[0][0])  # Download the first result

        if len(downloaded_files) == 0:
          if raise_error:
            raise ValueError("Failed to download FSI data.")
          else:
            print("Failed to download FSI data.")
            return None
        # Load the first downloaded file into an HDU list
        hdu_list = fits.open(downloaded_files[0])
        self.fsi174_path = downloaded_files[0]
        return hdu_list
    
    def coaline_with_FSI_171(self,source=None,index=1,verbose=0,):
      try:
        from euispice_coreg.hdrshift.alignment import Alignment
        from euispice_coreg.plot.plot import PlotFunctions
        from euispice_coreg.utils.Util import AlignCommonUtil
      except:
        raise Exception("Please install the euispice_coreg package by running the forked version of the package from the following link:\npip install git+https://github.com/slimguat/euispice_coreg.git")
        
      if True: #Get the large_FOV
        def _check_index(hdul,index):
          if index >= len(hdul):
            raise ValueError(f"Index {index} out of range. HDU list has {len(hdul)} HDUs.")
        if source is None:
          if self.fsi174_path is not None:
            if verbose>=1:print("Using the previously available FSI 174 image in self.fsi174_path")
            hdul = fits.open(self.fsi174_path)
            _check_index(hdul,index)
            hdu = hdul[index]
            header = hdu.header
            data = hdu.data
          else:
            if verbose>=1:print("No source provided, trying to get the closest FSI 174 image using sunpy.Fido.")
            hdul = self._get_from_fido_closest_fsi174(self)
            _check_index(hdul,index)
            hdu = hdul[index]
            header = hdu.header
            data = hdu.data
        elif isinstance(source, (str, PosixPath, WindowsPath,)):
          if verbose>=1:print("Source is a path to a fits file.")
          hdul = fits.open(source)
          _check_index(hdul,index)
          hdu = hdul[index] 
          header = hdu.header
          data = hdu.data
        elif isinstance(source,astropy.io.fits.hdu.hdulist.HDUList):
          if verbose>=1:print("Source is an HDUList.")
          hdul = source
          _check_index(hdul,index)
          hdu = hdul[index] 
          header = hdu.header
          data = hdu.data
        elif isinstance(source, astropy.io.fits.hdu.compressed.compressed.CompImageHDU):
          if verbose>=1:print('Source is a compressed image HDU.')
          hdu = source
          header = hdu.header
          data = hdu.data
        elif isinstance(source, GenericMap):
          if verbose>=1:print("Source is a GenericMap.")
          header = source.meta
          data = source.data
        else:
          raise ValueError("source must be a path to a fits file or an HDUList or HDU or a a GenericMap")
      if True:#cut_down_the_FOV 
        corrected_map = reduce_largeMap_SmallMapFOV(large_map = Map(hdu.data,hdu.header),small_map = self.lines[0].get_map(param='rad'),offset={"left": -150,"right":150,"top":150,"bottom":-150})
      "_________________________________________________________________________________" 
      "_________________________________________________________________________________"
      "_________________________________________________________________________________" 
      alignement_lines = [{'ion' : 'mg_9',"closest_wavelength": 706}, {"ion" : 'ne_8'}]
      "_________________________________________________________________________________"
      "_________________________________________________________________________________"
      "_________________________________________________________________________________"  
      selected_line = None
      for line in alignement_lines:
        lines = self.search_lines(**line)
        if len( lines) == 0:
          continue
        elif 'closest_wavelength' in line.keys():
          for line_ in lines: 
            if line_.wavelength- line['closest_wavelength'] < 1:
              selected_line = line_
              break
        else:
          selected_line = lines[0]
        if selected_line is not None:
          break
      if  selected_line is None:
        raise Exception('No lines match the list of lines to coalign with FSI 171')
      if verbose>=1: print('selected line for alignement', selected_line)
      if True:#Save the result into a temporary file
        tmp_location = Path('./tmp')
        tmp_location.mkdir(exist_ok=True)
        tobecorrected_map = selected_line.get_map()
        tobecorrected_path = tmp_location/f"{int(np.random.rand()*100000)}.fits"
        corrected_path = tmp_location/f"{int(np.random.rand()*100000)}.fits"
        if verbose>=1:
          print(f"saving the two maps into {tobecorrected_path} and  {corrected_path}")
        tobecorrected_map.save(tobecorrected_path )
        corrected_map    .save(corrected_path     )
      tobecorrected_path_ = tobecorrected_path 
      if True: #start aligning (Second round)
        lag_crval1s = [
          [-300,300,20],      [-50,50,4], [-10,10,1],
        ]
        lag_crval2s = [
          [-300,300,20],      [-50,50,4], [-10,10,1],
        ]
        for ind,(lag_crval1_,lag_crval2_) in enumerate(zip(lag_crval1s,lag_crval2s)): 
          print(f"correlation {ind+1}/{len(lag_crval1s)}: {lag_crval1_} {lag_crval2_}")
          lag_crval1 = np.arange(*lag_crval1_)
          lag_crval2 = np.arange(*lag_crval2_)    
          lag_cdelta1 = [0]
          lag_cdelta2 = [0]
          lag_crota = [0]
          A = Alignment(large_fov_known_pointing=corrected_path, small_fov_to_correct=tobecorrected_path_, lag_crval1=lag_crval1,
                    lag_crval2=lag_crval2, lag_cdelta1=lag_cdelta1, lag_cdelta2=lag_cdelta2, lag_crota=lag_crota,
                    parallelism=True, use_tqdm=True, counts_cpu_max=os.cpu_count(),
                    )
          corr = A.align_using_helioprojective(method='correlation')
          max_index = np.unravel_index(corr.argmax(), corr.shape)


          parameter_alignment = {
          "lag_crval1": lag_crval1,
          "lag_crval2": lag_crval2,
          "lag_crota": lag_crota,
          "lag_cdelta1": lag_cdelta1,
          "lag_cdelta2": lag_cdelta2,
          }

          PlotFunctions.plot_correlation(corr,  show=True,
                                path_save=os.path.join(tmp_location, f"{tobecorrected_path_.stem}_{lag_crval1[2]}_arcsec_correlation.pdf"), **parameter_alignment)
          PlotFunctions.plot_co_alignment(small_fov_window=-1, large_fov_window=-1, corr=corr,
                                      small_fov_path=tobecorrected_path_, large_fov_path=corrected_path, show=True,
                                      results_folder=tmp_location, levels_percentile=[95],
                                      **parameter_alignment)
          AlignCommonUtil.write_corrected_fits(path_l2_input=tobecorrected_path, window_list=[-1],
                                          path_l3_output=tobecorrected_path.parent/f'{tobecorrected_path.stem}_corrected_{lag_crval1[2]}_arcsec.fits', corr=corr,
                                          **parameter_alignment)
          tobecorrected_path_ = tobecorrected_path.parent/f'{tobecorrected_path.stem}_corrected_{lag_crval1[2]}_arcsec.fits'
        with fits.open(tobecorrected_path.parent/f'{tobecorrected_path.stem}_corrected_{lag_crval1[2]}_arcsec.fits') as hdul:
            self.new_crvals['CRVAL1'] = hdul[0].header['CRVAL1']
            self.new_crvals['CRVAL2'] = hdul[0].header['CRVAL2']
        return tobecorrected_map,corrected_map,A
    
    def set_coaligned_crvals(self):
        if self.new_crvals['CRVAL1'] is None:
            raise Exception('No new crvals are set run coaline_with_FSI_171 first')
        for i in range(len(self.lines)):
            for key in self.lines[i]._all.keys():
                self.lines[i]._all[key][1]['CRVAL1'] = self.new_crvals['CRVAL1']
                self.lines[i]._all[key][1]['CRVAL2'] = self.new_crvals['CRVAL2']
        if self.FIP_header is not None:
            self.FIP_header['CRVAL1'] = self.new_crvals['CRVAL1']
            self.FIP_err_header['CRVAL1'] = self.new_crvals['CRVAL1']
            self.density_header['CRVAL1'] = self.new_crvals['CRVAL1']
            
            self.FIP_header['CRVAL2'] = self.new_crvals['CRVAL2']
            self.FIP_err_header['CRVAL2'] = self.new_crvals['CRVAL2']
            self.density_header['CRVAL2'] = self.new_crvals['CRVAL2']
    
    def reset_crvals(self):
        for i in range(len(self.lines)):
            for key in self.lines[i]._all.keys():
                self.lines[i]._all[key][1]['CRVAL1'] = self.old_crvals['CRVAL1']
                self.lines[i]._all[key][1]['CRVAL2'] = self.old_crvals['CRVAL2']
        if self.FIP_header is not None:
            self.FIP_header['CRVAL1'] = self.old_crvals['CRVAL1']
            self.FIP_err_header['CRVAL1'] = self.old_crvals['CRVAL1']
            self.density_header['CRVAL1'] = self.old_crvals['CRVAL1']
            
            self.FIP_header['CRVAL2'] = self.old_crvals['CRVAL2']
            self.FIP_err_header['CRVAL2'] = self.old_crvals['CRVAL2']
            self.density_header['CRVAL2'] = self.old_crvals['CRVAL2']
    
    def init_L2_recall(self,L2_data=None):
        #starting by loading the L2_data
        if L2_data is None and self.L2_data is None:
            raise Exception('No L2 data is provided')
        elif L2_data is None:
            pass
        else:
            self.L2_data = L2_data
        if isinstance(self.L2_data, (str, PosixPath, WindowsPath, pathlib.WindowsPath)):
            self.L2_data = fits.open(self.L2_data)
            self.L2_path = self.L2_data
        else:
            self.L2_path = None
        # for each window we load the model
        # we search for all FIT_IDs so we can reconstruct the windows and the models
        FIT_IDs = set([line.model_header['FIT_ID'] for line in self.lines])
        if self.params_matrix is None:
            self.params_matrix = {FIT_ID:None for FIT_ID in FIT_IDs}
        # retrieving the windows 
        #first assemble each line with a FIT_ID
        window_fit = {FIT_ID:None for FIT_ID in FIT_IDs}
        
        #searching foir the lines that have the same FIT_ID
        for line in self.lines:
            #search the MODEL's FIT_ID
            FIT_ID = line.model_header['FIT_ID']
            SIB_keys = [key for key in line.model_header if 'SIB' in key]
            # print(SIB_keys)
            if window_fit[FIT_ID] is not None:
                continue
            
            #expeted siblings 
            siblings = np.array([[line.model_header[SIB_key],line.model_header["ORD"+SIB_key[3:]]] for SIB_key in SIB_keys],dtype=object).T
            # this will be filled with the lines index in SPICERaster object, and the oreders of the parameters of this line
            window_fit[FIT_ID] = np.empty((3,siblings.shape[1]),dtype=object)
            
            
            #search for the siblings
            for ind,line2 in enumerate(self.lines):
                if line2.filename in siblings[0]:
                    position = np.where(siblings[0] == line2.filename)[0][0]
                    order = siblings[1][position]
                    window_fit[FIT_ID][:,position] = ["line",ind,order]
            #searching for the backgrounds that have the same FIT_ID
            for background_name in self.backgrounds:
                if background_name in siblings[0]:
                    position = np.where(siblings[0] == background_name)[0][0]
                    order = siblings[1][position]
                    window_fit[FIT_ID][:,position] = ["background",background_name,order]
                
        #turning to multi index dataframe
        window_fit_rows = []
        for FIT_ID, fit_data in window_fit.items():
            if fit_data is not None:
                for col in range(fit_data.shape[1]):  # Iterate over the columns (siblings)
                    row = {
                        "FIT_ID": FIT_ID,
                        "FILE_TYPE": fit_data[0, col],  # Sibling filename
                        "LINE_ARG": fit_data[1, col],  # Line index in SPICERaster
                        "ORDER": fit_data[2, col],  # Order of parameters
                    }
                    window_fit_rows.append(row)

        # Convert to  DataFrame
        self.window_fit_df = pd.DataFrame(window_fit_rows)

    def reconstruct_window(self,windowindex=None,line=None,redo=False):
        #assertions not to call this function without initializing the L2 data either windowindex or line should be provided
        assert self.window_fit_df is not None, "init_L2_recall should be called first"
        assert (windowindex is not None or line is not None) and not (windowindex is not None and line is not None), "either windowindex or line should be provided"
        
        #find the EXTNAME of the window
        if windowindex is None:
            EXTNAME = line.headers['int']['L2WINDOW']
            WINDEX = [1 if hdu.header['EXTNAME'] == EXTNAME else 0 for hdu in self.L2_data].index(1)
            #finding the FIT_ID of the sample line
            FIT_ID = line.model_header['FIT_ID']
        else:
            EXTNAME = (self.L2_data[windowindex].header['EXTNAME'])
            WINDEX = windowindex
        
            # finding at least on line that has the same EXTNAME
            line = self.lines[[1 if (EXTNAME in (line.headers['int']['L2WINDOW'].split(',')))   else 0 for line in self.lines].index(1)]
            #now finding the FIT_ID of the sample line
            FIT_ID = line.model_header['FIT_ID']
        #getting the model
        model = line.model
        len_model = len(model.get_unlock_params())
        # model = ModelFactory.from_hdu(line._FIT_MODEL[1]      
        needed_lines = self.window_fit_df.query("FIT_ID == @FIT_ID")
        if self.params_matrix is not None and FIT_ID in self.params_matrix and not redo:
            # the code to print in red color is \033[91m and the code to reset the color is \033[0m
            # print(f"\033[91mThe window {FIT_ID} is already reconstructed if you want to redo it set redo=True\033[0m")
            colored_text(f"The window {FIT_ID} is already reconstructed if you want to redo it set redo=True","yellow")
            return needed_lines
        data = np.empty(([len_model,*line["int"].shape]),dtype=float)*0
        
        for row in needed_lines.iterrows():
            if row[1]["FILE_TYPE"] == "line":
                _line = self.lines[row[1]["LINE_ARG"]]
                order = [int(i) for i in (row[1]["ORDER"]).split(',')]
                keys = list(_line._all.keys())
                for ind in range(len(order)):
                    data[order[ind]] = _line[keys[ind]]
                L2window_index = _line.headers['int']['L2WINDOW'].split(",")
            elif row[1]["FILE_TYPE"] == "background":
                _background = self.backgrounds[row[1]["LINE_ARG"]]
                order = [int(i) for i in (row[1]["ORDER"]).split(',')]
                for ind in range(len(order)):
                    data[order[ind]] = _background[ind].data[0]
                L2window_index = _background[ind].header['L2WINDOW'].split(",")
                    
            self.params_matrix[FIT_ID] = {
                'data':data,
                'model':model,
                'L2window_name':L2window_index,
                'L2window_index': [[hdu.header['EXTNAME'] for hdu in self.L2_data].index(L2window_index_) for L2window_index_ in L2window_index],
                }
        return needed_lines
    
    def reconstruct_raster(self,redo=False):
        for window_index in range(len(self.L2_data)):
            if self.L2_data[window_index].header['EXTNAME'] not in ["VARIABLE_KEYWORDS", "WCSDVARR", "WCSDVARR"]:
                self.reconstruct_window(window_index,redo=redo)
    
    # def plot_pixels(self,list_indecies,window_index,axis=None):
    #     list_indecies = np.array(list_indecies)
    #     #Get FIT_ID based on the window_index
    #     FIT_IDs = list(self.params_matrix.keys())
    #     window_indecies = [self.params_matrix[FIT_ID]['L2window_index'] for FIT_ID in FIT_IDs]
    #     FIT_ID = FIT_IDs[[1 if window_index in windex else 0 for windex in window_indecies].index(1)]
    #     index_windows_involved = self.params_matrix[FIT_ID]['L2window_index']
    #     if len(index_windows_involved)>1:
    #         colored_text("The fit is involved in more than one window\nNot plotting them all","green") 
    #     #assert that the list_indecies is of shape N,2
    #     assert len(list_indecies.shape) == 2 and list_indecies.shape[1] == 2, "list_indecies should be of shape N,2"
    #     if axis is None:
    #         c = int(min(5, math.ceil(np.sqrt(len(list_indecies)))))
    #         r = int(np.ceil(len(list_indecies)/c))
    #         fig, axes = plt.subplots(r,c,figsize=(c*3,r*3))
    #         axes = axes.flatten()
    #         [ax.remove() for ax in axes[len(list_indecies):]]
    #         [ax.grid() for ax in axes]
    #     else:
    #         pass
        
    #     hdu = self.L2_data[window_index]
    #     specaxis = get_specaxis(hdu)
    #     model = self.params_matrix[FIT_ID]['model']
    #     function = model.callables['function']
    #     for ind,index in enumerate(list_indecies):
    #         data = self.L2_data[window_index].data[0,:,index[0],index[1]]
    #         params = self.params_matrix[FIT_ID]['data'][:,index[0],index[1]]
    #         lock_params = model.get_lock_params(params)
    #         axes[ind].step(specaxis,data,ls='--',color='black')
    #         axes[ind].plot(specaxis,function(specaxis,*lock_params),color='red')
    #         axes[ind].set_title(f"index: {index}")
    
    def plot_pixels(self, list_indices, window_index, axis=None):
        """
        Plot pixel fits for a given list of indices.

        Parameters:
            list_indices (list): List of (y, x) indices to plot.
            window_index (int): Index of the data window.
            axis (matplotlib axis): Axis object if provided.
        """
        list_indices = np.array(list_indices)

        # Get FIT_ID for the window
        FIT_ID = self._get_fit_id(window_index)
        involved_windows = self.params_matrix[FIT_ID]['L2window_index']
        if len(involved_windows) > 1:
            colored_text("The fit is involved in more than one window\nNot plotting them all", "green")

        # Validate list_indices shape
        assert len(list_indices.shape) == 2 and list_indices.shape[1] == 2, "list_indices should be of shape N,2"

        # Prepare axes
        fig, axes = self._prepare_axes(len(list_indices), axis)

        # Plot each pixel using _plot_pixel
        for ind, index in enumerate(list_indices):
            self._plot_pixel(index, window_index, FIT_ID, axes[ind]) 
        return axes
    
    def plot_random_pixels(self, num_lines, window_index, axis=None, exclude_nans=True):
        """
        Plot a random selection of pixel fits from a given window.

        Parameters:
            num_lines (int): Number of lines (pixels) to plot.
            window_index (int): Index of the data window.
            axis (matplotlib axis): Axis object if provided.
            exclude_nans (bool): Whether to exclude pixels with NaN values.
        """
        # Get the data for the window and the corresponding FIT_ID
        FIT_ID = self._get_fit_id(window_index)
        params_data = self.params_matrix[FIT_ID]['data']  # Shape: (num_params, y, x)
        
        # Get the data for the window
        hdu = self.L2_data[window_index]
        data = hdu.data[0]  # Shape: (spectra, y, x)

        # Efficiently generate all indices using NumPy
        y_size, x_size = data.shape[1:]  # Assume shape is (spectra, y, x)
        y_coords, x_coords = np.meshgrid(np.arange(y_size), np.arange(x_size), indexing='ij')
        all_indices = np.stack((y_coords.ravel(), x_coords.ravel()), axis=-1)  # Shape: (total_pixels, 2)

        if exclude_nans:
            # Create a mask for NaN values and filter indices
            nan_mask = ~np.any(np.isnan(params_data), axis=0).ravel()  # Shape: (total_pixels,)
            all_indices = all_indices[nan_mask]

        # Ensure there are enough indices to sample
        assert len(all_indices) >= num_lines, (
            f"Not enough valid pixels to plot. Available: {len(all_indices)}, Requested: {num_lines}"
        )

        # Randomly select `num_lines` indices
        random_indices = all_indices[np.random.choice(len(all_indices), num_lines, replace=False)]

        # Convert to list of tuples for compatibility with `plot_pixels`
        random_indices = [tuple(index) for index in random_indices]

        # Call `plot_pixels` with the random indices
        self.plot_pixels(random_indices, window_index, axis=axis)
    
    

    def _prepare_axes(self,num_plots, axis=None):
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
            axes = axes.flatten()
            [ax.remove() for ax in axes[num_plots:]]
            [ax.grid() for ax in axes[:num_plots]]
            return fig, axes[:num_plots]
        else:
            return None, [axis]


    def _get_fit_id(self, window_index):
        """
        Get the FIT_ID corresponding to a window index.

        Parameters:
            window_index (int): Index of the window.

        Returns:
            str: The FIT_ID for the given window index.
        """
        FIT_IDs = list(self.params_matrix.keys())
        window_indices = [self.params_matrix[FIT_ID]['L2window_index'] for FIT_ID in FIT_IDs]
        FIT_ID = FIT_IDs[[1 if window_index in windex else 0 for windex in window_indices].index(1)]
        return FIT_ID


    def _plot_pixel(self, index, window_index, FIT_ID, ax):
        """
        Plot a single pixel's data and fit on the given axis.

        Parameters:
            index (tuple): (y, x) coordinates of the pixel to plot.
            window_index (int): Index of the data window.
            FIT_ID (str): Identifier for the fitting model.
            ax (matplotlib axis): Axis to plot on.
        """
        if True:#Get the data
            hdu = self.L2_data[window_index]
            specaxis = get_specaxis(hdu)
            data = hdu.data[0, :, index[0], index[1]]
            params = self.params_matrix[FIT_ID]['data'][:, index[0], index[1]]
            model = self.params_matrix[FIT_ID]['model']
            function = model.callables['function']
            lock_params = model.get_lock_params(params)
            quentities = model.get_unlock_quentities()
            fitted_values = function(specaxis, *lock_params)

        ax.step(specaxis, data, ls='--', color='black')
        ax.plot(specaxis, fitted_values, color='red')
        for param in params[quentities=="x"]:
            if np.nanmin(specaxis)<=param<=np.nanmax(specaxis):
                ax.axvline(param,ls=':',color='blue')
        ax.set_title(f"index: {index}")
    
    

    
def get_celestial_L3(raster, **kwargs):
    if type(raster) == HDUList:

        shape = raster[0].data.shape
        wcs = WCS(raster[0].header)
        y = np.arange(shape[0], dtype=int)
        x = np.arange(shape[1], dtype=int)

        y = np.repeat(y, shape[1]).reshape(shape[0], shape[1])
        x = np.repeat(x, shape[0]).reshape(shape[1], shape[0])
        x = x.T
        lon, lat, time = wcs.wcs_pix2world(x.flatten(), y.flatten(), 0, 0)

        lon[lon > 180] -= 360
        lat[lat > 180] -= 360

        lon[lon < -180] += 360
        lat[lat < -180] += 360

        lon = lon.reshape(shape[0], shape[1]) * 3600
        lat = lat.reshape(shape[0], shape[1]) * 3600
        time = time.reshape(shape[0], shape[1])

    elif isinstance(raster, WCS):
        shape = kwargs["shape"]

        wcs = raster
        y = np.arange(shape[0], dtype=int)
        x = np.arange(shape[1], dtype=int)

        y = np.repeat(y, shape[1]).reshape(shape[0], shape[1])
        x = np.repeat(x, shape[0]).reshape(shape[1], shape[0])
        x = x.T
        lon, lat, time = wcs.wcs_pix2world(x.flatten(), y.flatten(), 0, 0)

        lon[lon > 180] -= 360
        lat[lat > 180] -= 360

        lon[lon < -180] += 360
        lat[lat < -180] += 360

        lon = lon.reshape(shape[0], shape[1]) * 3600
        lat = lat.reshape(shape[0], shape[1]) * 3600
        time = time.reshape(shape[0], shape[1])
    elif isinstance(raster, SPICEL3Raster):
        lon, lat, time = get_celestial_L3(raster.lines[0])
    elif isinstance(raster, SPECLine):
        shape = raster["int"].shape
        wcs = WCS(raster.headers["int"])
        lon, lat, time = get_celestial_L3(wcs, shape=shape)
    else:
        print(
            f"The raster passed doesn't match any known types: {type(raster)} but it has to be one of these types: \n{SPECLine}\n{SPICEL3Raster}\n{HDUList}"
        )
        raise ValueError("inacceptable type")
    return lon, lat, time


def filePath_manager(data_dir):
    files = os.listdir(data_dir)
    files.sort()
    file_cluster = []
    IDset = list(set([file[42:51] for file in files]))
    IDset.sort()
    for ind_ID, ID in enumerate(IDset):
        file_cluster.append([])
        filesByID = [file for file in files if file[42:51] in ID]
        ionIDset = set([file[73:-9] for file in filesByID if "B" not in file[73:-9]])
        ionIDset = list(ionIDset)
        ionIDset.sort()
        for ionID in ionIDset:
            filesByIonID = [
                data_dir / file for file in filesByID if file[73:-9] in ionID
            ]

            file_cluster[ind_ID].append([])
            file_cluster[ind_ID][-1] = filesByIonID
    return file_cluster
