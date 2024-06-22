from pathlib import Path
import os
import copy
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.wcs import WCS
import pathlib
import numpy as np
from astropy.io import fits
from pathlib import Path, WindowsPath, PosixPath
import matplotlib.pyplot as plt
import os
from ..utils import normit, suppress_output, gen_axes_side2side, get_coord_mat
from collections.abc import Iterable
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
    def __init__(self, hdul_or_path):
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
        self._prepare_data(hdul_or_path)
        self.uncorrected_wavelength = None

    @property
    def wavelength(self):
        return self._all["int"].header["WAVELENGTH"]

    @property
    def observatory(self):
        return self._all["int"].header["OBSERVATORY"]

    @property
    def instrument(self):
        return self._all["int"].header["INSTRUMENT"]

    @property
    def ion(self):
        return self._all["int"].header["ION"]

    @property
    def line_id(self):
        return self._all["int"].header["LINE_ID"]

    @property
    def headers(self):
        return {
            "int": self._all["int"].header,
            "wav": self._all["wav"].header,
            "wid": self._all["wid"].header,
            "int_err": self._all["int_err"].header,
            "wav_err": self._all["wav_err"].header,
            "wid_err": self._all["wid_err"].header,
        }

    @property
    def obs_date(self):
        return self._all["int"].header["DATE-OBS"]

    def get_map(self, param="rad"):
        data = self[param]
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
        self._all["wav"].data = data
        return coeffs, errors

    def reset_doppler(self):
        try:
            self._all["wav"].data = self.uncorrected_wavelength.copy()
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
                return self._all["int"].data
            elif val == "wav":
                return self._all["wav"].data
            elif val == "wid":
                return self._all["wid"].data
            elif val == "int_err":
                return self._all["int_err"].data
            elif val == "wav_err":
                return self._all["wav_err"].data
            elif val == "wid_err":
                return self._all["wid_err"].data
            elif val == "rad":
                return self._all["rad"]
            elif val == "rad_err":
                return self._all["rad_err"]
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

        for hdu in hdul:
            if hdu.header["MEASRMNT"] == "bg":
                raise Exception("The background is not needed in this Object")
            self._all[hdu.header["MEASRMNT"]] = hdu

    def compute_params(
        self,
    ):
        if any([self._all[key] is None for key in ["int", "wav", "wid"]]):
            raise Exception(
                f"Call self.charge_data first because there is no {[key for key in ['int','wav','wid'] if self._all[key] is None]}"
            )
        self._all["rad"] = self["int"] * self["wid"] * np.sqrt(np.pi)
        self._all["rad_err"] = (
            self["int_err"] / self["int"] + self["wid_err"] / self["wid"]
        ) * self["rad"]

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


class SPICEL3Raster:
    def __init__(self, list_paths=None, folder_path=None):
        if (list_paths is None and folder_path is None) or (
            list_paths is not None and folder_path is not None
        ):
            raise Exception(
                "you need to specify strictly one of these arguments list_paths or folder_path"
            )
        elif folder_path is not None:
            list_paths = [str(file) for file in Path(folder_path).glob("*.fits")]
        else:
            # nothing to do if the list is given
            pass
        self.lines = []
        self.ll = None
        self.FIP_err = None
        self._prepare_data(list_paths)

    def _prepare_data(self, list_paths):
        for paths in list_paths:
            try:
                line = SPECLine(paths)
                self.lines.append(SPECLine(paths))
            except:
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

    def gen_compo_LCR(
        self,
        HFLines=None,
        LFLines=None,
        ll=None,
        suppressOutput=True,
        using_S_as_LF=True,
    ):
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
        logdens = 8.3  # try  9.5
        idens = np.argmin(abs(self.ll.density_array.value - 10**logdens))
        density_map = (
            10**logdens * np.ones(self.lines[0]["int"].shape, dtype=float) * u.cm**-3
        )

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
            + selected_lines[0].obs_date,
            va="top",
            ha="center",
        )

        return axes


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
