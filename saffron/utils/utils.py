import math
import numpy as np
from scipy.ndimage import uniform_filter
import cv2
from time import sleep
from datetime import datetime
from typing import Union, List, Dict, Any, Callable, Tuple, Optional, Iterable

import astropy
import os
import sys

from astropy.io import fits as fits_reader
from astropy.io.fits.hdu.image import PrimaryHDU, ImageHDU
from astropy.io.fits import HDUList
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
from astropy.wcs import WCS

import sunpy

import ndcube

from numba import jit
from multiprocess.shared_memory import SharedMemory

import matplotlib.pyplot as plt

import os
import contextlib
import shutil
import pkg_resources
from pathlib import Path, PosixPath, WindowsPath

from pathlib import Path, WindowsPath, PosixPath
from sunpy.map import GenericMap
import astropy 
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.units.quantity import Quantity
from sunpy.map import Map
from IPython.display import HTML, display
import colorama
import matplotlib.dates as mdates

def colored_text(text, color):
    """
    Display colored text dynamically based on the environment (Jupyter or standard terminal).
    
    Parameters:
        text (str): The text to display.
        color (str): The color name (e.g., 'red', 'green', 'blue') or a hex code (e.g., '#FF0000').
    """
    print(text)
    return
    def is_hex_color(c):
        """Check if the string is a valid hex color."""
        if c.startswith("#") and len(c) == 7:
            try:
                int(c[1:], 16)  # Validate hex value
                return True
            except ValueError:
                return False
        return False

    def colored_text_terminal(text, hex_color):
        """Display text with a True Color ANSI escape sequence in the terminal."""
        # Convert hex to RGB
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        # Generate ANSI escape sequence
        ansi_color = f"\033[38;2;{r};{g};{b}m"
        reset = "\033[0m"
        # Print colored text
        print(f"{ansi_color}{text}{reset}")

    try:
        # Detect if running in an IPython or Jupyter environment
        ipython_shell = get_ipython().__class__.__name__
        if ipython_shell in ['ZMQInteractiveShell', 'GoogleColabShell', 'InteractiveShellEmbed']:
            # Jupyter Notebook, Google Colab, or Embedded Shell
            from IPython.display import HTML, display
            if is_hex_color(color):
                display(HTML(f'<span style="color: {color}; font-weight: bold;">{text}</span>'))
            else:
                display(HTML(f'<span style="color: {color}; font-weight: bold;">{text}</span>'))
        else:
            raise NameError("Not a Jupyter-like environment")
    except NameError:
        # Standard terminal or non-Jupyter environment
        try:
            from colorama import Fore, Style, init
            init(autoreset=True)  # Initialize Colorama for Windows compatibility

            if is_hex_color(color):
                # Use the enhanced hex-to-ANSI function for terminals
                colored_text_terminal(text, color)
            else:
                # Handle named colors with Colorama
                colors = {
                    'red': Fore.RED,
                    'green': Fore.GREEN,
                    'blue': Fore.BLUE,
                    'yellow': Fore.YELLOW,
                    'magenta': Fore.MAGENTA,
                    'cyan': Fore.CYAN,
                    'white': Fore.WHITE,
                    'black': Fore.BLACK,
                }
                print(colors.get(color.lower(), Fore.WHITE) + text + Style.RESET_ALL)
        except ImportError:
            # Fallback if colorama is not available
            print(text)

if True: #FOV functions
  def draw_FOV(obj_map:Map):
      
      pix_vertices = np.array([
          [0                    ,0                    ],
          [0                    ,obj_map.data.shape[1]],
          [obj_map.data.shape[0],obj_map.data.shape[1]],
          [obj_map.data.shape[0],0                    ],
      ])
      
      hlp_vertices =np.array(obj_map.wcs.wcs_pix2world(pix_vertices[:,1],pix_vertices[:,0],0))
      
      # Draw the rectangle around the FOV
      hlp_vertices[hlp_vertices>180 ] -= 360
      hlp_vertices[hlp_vertices<-180] += 360
      hlp_vertices*=3600
          
      polygon = gen_polygone(
          vertices = hlp_vertices.T
          ,unit=u.arcsec
          )
      polySkycoord = SkyCoord(polygon[0],polygon[1],
                          frame=obj_map.coordinate_frame)
      return polySkycoord 
  def gen_polygone(vertices,point_num_per_side=100,unit=1):
      X,Y = [],[]
      vertices = [*vertices]
      vertices.append(vertices[0])
      for i in range(1,len(vertices)):
          x,y = gen_line(vertices[i-1],vertices[i],num=point_num_per_side,unit=1)
          X.extend(list(x.copy()))
          Y.extend(list(y.copy()))
      try: return X*unit,Y*unit
      except: return [x *unit for x in X],[y *unit for y in Y]
  def gen_line(start,end,num=100,unit=1):
      
      a = (start[1]-end[1])/(start[0]-end[0]) 
      b = start[1] - a*start[0]
      x = np.linspace(start[0],end[0],num=num)
      y = a*x+b
      if np.isnan(y).any():
          y,x = gen_line(start[::-1],end[::-1],num=num,unit=unit) 
          
      return x*unit,y*unit

if True: #reduce map functions
  def get_corner_HLP(FOV,outer_rectangle=True):
      min_lon_arcsec = np.min(FOV.spherical.lon.arcsec)
      max_lon_arcsec = np.max(FOV.spherical.lon.arcsec)
      min_lat_arcsec = np.min(FOV.spherical.lat.arcsec)
      max_lat_arcsec = np.max(FOV.spherical.lat.arcsec)
      
      if not outer_rectangle:
          side_size= FOV.spherical.lon.arcsec.shape[0]//4
          min_lon_arcsec  = np.max(FOV.spherical.lon.arcsec[3*side_size:3*side_size+99])
          max_lon_arcsec  = np.min(FOV.spherical.lon.arcsec[1*side_size:1*side_size+99])
          min_lat_arcsec  = np.max(FOV.spherical.lat.arcsec[0*side_size:0*side_size+99])
          max_lat_arcsec  = np.min(FOV.spherical.lat.arcsec[2*side_size:2*side_size+99])
          
      corners = SkyCoord([min_lon_arcsec,min_lon_arcsec,max_lon_arcsec,max_lon_arcsec]*u.arcsec,
                  [min_lat_arcsec,max_lat_arcsec,max_lat_arcsec,min_lat_arcsec]*u.arcsec,
                  frame=FOV.frame)
      return  corners
  def get_lims(coords:SkyCoord,map):
    """For Sunpy maps"""
    xlim_pix1, ylim_pix1 = get_frame(
        [min(coords.spherical.lon.arcsec), 
        max(coords.spherical.lon.arcsec)]*u.arcsec,
        [min(coords.spherical.lat.arcsec), 
        max(coords.spherical.lat.arcsec)]*u.arcsec,
        map=map)
    return xlim_pix1, ylim_pix1
  def get_frame(xlims_world=[0,0]*u.arcsec,ylims_world=[0,0]*u.arcsec,map=None):
    try:
        world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=map.coordinate_frame)
    except:
        world_coords = SkyCoord(lon=xlims_world, lat=ylims_world, frame=map.coordinate_frame)

    pixel_coords = map.world_to_pixel(world_coords)

    # we can then pull out the x and y values of these limits.
    xlims_pixel = pixel_coords.x.value
    ylims_pixel = pixel_coords.y.value
    return xlims_pixel,ylims_pixel
  def reduce_largeMap_SmallMapFOV(large_map,small_map,offset = None):
      FOV = draw_FOV(small_map)
      FOV_small_inlarge = FOV.transform_to(large_map.coordinate_frame)
      small_corners_inlarge = get_corner_HLP(FOV_small_inlarge)
      xlim, ylim = get_lims(small_corners_inlarge, large_map)
      if type(offset) is not dict: offset = {"left":offset[0],"right":offset[0],"top":offset[1],"bottom":offset[1]}
      # print(u.arcsec if isinstance(offset["left"],Quantity) else 1)
      # print(1*(u.arcsec if isinstance(offset["left"],Quantity) else 1)).to(u.arcsec)
      submap_bottom_left = SkyCoord(
          np.min(FOV_small_inlarge.spherical.lon.arcsec)*u.arcsec + ((offset["left"]*(u.arcsec if not isinstance(offset["left"],Quantity) else 1)).to(u.arcsec) if offset is not None else 0),
          np.min(FOV_small_inlarge.spherical.lat.arcsec)*u.arcsec + ((offset["bottom"]*(u.arcsec if not isinstance(offset["bottom"],Quantity) else 1)).to(u.arcsec) if offset is not None else 0),
          frame = large_map.coordinate_frame,
          )
      submap_top_right   = SkyCoord(
          np.max(FOV_small_inlarge.spherical.lon.arcsec)*u.arcsec + ((offset["right"]*(u.arcsec if not isinstance(offset["left"],Quantity) else 1)).to(u.arcsec) if offset is not None else 0),
          np.max(FOV_small_inlarge.spherical.lat.arcsec)*u.arcsec + ((offset["top"]*(u.arcsec if not isinstance(offset["left"],Quantity) else 1)).to(u.arcsec) if offset is not None else 0),
          frame = large_map.coordinate_frame,
      )
      
      sublarge_map = large_map.submap(submap_bottom_left, top_right = submap_top_right)
      return sublarge_map

def gen_axes_side2side(
    row=1,
    col=1,
    figsize=None,
    wspace=0,
    hspace=0,
    top_pad=0,
    bottom_pad=0,
    right_pad=0,
    left_pad=0,
    sharex=True,
    sharey=True,
    aspect=1,
    ax_size=5,
):
    if not isinstance(wspace, Iterable):
        wspaces = []
        for i in range(col - 1):
            wspaces.append(wspace)
    else:
        wspaces = wspace

    if not isinstance(hspace, Iterable):
        hspaces = []
        for i in range(row - 1):
            hspaces.append(hspace)
    else:
        hspaces = hspace
    assert len(hspaces) == row - 1
    assert len(wspaces) == col - 1
    effective_size = (1 - right_pad - left_pad, 1 - top_pad - bottom_pad)
    ax_w = (effective_size[0] - np.sum(wspaces)) / col
    ax_h = (effective_size[1] - np.sum(hspaces)) / row

    if figsize is None:
        h = ax_size * row
        ratio = ax_h / ax_w
        w = h * ratio * aspect
    else:
        w, h = figsize
    fig = plt.figure(figsize=(w, h))

    axes = np.array(np.zeros(shape=(row, col)), dtype="O")
    for i in range(row):
        for j in range(col):
            if i == 0:
                y0 = bottom_pad
            else:
                y0 = axes[i - 1][j].get_position().y0 + hspaces[i - 1] + ax_h
            if j == 0:
                x0 = left_pad
            else:
                x0 = axes[i][j - 1].get_position().x0 + wspaces[j - 1] + ax_w

            rect = [
                x0,
                y0,
                ax_w,
                ax_h,
            ]

            axes[i][j] = fig.add_axes(rect,)
    axes = axes[::-1]
    if sharex or sharey:
        for i, row_ax in enumerate(axes):
            for j, ax in enumerate(row_ax):
                if sharex and i != len(axes) - 1:
                    ax.set_xticklabels([])
                if sharey and j != 0:
                    ax.set_yticklabels([])
    return axes

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
        if (hdu.header["EXTNAME"]
        not in ["VARIABLE_KEYWORDS", "WCSDVARR", "WCSDVARR"]) and ('SATPIXLIST' not in hdu.header["EXTNAME"])
    ]
    return unq
def get_data_raster(hdul: HDUList) -> List[np.ndarray]:
    unq = get_extnames(hdul)
    raster = [hdu for hdu in hdul if hdu.header["EXTNAME"] in unq] 
    return raster    
    
def get_coord_mat(map, as_skycoord=False):
    res = sunpy.map.maputils.all_coordinates_from_map(map)
    if as_skycoord:
        return res
    try:
        lon = res.spherical.lon.arcsec
        lat = res.spherical.lat.arcsec
    except AttributeError:
        lon = res.lon.value
        lat = res.lat.value
    return lon, lat

def function_to_string(func):
    source_lines, _ = inspect.getsourcelines(func)
    return "".join(source_lines)

def flatten(iterable):
    flattened = []
    for item in iterable:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened

def ArrToCode(arr):
    # Check if the input is a numpy array
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")
    # Convert the array to a string
    arr_str = f"np.array({arr.tolist()})"
    arr_str = arr_str.replace("nan", "np.nan")
    return arr_str

def prepare_filenames(
    prefix=None,
    data_filename=None,
    plot_filename=None,
    data_save_dir="./.p/",
    plot_save_dir="./imgs/",
    i=None,
    verbose=0,
):

    if type(prefix) == str:
        filename = prefix + "_window_{:03d}_" + "{:}.p"
    elif prefix == None:
        dir = data_save_dir
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir_list = os.listdir(dir)
        j = 0
        for file in dir_list:
            try:
                j2 = int(file[0:3])
                if j2 >= j:
                    j = j2 + 1

            except Exception:
                pass
        j3 = j
        dir2 = dir
    if type(plot_filename) == str:
        if (
            plot_filename.format(" ", 0, 0) == plot_filename
        ):  # make sure this passed variable is subscriptable
            filename_a = plot_filename + "plot_{:03d}_{}_{}.jpg"
            filename_b = plot_filename + "hist_{:03d}_{}_{}.jpg"

    elif prefix == None:
        dir = plot_save_dir
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir_list = os.listdir(dir)
        j = 0
        for file in dir_list:
            try:
                j2 = int(file[0:3])

                if j2 >= j:
                    j = j2 + 1

            except Exception:
                pass
        j = max(j3, j)
        # Delete these later------
        j = i if type(i) != type(None) else j
        if verbose >= 1:
            print("working in the file with prefix i={:03d} ".format(j))
        # ------------------------
        filename_a = dir + "{:03d}_".format(j) + "plot_{:03d}_" + "{}_{}.jpg"
        filename_b = dir + "{:03d}_".format(j) + "hist_{:03d}_" + "{}_{}.jpg"
        filename = dir2 + "{:03d}_".format(j) + "window_{:03d}_" + "{}_{}.p"
    # print(data_filename)
    if type(data_filename) != type(None):
        dir2 = data_save_dir
        dir = plot_save_dir
        flnm = data_filename
        if flnm[-2:] == ".p":
            flnm = flnm[:-2]
        filename_a = (
            dir
            + (
                "/"
                if (dir[-1] not in ["/", "\\"]) or (flnm[0] not in ["/", "\\"])
                else 0
            )
            + flnm
            + "_plot_{1}.jpg"
        )
        filename_b = (
            dir
            + (
                "/"
                if (dir[-1] not in ["/", "\\"]) or (flnm[0] not in ["/", "\\"])
                else 0
            )
            + flnm
            + "_hist_{1}.jpg"
        )
        filename = (
            dir2
            + (
                "/"
                if (dir2[-1] not in ["/", "\\"]) or (flnm[0] not in ["/", "\\"])
                else 0
            )
            + flnm
            + "_hist_{1}.p"
        )
    # print(filename_a,filename_b,filename)
    return filename, filename_a, filename_b

def clean_nans(
    xdata: np.ndarray,
    ydata: np.ndarray,
    weights=None,
):
    """
    Function that returns a cleaned version of x and y arrays from "np.nan" values.

    Args:
        xdata   (np.ndarray): x data.
        ydata   (np.ndarray): y data.
        weights (np.ndarray): weights of y data.
    Return:
        xdata_cleaned (np.ndarray): cleaned x data
        ydata_cleaned (np.ndarray): cleaned y data
        wdata_cleaned (np.ndarray): cleaned weights
    """
    assert xdata.shape == ydata.shape
    num_elements = np.zeros(xdata.shape)
    if type(weights) not in [str, type(None)]:
        num_elements = np.logical_not(
            (np.isnan(xdata))
            | (np.isinf(xdata))
            | (np.isinf(ydata))
            | (np.isnan(ydata))
            | (ydata < -100)
            | (np.isnan(weights))
            | (weights == 0)
        )
    else:
        num_elements = np.logical_not(
            (np.isnan(xdata))
            | (np.isinf(xdata))
            | (np.isinf(ydata))
            | (np.isnan(ydata))
            | (ydata < -100)
        )

    clean_x = xdata[num_elements]
    clean_y = ydata[num_elements]
    if type(weights) not in [str, type(None)]:
        weights = np.array(weights)
        assert xdata.shape == weights.shape
        # sigma = np.sum(weights[num_elements])/weights[num_elements]
        sigma = weights[num_elements]
        # if sigma[np.where(clean_y == np.max(clean_y))] < sigma[np.where(clean_x == np.min(clean_x))]:
        #     print("We found that the weights injected aren't decreasing with Intensity\n if you want to continue supress this message by deleting it from:\n SlimPy.clean_nans")
    elif type(weights) == str:
        if weights == "1/sqrtI":
            weights = 1.0 / np.sqrt(clean_y.copy())
        elif weights == "I":
            weights = clean_y.copy()
        elif weights == "expI":
            weights = clean_y.copy() ** 2
        elif weights == "I2":
            weights = np.exp(clean_y.copy())
        elif weights == "sqrtI":
            weights = np.sqrt(clean_y.copy())
        else:
            raise ValueError(
                "the weights are unknown make sure you give the right ones\n current value: {} {} \n the allowed ones are: I, expI, I2, sqrtI".format(
                    type(weights), weights
                )
            )

        try:
            weights2 = weights - np.nanmin(weights)
            weights = weights2
        except:
            pass
        sigma = 1 / (weights.copy() / np.sum(weights))
    elif type(weights) == type(None):
        sigma = 1 / (np.ones(len(clean_y)) / len(clean_y))
    return clean_x, clean_y, sigma

@jit(nopython=True)
def fst_neigbors(
    extent: float,
    pixel_size_lon: float = 1,
    pixel_size_lat: float = 1,
    verbose: float = 0,
):
    """Generates a list of first neiboors in a square lattice and returns inside the list
        [n,m,n**2+m**2]

    Args:
        extent (float): how far the pixels will extend
    Return:
        nm_list (np.ndarray): list of data [n,m,n**2+m**2]
    """
    print(pixel_size_lon, pixel_size_lat)
    a = min(pixel_size_lon, pixel_size_lat) / pixel_size_lon
    b = min(pixel_size_lon, pixel_size_lat) / pixel_size_lat
    if verbose >= 1:
        print("a=", a, "b=", b)
    nm_list = []
    extent_2 = extent**2
    for n in range(-extent, extent + 1):
        for m in range(-extent, extent + 1):
            s = a**2 * n**2 + b**2 * m**2
            if s <= extent_2:
                nm_list.append([n, m, s])
    return nm_list

@jit(nopython=True)
def join_px(data, i, j, ijc_list):
    res_px = float(0.0)
    s = float(0.0)

    for n_layer in ijc_list:
        i2, j2, c = n_layer

        if (
            data.shape[0] - (i + i2) > 0
            and data.shape[1] - (j + j2) > 0
            and i + i2 >= 0
            and j + j2 >= 0
        ):
            if not np.isnan(data[i + i2, j + j2]):
                res_px += float(c * data[i + i2, j + j2])
                s += float(c)
    if s != 0:
        return res_px / s
    else:
        return np.nan

def _cv2blur(data, size):
    try:
        len(size)
    except:
        size = [size, size]

    blured = np.empty_like(
        data,
    )
    for i in range(data.shape[1]):
        blured[0, i] = cv2.blur(data[0, i] * 1, size, borderType=cv2.BORDER_REFLECT_101)
    return blured

def get_specaxis(hdu: PrimaryHDU or ImageHDU) -> np.ndarray:
    """
    Get the spectral axis values from an HDU header using the WCS information.

    Args:
        hdu (PrimaryHDU or ImageHDU): An HDU (Header Data Unit) object of type PrimaryHDU or ImageHDU.

    Returns:
        np.ndarray: Array of spectral axis values in angstroms.
    """
    spec_pix = np.arange(hdu.data.shape[1])
    wcs = WCS(hdu.header)
    _, _, specaxis, _ = wcs.wcs_pix2world(0, 0, spec_pix, 0, 0)
    specaxis *= 10**10
    return specaxis

def _sciunif(data, size):
    if not isinstance(size, Iterable):
        # len(size)
        size = [1,1,size, size]
    else:
        pass
    
    blured = uniform_filter(data, size, mode="reflect")
    # print(size, blured.shape)
    return blured

# def deNaN(data):
#     clean_data = data.copy()

#     for i in range(clean_data.shape[1]):
#         xNaN = np.nanmean(data[:, i, :, :], axis=(0, 1))
#         yNaN = np.nanmean(data[:, i, :, :], axis=(0, 2))
#         if len(np.where(np.logical_not(np.isnan(xNaN)))[0]) == 0:
#             max_lon = 0
#             min_lon = 0
#         else:
#             max_lon = np.max(np.where(np.logical_not(np.isnan(xNaN)))) + 1
#             min_lon = np.min(np.where(np.logical_not(np.isnan(xNaN))))
#         if len(np.where(np.logical_not(np.isnan(yNaN)))[0]) == 0:
#             max_lat = 0
#             min_lat = 0
#         else:
#             min_lat = np.min(np.where(np.logical_not(np.isnan(yNaN))))
#             max_lat = np.max(np.where(np.logical_not(np.isnan(yNaN)))) + 1

#         clean_data[:, i, 0:min_lat, :] = 0
#         clean_data[:, i, :, 0:min_lon] = 0
#         clean_data[:, i, max_lat:, :] = 0
#         clean_data[:, i, :, max_lon:] = 0

#     return clean_data

# def reNaN(original_data, clean_data, size):
#     data = original_data.copy()
#     reclean_data = clean_data.copy()
#     for i in range(clean_data.shape[1]):
#         xNaN = np.nanmean(data[:, i, :, :], axis=(0, 1))
#         yNaN = np.nanmean(data[:, i, :, :], axis=(0, 2))

#         if len(np.where(np.logical_not(np.isnan(xNaN)))[0]) == 0:
#             max_lon = 0
#             min_lon = 0
#         else:
#             max_lon = np.max(np.where(np.logical_not(np.isnan(xNaN)))) + 1
#             min_lon = np.min(np.where(np.logical_not(np.isnan(xNaN))))
#         if len(np.where(np.logical_not(np.isnan(yNaN)))[0]) == 0:
#             max_lat = 0
#             min_lat = 0
#         else:
#             min_lat = np.min(np.where(np.logical_not(np.isnan(yNaN))))
#             max_lat = np.max(np.where(np.logical_not(np.isnan(yNaN)))) + 1
        
        
#         min_lat = min_lat + (size[1] // 2 + (1 if size[1] % 2 != 0 else 0))
#         min_lon = min_lon + (size[2] // 2 + (1 if size[2] % 2 != 0 else 0))
#         max_lat = max_lat - (size[1] // 2 + (1 if size[1] % 2 != 0 else 0))
#         max_lon = max_lon - (size[2] // 2 + (1 if size[2] % 2 != 0 else 0))
#         min_tim = 0 + (size[0] // 2 + (1 if size[0] % 2 != 0 else 0))
#         max_tim = data.shape[0] - (size[0] // 2 + (1 if size[0] % 2 != 0 else 0))


#         min_lat = np.min([min_lat, data.shape[3]+1])
#         min_lon = np.min([min_lon, data.shape[2]+1])
#         max_lat = np.max([max_lat, 0])
#         max_lon = np.max([max_lon, 0])
#         max_tim = np.max([max_tim, 0])
#         min_tim = np.min([min_tim, data.shape[0]-1])
        
        
#         reclean_data[:            , i,  : min_lat - 1, :            ] = np.nan
#         reclean_data[:            , i, :             , : min_lon - 1] = np.nan
#         reclean_data[:            , i, max_lat + 1 : , :            ] = np.nan
#         reclean_data[:            , i, :             , max_lon + 1 :] = np.nan
#         reclean_data[: min_tim - 1, i, :             , :            ] = np.nan
#         reclean_data[max_tim + 1 :, i, :             , :            ] = np.nan
        

#     return reclean_data

def boundary_size(size,data_shape):
  boundary = np.zeros((len(data_shape),2),dtype=int)
  for ind in range(len(size)):
    si = size[ind]
    sh = data_shape[ind]
    if si %2 == 0:
      boundary[ind] = [si//2,sh-si//2+1]
    else:
      boundary[ind] = [(si-1)//2,sh-(si-1)//2]
  return boundary

def reNaN(raw_data,conv_data,mask_data,size):
    boundaries = boundary_size(size,raw_data.shape)
    naned_data = np.empty_like(conv_data,dtype=float)*np.nan
    # first we remove all the reflected data in the boudaries
    slices = [slice(boundaries[ind][0],boundaries[ind][1]) for ind in range(len(size))]
    naned_data[*slices] = conv_data[*slices]
    # then we remove all the previous nan values
    naned_data[mask_data<1] = np.nan
    return naned_data



@jit(nopython=True)
def join_dt(data, ijc_list):
    # data_new = np.zeros_like(data,dtype=float) #numba has no zeros_like
    data_new = data.copy() * np.nan
    for k in range(data.shape[0]):
        for l in range(data.shape[1]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    data_new[k, l, i, j] = join_px(data[k, l], i, j, ijc_list)
    return data_new


def convolve(
    window,
    mode,
    lon_pixel_size,
    lat_pixel_size,
    convolution_extent_list,
    convolution_function,
    
    verbose=0,
):
    if verbose >= 1:
        print(f"convolving using {mode}")
    if mode == "cercle":
        conv_data = np.zeros((*convolution_extent_list.shape, *window.shape))
        if verbose >= 2:
            print("creating convolution list...")
        for i in range(convolution_extent_list.shape[0]):
            if convolution_extent_list[i] == 0:
                conv_data[i] = window.copy()
                continue
            else:
                ijc_list = np.array(
                    fst_neigbors(
                        convolution_extent_list[i],
                        lon_pixel_size,
                        lat_pixel_size,
                        verbose=verbose,
                    )
                ).astype(int)
                # print(ijc_list)
                # sleep(5)
                ijc_list[:, 2] = convolution_function(ijc_list)
                conv_data[i] = join_dt(window, ijc_list)
    elif mode == "box":
        conv_data = np.zeros((*convolution_extent_list.shape, *window.shape))
        clean_window = deNaN(window)
        for i in range(convolution_extent_list.shape[0]):
            if lat_pixel_size < lon_pixel_size:
                size = np.array(
                    [
                        1,
                        1 + (convolution_extent_list[i]),
                        1
                        + (convolution_extent_list[i])
                        * lat_pixel_size
                        / lon_pixel_size,
                    ],
                    dtype=int,
                )
            else:
                size = np.array(
                    [
                        1,
                        1
                        + (convolution_extent_list[i])
                        * lon_pixel_size
                        / lat_pixel_size,
                        1 + (convolution_extent_list[i]),
                    ],
                    dtype=int,
                )
            # print("box size:",size)
            if verbose >= 2:
                print("creating convolution list...")
            for i in range(convolution_extent_list.shape[0]):
                if convolution_extent_list[i] == 0:
                    conv_data[i] = window.copy()
                    continue
                else:
                    conv_data[i] = reNaN(window, _sciunif(clean_window, size), size[1:])
    else:
        raise ValueError(f"mode:{mode} is not implemented or there is a misspelling")
    return conv_data

def convolve_4D(
    window,
    mode,
    convolution_extent_list,
    verbose=0,
):
    convolution_extent_list = np.array(convolution_extent_list)
    if verbose >= 1:
        print(f"convolving using {mode}")
    if mode == "cercle":
        raise ValueError("cercle mode is deprecated")
    elif mode == "box":
        conv_data = np.zeros((convolution_extent_list.shape[0], *window.shape))
        # clean_window = deNaN(window)
        clean_window = window.copy()
        clean_window[np.isnan(clean_window)] = 0
        data_mask = np.ones_like(window)
        data_mask[np.isnan(window)] = 0
        for i,size in enumerate(convolution_extent_list):
            for j in size: 
                if size[i]>window.shape[i]:
                    colored_text(f'Warning:One of the dimentions of convolution kernel size {size} is larger than that of the window size {window.shape}','yellow')
                    break
            if verbose >= 2:
                print("creating convolution list...")
                print(f"convolving with size {size}")
            if (np.array(size)==1).all():
                conv_data[i] = window.copy()
                continue
            else:
                blured = _sciunif(clean_window[:,:,:,:], size)
                blured_mask = _sciunif(data_mask[:,:,:,:], size)
                # print(blured.shape,conv_data[j].shape)
                # conv_data[i] = reNaN(window, blured, [size[0],size[2],size[3]])
                conv_data[i] = reNaN(window, blured,blured_mask, size)
                
    else:
        raise ValueError(f"mode:{mode} is not implemented or there is a misspelling")
    return conv_data

# @jit(nopython=True) #not tryed yet
def Preclean(cube):
    cube2 = cube.copy()
    # logic=np.logical_or(np.isinf(cube2),cube2<-10**10)
    logic = np.logical_or(cube2 > 490, cube2 < -10)
    cube2[logic] = np.nan
    if (
        False
    ):  # this part is for elemenating the cosmic effected values but it's not well done (see it again)
        mean_cube = np.nanmean(cube2, axis=1) * 1000
        for i in range(cube2.shape[1]):
            cube2[:, i, :, :][cube2[:, i, :, :] > mean_cube] = np.nan

    return cube2


def round_up(n, decimals=0):
    multiplier = 10**decimals
    return math.ceil(n * multiplier) / multiplier


def gen_shmm(create=False, name=None, ndarray=None, size=0, shape=None, dtype=float):
    assert (type(ndarray) != type(None) or size != 0) or type(name) != type(None)
    assert type(ndarray) != type(None) or type(shape) != type(None)
    if ndarray is not None:
        dtype = ndarray.dtype
    size = size if type(ndarray) == type(None) else ndarray.nbytes
    shmm = SharedMemory(create=create, size=size, name=name)
    shmm_data = np.ndarray(
        shape=shape if type(ndarray) == type(None) else ndarray.shape,
        buffer=shmm.buf,
        dtype=dtype,
    )

    if create and type(ndarray) != type(None):
        shmm_data[:] = ndarray[:]
    elif create:
        shmm_data[:] = np.nan

    return shmm, shmm_data


def verbose_description(verbose):
    print(f"level {verbose:01d} verbosity")
    raise Exception
    if verbose == -2:
        print(
            "On-screen information mode: Dead \nNo information including warnings  (CAREFUL DUDE!!)"
        )
    elif verbose == -1:
        print(
            "On-screen information mode: Minimal\nHighly important ones only and wornings (Don't have a blind faith please)"
        )

    elif verbose == 0:
        print("On-screen information mode: Normie\nBasic information any normie needs")

    elif verbose == 1:
        print(
            "On-screen information mode: Extra\nmore detailed information for tracking and debugging"
        )
    elif verbose == 2:
        print(
            "On-screen information mode: Stupid\nUnless you are as stupid as the writer of this script, you don't need this much information for debugging an error"
        )
    elif verbose == 3:
        print(
            "On-screen information mode: Visual\nPlot extra figures in a ./tmp file with "
        )


def gen_velocity(
    doppler_data,
    quite_sun=[60, 150, 550, 600],
    correction=False,
    verbose=0,
    get_0lbdd=False,
):
    qs = quite_sun
    mean_doppler = np.nanmean(doppler_data[qs[2] : qs[3], qs[0] : qs[1]])
    # print("mean_doppler",mean_doppler)
    results = (doppler_data - mean_doppler) / mean_doppler * 3 * 10**5
    if correction:
        if verbose > 0:
            print("Correcting")
        hist, bins = gen_velocity_hist(
            results, bins=np.arange(-600, 600, 1), verbose=verbose
        )
        vel_corr, ref = correct_velocity(hist, bins, verbose=verbose)
        if verbose > 0:
            print(f"The correction found the distribution was off by {ref}")
        results -= ref

    if verbose > 1:
        fig = plt.figure()
        plt.pcolormesh(results, cmap="twilight_shifted", vmax=80, vmin=-80)
        plt.plot(
            [qs[1], qs[0], qs[0], qs[1], qs[1]],
            [qs[2], qs[2], qs[3], qs[3], qs[2]],
            color="green",
            label="mean value {:06.1f}".format(mean_doppler),
        )
        plt.legend()
        # plt.savefig('fig_test.jpg')
    return (
        results,
        (None if not correction else ref),
        (None if not get_0lbdd else mean_doppler),
    )


def gen_velocity_hist(velocity_data, axis=None, bins=None, verbose=0):
    if type(bins) == type(None):
        bins = np.linspace(np.nanmin(velocity_data), np.nanmax(velocity_data), num=200)
    hist, bins = np.histogram(velocity_data, bins=bins)
    bins = (bins[:-1] + bins[1:]) / 2
    if verbose > 1:
        if type(axis) == type(None):
            fig, axis = plt.subplots(1, 1)
        axis.step(bins, hist)
        axis.set_yscale("log", base=10)
        plt.axvline(0, ls="--", color="red", alpha=0.5)
    return hist, bins


def correct_velocity(velocity_hist, velocity_values, verbose=0):
    if verbose > 1:
        print(
            "correct_velocity<func>.velocity_hist.shape: {}\n,correct_velocity<func>.velocity_values.shape: {}".format(
                velocity_hist.shape, velocity_values.shape
            )
        )
    ref_velocity = velocity_values[
        np.where(velocity_hist == np.nanmax(velocity_hist))[0]
    ]
    ref_velocity = np.mean(ref_velocity)
    if verbose > 0:
        print(
            f"the velocity reference was found at {ref_velocity}\n now it will be set to 0"
        )
    velocity_values_corr = velocity_values - ref_velocity
    return velocity_values_corr, ref_velocity


def get_all_celestials(raster,**kwargs):
    if isinstance(raster, WCS):
        shape = np.array(kwargs["shape"])
        if len(shape) == 3:pass
        elif len(shape) == 4:
            shape[1] = 1
        else:
            raise ValueError("The shape of the raster is not acceptable")
        wcs = raster
        
        t = np.arange(shape[0], dtype=int)
        lbd= np.array([0], dtype=int)
        y = np.arange(shape[2], dtype=int)
        x = np.arange(shape[3], dtype=int)
        # Generate the meshgrid
        tlbdxy = np.array(np.meshgrid(x,y,lbd,t , indexing='ij'))  
        res = wcs.pixel_to_world(tlbdxy[0], tlbdxy[1], tlbdxy[2],tlbdxy[3])
        coords = res[0].reshape(shape[::-1], )
        time = res[2].reshape(shape[::-1],)
        datetime64_array = time.to_value('datetime64')
    elif type(raster) == astropy.io.fits.hdu.hdulist.HDUList or isinstance(
        raster, Iterable
    ):

        shape = raster[0].data.shape
        wcs = WCS(raster[0].header)
        return (get_all_celestials(wcs,shape= shape))
        # specoords = res[1].reshape(shape[::-1],)
        
    else:
        print(
            f"The raster passed doesn't match any known types: {type(raster)} but it has to be one of these types: \n{ndcube.ndcollection.NDCollection}\n{astropy.io.fits.hdu.hdulist.HDUList}"
        )
        print(raster)
        print("--------------------------------")
        print(type(raster), isinstance(raster, Iterable))
        raise ValueError("inacceptable type")
    
    return coords,datetime64_array
    

def get_celestial(raster, include_time=False, **kwargs):
    lonlat,time = get_all_celestials(raster,**kwargs)
    if lonlat.shape[3] == 1:# dealing wwith a raster
        lon = lonlat.spherical.lon.arcsec[:,:,0,0]
        lat = lonlat.spherical.lat.arcsec[:,:,0,0]
        time = time[:,:,0,0]
    else:# dealing wwith a time series
        lon = lonlat.spherical.lon.arcsec[0,:,0,:]
        lat = lonlat.spherical.lat.arcsec[0,:,0,:]
        time = time[0,:,0,:]
    return (lon.T, lat.T, time.T) if include_time else (lon.T, lat.T)


def quickview(
    RasterOrPath,
    fig1=None,
    imag_ax=None,
    fig2=None,
    spec_ax=None,
    remove_dumbles = slice(None,None,None),
):
    from pathlib import PosixPath, WindowsPath, Path

    if type(RasterOrPath) in (str, PosixPath, WindowsPath):
        raster = fits_reader.open(RasterOrPath)
    else:
        raster = RasterOrPath
    
    raster = get_data_raster(raster)
    
    lon, lat,time = get_celestial(raster,include_time=True)
    n = 3
    m = len(raster) // 3 + (1 if len(raster) % 3 != 0 else 0)

    if type(imag_ax) == type(None):
        fig1, ax1 = plt.subplots(m, n, figsize=(n * 3, m * 3), sharex=True, sharey=True)
        ax1 = ax1.flatten()
        #reduce inter space in h and w 
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1)
        for ax in range(len(raster), len(ax1)):
            ax1[ax].remove()
    else:
        ax1 = imag_ax
        fig1 = fig1 if type(fig1) != type(None) else imag_ax[0].figure
    if type(spec_ax) == type(None):
        fig2, ax2 = plt.subplots(m, n, figsize=(n * 3, m * 3))
        ax2 = ax2.flatten()
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1, right=0.9, top=0.9, bottom=0.1)
        for ax in range(len(raster), len(ax2)):
            ax2[ax].remove()
    else:
        ax2 = spec_ax
        fig2 = fig2 if type(fig2) != type(None) else spec_ax[0].figure
    fig1.suptitle(raster[0].header["DATE-OBS"])
    fig2.suptitle(raster[0].header["DATE-OBS"])
    
    #Dealing with a raster
    for i in range(len(raster)):
        data = raster[i].data
        if raster[0].data.shape[0]==1:
            image = np.nanmean(data, axis=(0, 1))
        else:
            image = np.nanmean(data, axis=(1, 3))
        spect = np.nanmean(data, axis=(0, 2, 3))
        spec_ax = get_specaxis(raster[i])
        kw = raster[i].header["EXTNAME"]
        
        if raster[0].data.shape[0]==1:    
            norm = normit(image[200:700])
            ax1[i].pcolormesh(lon[remove_dumbles,:], lat[remove_dumbles,:], image[remove_dumbles,:], norm=norm, cmap="magma")
        else:
            norm = normit(image.T[200:700])
            ax1[i].pcolormesh(time[:,remove_dumbles],lat[:,remove_dumbles],image[:,remove_dumbles], norm=norm, cmap="magma")
            # make the locators for the time axis of order of 1 hour 
            # ax1[i].xaxis.set_major_locator(plt.MaxNLocator(3))
            #Hour locator 
            from matplotlib.dates import HourLocator
            ax1[i].xaxis.set_major_locator(HourLocator())
            # change the date format to be dayThour:min 
            ax1[i].xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
        
        ax2[i].step(spec_ax, spect)
        ax1[i].set_title(kw)
        ax2[i].set_title(kw)
    
            
    return ((fig1, ax1), (fig2, ax2))


def getfiles(
    YEAR="all",
    MONTH="all",
    DAY="all",
    STD_TYP="all",
    STP_NUM="all",
    SOOP_NAM="all",
    MISOSTUD_NUM="all",
    in_name=None,
    verbose=0,
    L2_folder: str = "/archive/SOLAR-ORBITER/SPICE/fits/level2/",
):
    """
    Summary
        Find all the fits in the archive by YEAR/MONTH/DAY/STUDY_TYPE
    Args:
        YEAR       [list or float or int or str("ALL")] : the selected year(s)
        MONTH      [list or float or int or str("ALL")] : the selected month(s)
        DAY        [list or float or int or str("ALL")] : the selected day(s)
        STUDY_TYPE [str(SIT) or str(COMPO) or str ("DYN") or str("ALL")] : the selected study type(s)
    """
    path_l2 = Path(L2_folder)
    selected_fits = []
    searching_paths = []
    if True:  # reading years
        if type(YEAR) in [int, float]:
            years = [YEAR]
        elif type(YEAR) in (list, np.ndarray):
            years = np.array(YEAR)
        elif YEAR in ["ALL", "all"]:
            years = np.array([i for i in range(2018, 2030)])
        else:
            raise ValueError(
                'YEAR should be an integer, a list, or a string of value "all" not {}'.format(
                    type(YEAR)
                )
            )
    if True:  # reading months
        if type(MONTH) in [int, float]:
            months = [MONTH]
        elif type(MONTH) in (list, np.ndarray):
            months = np.array(MONTH)
        elif MONTH in ["ALL", "all"]:
            months = np.array([i for i in range(1, 13)])
        else:
            raise ValueError(
                'MONTH shuld be an integer, a list, or a string of value "all" not {}'.format(
                    type(MONTH)
                )
            )
    if True:  # reading days
        if type(DAY) in [int, float]:
            days = [DAY]
        elif type(DAY) in (list, np.ndarray):
            days = np.array(DAY)
        elif DAY in ["ALL", "all"]:
            days = np.array([i for i in range(1, 32)])
        else:
            raise ValueError(
                'DAY shuld be an integer, a list, or a string of value "all" not {}'.format(
                    type(DAY)
                )
            )
    for day in days:  # combining the path to the targeted folders
        for month in months:
            for year in years:
                searching_paths.append(
                    path_l2 / f"{year}" / f"{month:02d}" / f"{day:02d}"
                )
    if verbose >= 2:  # for seeing what are the paths chosen
        for i in searching_paths:
            print(i)
    for (
        path
    ) in (
        searching_paths
    ):  # combining the path to the targeted set and filtring study type
        if Path.exists(path):
            available_fits = os.listdir(path)
            # print(path,available_fits)
            for fits in available_fits:
                if str(fits)[-5:] == ".fits":
                    _sample = path / fits
                    # print("in_name",in_name)
                    if in_name is not None:
                        if type(in_name) == str:
                            in_name = [in_name]
                        for name in in_name:
                            if name in str(fits):
                                selected_fits.append(_sample)
                    else:
                        # print('this is the data',_sample)
                        data = fits_reader.open(_sample)
                        # PURPOSE = data[0].header['PURPOSE']
                        STUDY = data[0].header["STUDY"].upper()
                        STP = data[0].header["STP"]
                        SOOPNAME = data[0].header["SOOPNAME"].upper()
                        MISOSTUD = data[0].header["MISOSTUD"]
                        # print(STUDY   )
                        # print(STP     )
                        # print(SOOPNAME)
                        # print(MISOSTUD)
                        if True:
                            GOOD_STP = False
                            if STP_NUM in ["ALL", "all"]:
                                GOOD_STP = True
                            elif int(STP) == int(STP_NUM):
                                GOOD_STP = True
                        if True:
                            GOOD_MISOSTUD = False
                            if MISOSTUD_NUM in ["ALL", "all"]:
                                GOOD_MISOSTUD = True
                            elif int(MISOSTUD) == int(MISOSTUD_NUM):
                                GOOD_MISOSTUD = True

                        if (
                            ((STD_TYP in ["ALL", "all"]) or (STD_TYP in STUDY))
                            and GOOD_STP
                            and ((SOOP_NAM in ["ALL", "all"]) or (SOOP_NAM in SOOPNAME))
                        ):

                            selected_fits.append(_sample)
    if verbose >= 1:
        for i in selected_fits:
            print(i, "***********")
    return selected_fits


def get_input_template(where="./input_config_template.json",overwrite=False):
    PATH = pkg_resources.resource_filename(
        "saffron", "manager/input_config_template.json"
    )
    if Path(where).exists() and not overwrite:
        raise FileExistsError(f"{where} already exists, set overwrite to True to overwrite it")    
    else:
        shutil.copy(PATH, where)


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def normit(
    data=None,
    interval: interval = AsymmetricPercentileInterval(1, 99),
    stretch: stretch = SqrtStretch(),
    vmin: float = None,
    vmax: float = None,
    clip: bool = False,
    invalid=-1.0,
) -> ImageNormalize:
    """Normalize the data using the specified interval, stretch, vmin, and vmax.

    Args:
        data (numpy.ndarray): The data to be normalized.
        interval (astropy.visualization.Interval, optional): The interval to use for normalization.
            Defaults to AsymmetricPercentileInterval(1, 99).
        stretch (astropy.visualization.Stretch, optional): The stretch to apply to the data.
            Defaults to SqrtStretch().
        vmin (float, optional): The minimum value for normalization. Defaults to None.
        vmax (float, optional): The maximum value for normalization. Defaults to None.

    Returns:
        astropy.visualization.ImageNormalize: The normalized data.
    """
    if vmin is not None or vmax is not None:
        interval = None
    if stretch is not None:
        if np.all(np.isnan(data)):
            return None
        return ImageNormalize(
            data,
            interval,
            stretch=stretch,
            vmin=vmin,
            vmax=vmax,
            clip=clip,
            invalid=invalid,
        )

    return ImageNormalize(
        data, interval, vmin=vmin, vmax=vmax, clip=clip, invalid=invalid
    )


# def default_convolution_function(lst):
    """
    Default convolution function that creates an array of ones matching 
    the shape of the input's third column.

    Parameters:
    ----------
    lst : ndarray
        The input data array.

    Returns:
    -------
    ndarray
        An array of ones with the same length as the input's third column.
    """
    return np.zeros_like(lst[:, 2]) + 1