from pathlib import Path 
import os
import copy
from astropy.io.fits.hdu.hdulist import HDUList
import pathlib
import numpy as np 
from astropy.io import fits
from pathlib import Path
import matplotlib.pyplot as plt
import os
from ..utils import normit,suppress_output
from collections.abc import Iterable
from astropy.visualization import SqrtStretch,PowerStretch,LogStretch, AsymmetricPercentileInterval, ImageNormalize, MinMaxInterval, interval,stretch
from sunpy.map import Map
from fiplcr import Line
from fiplcr import LinearComb as lc
from fiplcr import fip_map
import astropy.units as u
import sunpy

#TODO MOVE somewhere else
def get_coord_mat(map,as_skycoord = False):
    res = sunpy.map.maputils.all_coordinates_from_map(map)
    if as_skycoord: return res
    try:
        lon = res.spherical.lon.arcsec
        lat = res.spherical.lat.arcsec
    except AttributeError:    
        lon = res.lon.value
        lat = res.lat.value 
    return lon,lat
def FIP_error(ll,Errors,Datas):
    S_delHF = 0
    S_delLF = 0
    
    # S_LF = ll.lc_LF.value
    # S_HF = ll.lc_HF.value
    S_LF   = 0
    S_HF   = 0 
        
    for i in range(len(ll.lines)):
        
        if i < ll.xLF.shape[1]: 
            S_delLF += (
                ll.ionsLF[i                ].coeff_map / ll.ionsLF[i                ].ph_abund * 
            Errors[i])
            S_LF    += (
                ll.ionsLF[i                ].coeff_map / ll.ionsLF[i                ].ph_abund * 
            Datas[i])
            # plt.figure()
            # plt.pcolormesh(S_delLF,vmin=0,vmax=150);plt.colorbar()
            # plt.title("S_delLF")
        else: 
            S_delHF += (
                ll.ionsHF[i-ll.xLF.shape[1]].coeff_map / ll.ionsHF[i-ll.xLF.shape[1]].ph_abund * 
            Errors[i]
            )
            S_HF    += (
                ll.ionsHF[i-ll.xLF.shape[1]].coeff_map / ll.ionsHF[i-ll.xLF.shape[1]].ph_abund * 
            Datas[i]
            )
            
    FIP_error = S_delHF/S_HF + S_delLF/S_LF 
    if False:
        if False:
            norm = ImageNormalize(S_delHF,interval=AsymmetricPercentileInterval(1,99),stretch=SqrtStretch())
            plt.figure()
            plt.pcolormesh(S_delHF,norm=norm,cmap="magma",
            # vmin=0,vmax=150
            );plt.colorbar()
            plt.title("S_delHF")
                    
            plt.figure()
            norm = ImageNormalize(S_delLF,interval=AsymmetricPercentileInterval(1,99),stretch=SqrtStretch())
            plt.pcolormesh(S_delLF,norm=norm,cmap='magma',
            # vmin=0,vmax=150
            );plt.colorbar()
            plt.title("S_delLF")
            
            plt.figure()
            norm = ImageNormalize(S_HF,interval=AsymmetricPercentileInterval(1,99),stretch=SqrtStretch())
            plt.pcolormesh(S_HF,norm=norm,cmap='magma',
            # vmin=0,vmax=150
            );plt.colorbar()
            plt.title("S_HF")
            
            plt.figure()
            norm = ImageNormalize(S_LF,interval=AsymmetricPercentileInterval(1,99),stretch=SqrtStretch())
            plt.pcolormesh(S_LF,norm=norm,cmap='magma',
            # vmin=0,vmax=150
            );plt.colorbar()
            plt.title("S_LF")
            
            plt.figure()
            norm = ImageNormalize(S_delLF/S_LF,interval=AsymmetricPercentileInterval(1,99),stretch=SqrtStretch())
            plt.pcolormesh(S_delLF/S_LF,norm=norm,cmap='magma',
            # vmin=0,vmax=150
            );plt.colorbar()
            plt.title("S_delLF/S_LF")
            
            plt.figure()
            norm = ImageNormalize(S_delHF/S_HF,interval=AsymmetricPercentileInterval(1,99),stretch=SqrtStretch())
            plt.pcolormesh(S_delHF/S_HF,norm=norm,cmap='magma',
            # vmin=0,vmax=150
            );plt.colorbar()
            plt.title("S_delHF/S_HF")
            
        plt.figure()
        norm = ImageNormalize(FIP_error,interval=AsymmetricPercentileInterval(1,99),stretch=SqrtStretch())
        plt.pcolormesh(FIP_error,norm=norm,cmap='magma',
        # vmin=0,vmax=1
        );plt.colorbar()
        plt.title("FIP_error")
        
    return FIP_error

  
def filePath_manager(data_dir):
  files = os.listdir(data_dir)
  files.sort()
  file_cluster = []
  IDset = list(set([file[42:51] for file in files]))
  IDset.sort()
  for ind_ID,ID in enumerate(IDset):
    file_cluster.append([])
    filesByID = [file for file in files if file[42:51] in ID]
    ionIDset = set([file[73:-9] for file in filesByID if 'B' not in file[73:-9]])
    ionIDset =list(ionIDset)
    ionIDset.sort()
    for ionID in ionIDset:
      filesByIonID = [data_dir/file for file in filesByID if file[73:-9] in ionID]
    
      file_cluster[ind_ID].append([])
      file_cluster[ind_ID][-1] = filesByIonID
  return(file_cluster)
class SPECLine():
  def __init__(self,hdul_or_path):
    self.hdul = {'int':None,'wid':None,'wav':None}
    self.int  = None
    self.wid  = None
    self.wav  = None
    self.rad  = None
    self.int_err = None
    self.wid_err = None
    self.wav_err = None
    self.rad_err = None
    self._prepare_data(hdul_or_path)
  
  def _prepare_data(self,hdul_or_path):
    self.charge_data(hdul_or_path)
    self.compute_params()
    
  def charge_data(self,hdul_or_path):
    if isinstance(hdul_or_path,(str, pathlib.PosixPath, WindowsPath, pathlib.WindowsPath,HDUList)):raise ValueError('The hdul_or_path sould be a list of 3')
    for val in hdul_or_path:
      if isinstance(val, (str, pathlib.PosixPath, WindowsPath, pathlib.WindowsPath)):
        hdul = fits.open(val)
      elif isinstance(val,HDUList):
        hdul = val.copy()
      else:raise TypeError(str(val))
      
      if hdul[0].header["MEASRMNT"] == 'intensity':
        self.hdul["int"] = hdul
      if hdul[0].header["MEASRMNT"] == 'wavelength':
        self.hdul["wav"] = hdul
      if hdul[0].header["MEASRMNT"] == 'width':
        self.hdul["wid"] = hdul
    
  def compute_params(self,):
    if self.hdul is None: raise Exception('Call self.charge_data first')
    self.int     = self.hdul["int"][0].data 
    self.wid     = self.hdul["wid"][0].data 
    self.wav     = self.hdul["wav"][0].data 
    self.rad     = self.int * self.wid *np.sqrt(np.pi)
    self.int_err = self.hdul["int"][1].data
    self.wid_err = self.hdul["wid"][1].data
    self.wav_err = self.hdul["wav"][1].data
    self.rad_err = (self.int_err/self.int + self.wid_err/self.wid) * self.rad
  
  def plot(self,params='all',axes =None,add_keywords = False):
    """_summary_

      Args:
          params (str or list, optional): all,int,wid,wav,rad. Defaults to 'all'.
          axes (list, optional): axes length = params. Defaults to None.
      """
    if params == 'all': params = ['int','wav','wid','rad'] 
    if isinstance(params, str): params = [params] 
    if axes is None: fig,axes = plt.subplots(1,len(params),figsize=(len(params)*4,4))
    if not isinstance(axes,Iterable): axes = [axes]
    if len(params)!=1:
      for ind,param in enumerate(params): self.plot(params=param, axes = axes[ind])
    else:
      if params[0]   == 'int':
        data = self.int
        norm = normit(data)
      elif params[0] == 'wid':
        data = self.wid
        norm = normit(data,stretch=None)
      elif params[0] == 'wav':
        data = self.wav
        norm = normit(data,stretch=None)
      elif params[0] == 'rad':
        data = self.rad
        norm = normit(data)
      elif params[0] == 'int_err':
        data = self.int_err
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      elif params[0] == 'wav_err':
        data = self.wav_err
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      elif params[0] == 'wid_err':
        data = self.wid_err
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      elif params[0] == 'rad_err':
        data = self.rad_err
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      else: raise ValueError
      
      map = Map(self.hdul['int'][0].data,self.hdul['int'][0].header)
      lon,lat = get_coord_mat(map)
      im = axes[0].pcolormesh(lon,lat,data,norm=norm,zorder=-1,cmap="magma")
      # plt.colorbar(im,ax=axes[0])
      # axes[0].set_title(params[0])
      
      if add_keywords: pass
class CompoRaster():
  def __init__(self,list_paths):
    self.lines   = []
    self.ll      = None
    self.FIP_err = None
    self._prepare_data(list_paths)
  def _prepare_data(self,list_paths):
    for paths in list_paths:
      self.lines.append(SPECLine(paths))
    self.FIP_err = self.lines[0].hdul['int'][0].data * 0
    pass
    
  @property
  def FIP(self):
    try:
      res = self.ll.FIP_map.copy()
      if res is None: res = self.FIP_err.copy()+1
      return res 
    except:
      return self.FIP_err.copy()+1
    
  def gen_compo_LCR(self,HFLines=None,LFLines=None,ll=None,suppressOutput=True):
    All_lines= list(HFLines)
    All_lines.extend(LFLines)
    if ll is None:
      self.ll = lc([
        Line(ionid, wvl) for ionid, wvl in All_lines
        ],using_S_as_LF=True)

      if suppressOutput:
        with suppress_output():
          self.ll.compute_linear_combinations()
      else:self.ll.compute_linear_combinations()
    else:
      self.ll = copy.deepcopy(ll)
    logdens = 8.3# try  9.5
    idens = np.argmin(abs(self.ll.density_array.value - 10**logdens))
    density_map = 10**logdens*np.ones(self.lines[0].int.shape, dtype=float)*u.cm**-3
    
    wvls = np.empty(shape=(len(self.lines),))
    for ind,line in enumerate(self.lines):wvls[ind] = line.hdul['int'][0].header["WAVELENGTH"]
    
    data = []
    err  = []
    for ind,ionLF in  enumerate(self.ll.ionsLF):
      diff = np.abs(ionLF.wvl.value-wvls)
      argmin = np.argmin(diff)
      if diff[argmin]>0.1: raise Exception(f"Haven't found a for: {ionLF} {wvls}")
      self.ll.ionsLF[ind].int_map = self.lines[argmin].rad*u.W*u.m**-2/10
      data.append(self.lines[argmin].rad)
      err.append (self.lines[argmin].rad_err)
      # print(f"rad: {self.lines[argmin].hdul['int'][0].header['wavelength']},lc: {ionLF.wvl.value}")
    for ind,ionHF in  enumerate(self.ll.ionsHF):
      diff = np.abs(ionHF.wvl.value-wvls)
      argmin = np.argmin(diff)
      if diff[argmin]>0.1: raise Exception(f"Haven't found a for: {ionHF} {wvls}")
      self.ll.ionsHF[ind].int_map = self.lines[argmin].rad*u.W*u.m**-2/10
      data.append(self.lines[argmin].rad)
      err.append (self.lines[argmin].rad_err)
      # print(f"rad: {self.lines[argmin].hdul['int'][0].header['wavelength']},lc: {ionHF.wvl.value}")
      
    fip_map(self.ll, density_map)
    if True: #Complete calculation of error based on differentiation method
      self.FIP_err = FIP_error(
        self.ll,
        err,
        data,
        )/self.FIP
    else:
      S_ind = np.argmin(np.abs(wvls-750.22))
      self.FIP_err = self.lines[S_ind].rad_err/self.lines[S_ind].rad
  def find_line(self,wvl):
    wvls = np.empty(shape=(len(self.lines),))
    for ind,line in enumerate(self.lines):wvls[ind] = line.hdul['int'][0].header["WAVELENGTH"]
    diff = np.abs(wvls-wvl)
    ind = np.argmin(diff)
    if diff[ind]>1: raise Exception(f'big difference {wvl}\n{wvls}',)
    return self.lines[ind]
  def show_all_wvls(self):
    wvls = np.empty(shape=(len(self.lines),))
    for ind,line in enumerate(self.lines):wvls[ind] = line.hdul['int'][0].header["WAVELENGTH"]
    return(wvls)