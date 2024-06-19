from pathlib import Path 
import os
import copy
from astropy.io.fits.hdu.hdulist import HDUList
from astropy.wcs import WCS
import pathlib
import numpy as np 
from astropy.io import fits
from pathlib import Path,WindowsPath,PosixPath
import matplotlib.pyplot as plt
import os
from ..utils import normit,suppress_output,gen_axes_side2side,get_coord_mat
from collections.abc import Iterable
from astropy.visualization import SqrtStretch,PowerStretch,LogStretch, AsymmetricPercentileInterval, ImageNormalize, MinMaxInterval, interval,stretch
from sunpy.map import Map
from fiplcr import Line
from fiplcr import LinearComb as lc
from fiplcr import fip_map
import astropy.units as u
import sunpy

#TODO MOVE somewhere else
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

class SPECLine():
  def __init__(self,hdul_or_path):
    self._all = {"int"    :None,
                "wav"    :None,
                "wid"    :None,
                "rad"    :None,
                "int_err":None,
                "wav_err":None,
                "wid_err":None,
                "rad_err":None,
                }
    self._prepare_data(hdul_or_path)
  
  @property
  def wavelength(self):
    return self._all['int'].header["WAVELENGTH"]
  @property
  def observatory(self):
    return self._all['int'].header["OBSERVATORY"]
  @property
  def instrument(self):
    return self._all['int'].header["INSTRUMENT"]
  @property
  def ion(self):
    return self._all['int'].header["ION"]
  @property
  def line_id(self):
    return self._all['int'].header["LINE_ID"]
  @property 
  def headers(self):
    return {'int'    :self._all['int'    ].header,
            'wav'    :self._all['wav'    ].header,
            'wid'    :self._all['wid'    ].header,
            'int_err':self._all['int_err'].header,
            'wav_err':self._all['wav_err'].header,
            'wid_err':self._all['wid_err'].header,
            }
  def __getitem__(self,val: ["int","wav","wid","rad","int_err","wav_err","wid_err",'rad_err']):
    if isinstance(val,Iterable) and not isinstance(val,str):
      return [self.__getitem__(key) for key in val]
    else:
      if   val == "int"    :
        return self._all["int"    ].data
      elif val == "wav"    :
        return self._all["wav"    ].data
      elif val == "wid"    :
        return self._all["wid"    ].data
      elif val == "int_err":
        return self._all["int_err"].data
      elif val == "wav_err":
        return self._all["wav_err"].data
      elif val == "wid_err":
        return self._all["wid_err"].data
      elif val == "rad"    :
        return self._all["rad"    ]
      elif val == "rad_err":
        return self._all["rad_err"]
      else: raise Exception(f'{val} is not a valid keyword \nValid keywords: int,wav,wid,rad,int_err,wav_err,wid_err,rad_err')
  
  def _prepare_data(self,hdul_or_path):
    self.charge_data(hdul_or_path)
    self.compute_params()
    
  def charge_data(self,hdul_or_path):
    # if isinstance(hdul_or_path,(str, PosixPath, WindowsPath, pathlib.WindowsPath,HDUList)):raise ValueError('The hdul_or_path sould be a list of 3')
    if isinstance(hdul_or_path, (str, PosixPath, WindowsPath, pathlib.WindowsPath)):
      hdul = fits.open(hdul_or_path)
    elif isinstance(hdul_or_path,HDUList):
      hdul = hdul_or_path.copy()
    else:raise TypeError(str(hdul_or_path))
    
    for hdu in hdul:
      if hdu.header["MEASRMNT"]=="bg": raise Exception('The background is not needed in this Object')
      self._all[hdu.header["MEASRMNT"]] = hdu 
      
  def compute_params(self,):
    if any([self._all[key] is None for key in ["int","wav","wid"]]): raise Exception(f"Call self.charge_data first because there is no {[key for key in ['int','wav','wid'] if self._all[key] is None]}")
    self._all["rad"]     = self['int'] * self['wid'] *np.sqrt(np.pi)
    self._all["rad_err"] = (self["int_err"]/self['int'] + self['wid_err']/self['wid']) * self['rad']
  
  def plot(self,params='rad',axes =None,add_keywords = False):
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
        data = self["int"]
        norm = normit(data)
      elif params[0] == 'wid':
        data = self["wid"]
        norm = normit(data,stretch=None)
      elif params[0] == 'wav':
        data = self["wav"]
        norm = normit(data,stretch=None)
      elif params[0] == 'rad':
        data = self["rad"]
        norm = normit(data)
      elif params[0] == 'int_err':
        data = self["int_err"]
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      elif params[0] == 'wav_err':
        data = self["wav_err"]
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      elif params[0] == 'wid_err':
        data = self["wid_err"]
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      elif params[0] == 'rad_err':
        data = self["rad_err"]
        norm = normit(data,AsymmetricPercentileInterval(5,95))
      else: raise ValueError
      
      # map = Map(self['int'],self.headers['int'])
      lon,lat,time = get_celestial_L3(self)
      im = axes[0].pcolormesh(lon,lat,data,norm=norm,zorder=-1,cmap="magma")
      axes[0].set_title(self.line_id+'\n'+params[0])
      if add_keywords: pass
class SPICEL3Raster():
  def __init__(self,list_paths=None, folder_path = None):
    if (list_paths is None and folder_path is None) or (list_paths is not None and folder_path is not None)  : raise Exception("you need to specify strictly one of these arguments list_paths or folder_path")
    elif folder_path is not None: 
      list_paths = [str(file) for file in Path(folder_path).glob('*') ]
    else: 
      # nothing to do if the list is given
      pass
    self.lines   = []
    self.ll      = None
    self.FIP_err = None
    self._prepare_data(list_paths)
  def _prepare_data(self,list_paths):
    for paths in list_paths:
      try: 
        line = SPECLine(paths)
        self.lines.append(SPECLine(paths))
      except: 
        pass 
    self.FIP_err = self.lines[0]['int'] * np.nan
    pass
    
  @property
  def FIP(self):
    try:
      res = self.ll.FIP_map.copy()
      if res is None: res = self.FIP_err.copy()+1
      return res 
    except:
      return self.FIP_err.copy()+1
  def gen_compo_LCR(self,HFLines=None,LFLines=None,ll=None,suppressOutput=True,using_S_as_LF=True):
    All_lines= list(HFLines)
    All_lines.extend(LFLines)
    if ll is None:
      self.ll = lc([
        Line(ionid, wvl) for ionid, wvl in All_lines
        ],using_S_as_LF=using_S_as_LF)

      if suppressOutput:
        with suppress_output():
          self.ll.compute_linear_combinations()
      else:self.ll.compute_linear_combinations()
    else:
      self.ll = copy.deepcopy(ll)
    logdens = 8.3# try  9.5
    idens = np.argmin(abs(self.ll.density_array.value - 10**logdens))
    density_map = 10**logdens*np.ones(self.lines[0]['int'].shape, dtype=float)*u.cm**-3
    
    wvls = np.empty(shape=(len(self.lines),))
    for ind,line in enumerate(self.lines):wvls[ind] = line.wavelength
    
    data = []
    err  = []
    for ind,ionLF in  enumerate(self.ll.ionsLF):
      diff = np.abs(ionLF.wvl.value-wvls)
      argmin = np.argmin(diff)
      if diff[argmin]>0.1: raise Exception(f"Haven't found an ion for: {ionLF} {wvls}")
      self.ll.ionsLF[ind].int_map = self.lines[argmin]['rad']*u.W*u.m**-2/10
      data.append(self.lines[argmin]['rad'])
      err.append (self.lines[argmin]['rad_err'])
      # print(f"rad: {self.lines[argmin].hdul['int'][0].header['wavelength']},lc: {ionLF.wvl.value}")
    for ind,ionHF in  enumerate(self.ll.ionsHF):
      diff = np.abs(ionHF.wvl.value-wvls)
      argmin = np.argmin(diff)
      if diff[argmin]>0.1: raise Exception(f"Haven't found a for: {ionHF} {wvls}")
      self.ll.ionsHF[ind].int_map = self.lines[argmin]['rad']*u.W*u.m**-2/10
      data.append(self.lines[argmin]['rad'])
      err.append (self.lines[argmin]['rad_err'])
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
    for ind,line in enumerate(self.lines):wvls[ind] = line.wavelength
    diff = np.abs(wvls-wvl)
    ind = np.argmin(diff)
    if diff[ind]>1: raise Exception(f'big difference {wvl}\n{wvls}',)
    return self.lines[ind]
  def show_all_wvls(self):
    wvls = np.empty(shape=(len(self.lines),))
    for ind,line in enumerate(self.lines):wvls[ind] = line.wavelength
    return(wvls)

  def plot(self,params='all',axes =None):
    if params == 'all': params = ['int','wav','wid','rad']
    if isinstance(params, str): params = [params]
    if axes is None:
      axes=[]
      for ind in range(len(params)):
        c = 5
        r = len(self.lines)//c +(1 if len(self.lines)%c!=0 else 0)
        inch_size= 2
        _axes = gen_axes_side2side(r,c,figsize=(c*inch_size,r*inch_size),wspace=0,hspace=0,top_pad = 1/(c*2*inch_size),bottom_pad=0,left_pad=0,right_pad=0)[::-1].flatten()
        axes.append(_axes)
      axes = np.array(axes)
    data = self.lines[0]["int"]
    header = self.lines[0].headers['int']
    lon,lat = get_coord_mat(Map(data,header))
    
    for ind,param in enumerate(params):
      for ind2,line in enumerate(self.lines):
        data = line[param]
        axes[ind,ind2].pcolormesh(lon,lat,data,norm=normit(data),cmap="magma")
        axes[ind,ind2].text(0.5,0.95,self.lines[ind2].line_id+' '+param,transform=axes[ind,ind2].transAxes,ha='center',va='top',bbox=dict(facecolor='white', alpha=0.5))
    axes[0][0].figure.suptitle(
    self.lines[0].observatory + ' ' + self.lines[0].instrument + ' ' + self.lines[0].headers['int']['DATE-OBS'],
    va='top', ha='center'
    )

    axes[0][0].figure.tight_layout(rect=[0,0,1,0.95])
def get_celestial_L3(raster,**kwargs):
    if type(raster)==HDUList:
        
        shape = raster[0].data.shape
        wcs = WCS(raster[0].header)
        y   = np.arange(shape[0],dtype=int)
        x   = np.arange(shape[1],dtype=int)
        
        y = np.repeat(y,shape[1]).reshape(shape[0],shape[1])    
        x = np.repeat(x,shape[0]).reshape(shape[1],shape[0])
        x = x.T
        lon,lat,time = wcs.wcs_pix2world(x.flatten(),y.flatten(),0,0)
        
        
        lon[lon>180] -= 360
        lat[lat>180] -= 360   
        
        lon[lon<-180] += 360
        lat[lat<-180] += 360   
             
        lon  = lon.reshape(shape[0],shape[1])*3600
        lat  = lat.reshape(shape[0],shape[1])*3600
        time = time.reshape(shape[0],shape[1])
        
    elif isinstance(raster,WCS):
        shape = kwargs['shape']
        
        wcs = raster
        y   = np.arange(shape[0],dtype=int)
        x   = np.arange(shape[1],dtype=int)
        
        y = np.repeat(y,shape[1]).reshape(shape[0],shape[1])    
        x = np.repeat(x,shape[0]).reshape(shape[1],shape[0])
        x = x.T
        lon,lat,time = wcs.wcs_pix2world(x.flatten(),y.flatten(),0,0)
        
        lon[lon>180] -= 360
        lat[lat>180] -= 360   
        
        lon[lon<-180] += 360
        lat[lat<-180] += 360   
             
        lon  = lon.reshape(shape[0],shape[1])*3600
        lat  = lat.reshape(shape[0],shape[1])*3600
        time = time.reshape(shape[0],shape[1])
    elif isinstance(raster,SPICEL3Raster):
        lon,lat,time = get_celestial_L3(raster.lines[0])
    elif isinstance(raster,SPECLine):
        shape = raster['int'].shape
        wcs = WCS(raster.headers['int'])
        lon,lat,time = get_celestial_L3(wcs, shape= shape)
    else:
        print(f"The raster passed doesn't match any known types: {type(raster)} but it has to be one of these types: \n{SPECLine}\n{SPICEL3Raster}\n{HDUList}")
        raise ValueError("inacceptable type")
    return lon,lat,time 


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

  
  
  