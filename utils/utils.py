import math
import numpy as np
from scipy.ndimage import uniform_filter
import cv2
from time import sleep
from datetime import datetime 

import astropy
from astropy.io import fits as fits_reader
from astropy.io.fits.hdu.image import PrimaryHDU,ImageHDU
from astropy.visualization import SqrtStretch,PowerStretch, AsymmetricPercentileInterval, ImageNormalize, MinMaxInterval
from astropy.wcs import WCS
from sunraster.instr.spice import read_spice_l2_fits
import ndcube

from numba import jit
from multiprocess.shared_memory import SharedMemory 

import matplotlib.pyplot as plt

import os
import shutil
import pkg_resources
from  pathlib import Path, PosixPath
    


def function_to_string(func):
    source_lines, _ = inspect.getsourcelines(func)
    return ''.join(source_lines)
def flatten(iterable):
    flattened = []
    for item in iterable:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten(item))
        else:
            flattened.append(item)
    return flattened
def find_nth_occurrence(lst, element, n):
    occurrences = [index for index, item in enumerate(lst) if item == element]
    
    if n <= len(occurrences):
        return occurrences[n - 1]
    else:
        return -1  # Element not found or nth occurrence doesn't exist
def ArrToCode(arr):
        # Check if the input is a numpy array
        if not isinstance(arr, np.ndarray):
            raise ValueError("Input must be a numpy array")
        # Convert the array to a string
        arr_str = f'np.array({arr.tolist()})'
        arr_str = arr_str.replace('nan',"np.nan")
        return arr_str
def prepare_filenames(prefix=None, 
                      data_filename =None,
                      plot_filename=None, 
                      data_save_dir="./.p/",
                      plot_save_dir="./imgs/", 
                      i=None,
                      verbose=0):
    
    
    if type(prefix)==str:
        filename = prefix+"_window_{:03d}_"+"{:}.p"
    elif prefix==None:
        dir = data_save_dir
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir_list = os.listdir(dir); j=0
        for file in dir_list:
            try:
                j2 = int(file[0:3])
                if j2>=j:
                    j=j2+1
                        
            except Exception:
                pass
        j3 = j
        dir2 = dir
    if type(plot_filename)==str:
        if plot_filename.format(" ",0,0) == plot_filename: #make sure this passed variable is subscriptable 
            filename_a = plot_filename+"plot_{:03d}_{}_{}.jpg"
            filename_b = plot_filename+"hist_{:03d}_{}_{}.jpg"
            
    elif prefix==None:
        dir = plot_save_dir
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir_list = os.listdir(dir); j=0
        for file in dir_list:
            try:
                j2 = int(file[0:3])
                    
                if j2>=j:
                    j=j2+1
                        
                    
            except Exception:
                pass
        j = max(j3,j)
            #Delete these later------
        j=(i if type(i)!=type(None) else j)
        if verbose>=1:print("working in the file with prefix i={:03d} ".format(j))
            #------------------------
        filename_a = dir + "{:03d}_".format(j)+"plot_{:03d}_"+"{}_{}.jpg"
        filename_b = dir + "{:03d}_".format(j)+"hist_{:03d}_"+"{}_{}.jpg"
        filename   = dir2+"{:03d}_".format(j)+"window_{:03d}_"+"{}_{}.p"
    # print(data_filename)
    if type(data_filename)!= type(None):
        dir2 = data_save_dir
        dir  = plot_save_dir
        flnm = data_filename
        if flnm[-2:] == ".p":
            flnm = flnm[:-2]
        filename_a = dir + (
            "/" if (dir[-1] not in ["/","\\"]) or (flnm[0] not in ["/","\\"]) else 0  
            )+ flnm+"_plot_{1}.jpg"
        filename_b = dir + (
            "/" if (dir[-1] not in ["/","\\"]) or (flnm[0] not in ["/","\\"]) else 0  
            )+ flnm+"_hist_{1}.jpg"
        filename   = dir2 + (
            "/" if (dir2[-1] not in ["/","\\"]) or (flnm[0] not in ["/","\\"]) else 0  
            )+ flnm+"_hist_{1}.p"
    # print(filename_a,filename_b,filename)
    return filename,filename_a,filename_b
def clean_nans(xdata:np.ndarray,
               ydata:np.ndarray,
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
    assert xdata.shape==ydata.shape
    num_elements = np.zeros(xdata.shape)
    if type(weights) not in [str,type(None)]:
        num_elements = np.logical_not(
            (np.isnan(xdata)) | (np.isinf(xdata)) | (np.isinf(ydata)) | (np.isnan(ydata)) | (ydata<-100) | (np.isnan(weights)) | (np.isinf(1/weights)) )
    else: 
        num_elements = np.logical_not(
            (np.isnan(xdata)) | (np.isinf(xdata)) | (np.isinf(ydata)) | (np.isnan(ydata)) | (ydata<-100)  )
            
    
    clean_x = xdata[num_elements]; clean_y = ydata[num_elements]
    if type(weights) not in [str,type(None)]:
        weights = np.array(weights)
        assert xdata.shape==weights.shape
        # sigma = np.sum(weights[num_elements])/weights[num_elements]
        sigma = weights[num_elements]
        # if sigma[np.where(clean_y == np.max(clean_y))] < sigma[np.where(clean_x == np.min(clean_x))]:
        #     print("We found that the weights injected aren't decreasing with Intensity\n if you want to continue supress this message by deleting it from:\n SlimPy.clean_nans")
    elif  type(weights) == str:
        if weights == "1/sqrtI":
            weights = 1./np.sqrt(clean_y.copy())
        elif weights == "I":
            weights = clean_y.copy()
        elif weights == "expI":
            weights = clean_y.copy()**2
        elif weights == "I2":
            weights = np.exp(clean_y.copy())
        elif weights == "sqrtI":
            weights = np.sqrt(clean_y.copy())
        else: 
            raise ValueError ("the weights are unknown make sure you give the right ones\n current value: {} {} \n the allowed ones are: I, expI, I2, sqrtI".format(type(weights),weights))

        try:
            weights2=weights - np.nanmin(weights)
            weights = weights2
        except:
            pass
        sigma = 1/(weights.copy()/np.sum(weights))
    elif type(weights)== type(None):
        sigma = 1/(np.ones(len(clean_y))/len(clean_y))
    return clean_x,clean_y,sigma  
@jit(nopython=True)
def fst_neigbors(
        extent        : float    ,
        pixel_size_lon: float = 1,
        pixel_size_lat: float = 1,
        verbose       : float = 0
    ):
    """Generates a list of first neiboors in a square lattice and returns inside the list
        [n,m,n**2+m**2]

    Args:
        extent (float): how far the pixels will extend
    Return:
        nm_list (np.ndarray): list of data [n,m,n**2+m**2]
    """
    print(pixel_size_lon,pixel_size_lat)
    a = min(pixel_size_lon,pixel_size_lat) / pixel_size_lon
    b = min(pixel_size_lon,pixel_size_lat) / pixel_size_lat
    if verbose>=1: 
        print("a=",a,"b=",b) 
    nm_list = []
    extent_2 = extent**2
    for n in range(-extent,extent+1):
        for m in range(-extent,extent+1):
            s = a**2*n**2 + b**2*m**2
            if s <= extent_2:
                nm_list.append([n,m,s])
    return nm_list
@jit(nopython=True)
def join_px(data,i,j,ijc_list):
    res_px = float(0.)
    s = float(0.)
    
    for n_layer in ijc_list:
        i2,j2,c = n_layer 
        
        if (data.shape[0] - (i+i2) > 0
            and 
            data.shape[1] - (j+j2) > 0
            and
            i+i2 >= 0 
            and
            j+j2 >= 0
            ):
            if not np.isnan(data[i+i2,j+j2]):
                res_px += float(c*data[i+i2,j+j2])
                s += float(c)
    if s!=0:
        return (res_px/ s)
    else: return np.nan

# def join_dt(data,ijc_list):
#     ijc_list = np.array(ijc_list)
#     XY = ijc_list[:,0:2] 
#     maxes = np.max(np.abs(XY),axis=0).astype(int)
    
#     shape = data.shape
#     shape_ld     = np.array(data.shape)
#     shape_ld[2] += maxes[0]*2+1
#     shape_ld[3] += maxes[1]*2+1
#     _largedata = np.zeros(shape_ld)*np.nan
#     _largedata[:,:,maxes[0]:-maxes[0]-1,maxes[1]:-maxes[1]-1] = data
#     _shifted_datas = np.zeros([len(ijc_list),*data.shape])*np.nan
#     _coeffs_data   = np.zeros_like(_shifted_datas)*np.nan
#     for i in range(len(ijc_list)):
#          min_x = int(maxes[0]+XY[i,0])
#          max_x = int(_largedata.shape[2]-1+int(XY[i,0]-maxes[0]))
#          min_y = int(maxes[1]+XY[i,1])
#          max_y = int(_largedata.shape[3]-1+int(XY[i,1]-maxes[1]))
#          _shifted_datas[i] = _largedata[
#               :,:,
#               min_x:max_x,
#               min_y:max_y,
#               ] * ijc_list[i,2]
#          _coeffs_data[i][np.logical_not(np.isnan(_shifted_datas[i]))] = ijc_list[i,2]
#     conv_data = np.nansum(_shifted_datas,axis=0)/np.nansum(_coeffs_data,axis=0)
#     return conv_data


def _cv2blur(data,size):
    try:
        len(size)
    except:
        size= [size,size]
    
    blured = np.empty_like(data,)
    for i in range(data.shape[1]):
        blured[0,i] = cv2.blur(data[0,i]*1,size,
                               borderType= cv2.BORDER_REFLECT_101
                              )   
    return blured
def get_specaxis(
    hdu: PrimaryHDU or ImageHDU
    ) -> np.ndarray:
    """
    Get the spectral axis values from an HDU header using the WCS information.

    Args:
        hdu (PrimaryHDU or ImageHDU): An HDU (Header Data Unit) object of type PrimaryHDU or ImageHDU.

    Returns:
        np.ndarray: Array of spectral axis values in angstroms.
    """
    spec_pix = np.arange(hdu.data.shape[1])
    wcs = WCS(hdu.header)
    _,_,specaxis,_ = wcs.wcs_pix2world(0,0,spec_pix,0,0)
    specaxis*=10**10
    return specaxis
def _sciunif(data,size):
    try:
        len(size)
    except:
        size= [size,size]
    blured = np.empty_like(data,)
    
    blured[0] = uniform_filter(data[0]*1,size,
                    mode='reflect'
                    )
    return blured
def deNaN(data):
    clean_data = data.copy()
    
    
    for i in range(clean_data.shape[1]):
        xNaN = np.nanmean(data[:,i,:,:],axis=(0,1))
        yNaN = np.nanmean(data[:,i,:,:],axis=(0,2))
        if len(np.where(np.logical_not(np.isnan(xNaN)))[0]) ==0 :
            max_lon = 0
            min_lon = 0
        else:
            max_lon = np.max(np.where(np.logical_not(np.isnan(xNaN))))+1
            min_lon = np.min(np.where(np.logical_not(np.isnan(xNaN))))
        if len(np.where(np.logical_not(np.isnan(yNaN)))[0]) == 0:
            max_lat = 0
            min_lat = 0
        else:
            min_lat = np.min(np.where(np.logical_not(np.isnan(yNaN))))
            max_lat = np.max(np.where(np.logical_not(np.isnan(yNaN))))+1
        
        clean_data[0,i,0:min_lat, :       ] = 0
        clean_data[0,i, :       ,0:min_lon] = 0
        clean_data[0,i,max_lat: , :       ] = 0
        clean_data[0,i, :       ,max_lon: ] = 0
        
    return clean_data
def reNaN(original_data,clean_data,size):
    data = original_data.copy()
    reclean_data =  clean_data.copy()
    for i in range(clean_data.shape[1]):
        xNaN = np.nanmean(data[:,i,:,:],axis=(0,1))
        yNaN = np.nanmean(data[:,i,:,:],axis=(0,2))
        
        if len(np.where(np.logical_not(np.isnan(xNaN)))[0]) ==0 :
            max_lon = 0
            min_lon = 0
        else:
            max_lon = np.max(np.where(np.logical_not(np.isnan(xNaN))))+1
            min_lon = np.min(np.where(np.logical_not(np.isnan(xNaN))))
        if len(np.where(np.logical_not(np.isnan(yNaN)))[0]) == 0:
            max_lat = 0
            min_lat = 0
        else:
            min_lat = np.min(np.where(np.logical_not(np.isnan(yNaN))))
            max_lat = np.max(np.where(np.logical_not(np.isnan(yNaN))))+1
        min_lat = min_lat + (  size[0]//2+  (1 if size[0]%2!=0 else 0)  ) 
        min_lon = min_lon + (  size[1]//2+  (1 if size[1]%2!=0 else 0)  ) 
        max_lat = max_lat - (  size[0]//2+  (1 if size[0]%2!=0 else 0)  ) 
        max_lon = max_lon - (  size[1]//2+  (1 if size[1]%2!=0 else 0)  ) 

        min_lat = np.min([min_lat,data.shape[3]]) 
        min_lon = np.min([min_lon,data.shape[2]])                   
        max_lat = np.max([max_lat,0            ])                   
        max_lon = np.max([max_lon,0            ])                   
                
        reclean_data[0,i,0:min_lat-1, :       ] = np.nan
        reclean_data[0,i, :       ,0:min_lon-1] = np.nan
        reclean_data[0,i,max_lat+1: , :       ] = np.nan
        reclean_data[0,i, :       ,max_lon+1: ] = np.nan
        
    return reclean_data
@jit(nopython=True)   
def join_dt(data,ijc_list):
    # data_new = np.zeros_like(data,dtype=float) #numba has no zeros_like 
    data_new = data.copy()*np.nan
    for k in range(data.shape[0]):
        for l in range(data.shape[1]):
            for i in range(data.shape[2]):
                for j in range(data.shape[3]):
                    data_new[k,l,i,j] = join_px(data[k,l],i,j,ijc_list)
    return data_new
def convolve(window,mode,lon_pixel_size,lat_pixel_size,convolution_extent_list,convolution_function,verbose=0   ):
    print(f"convolving using {mode}")
    if mode == "cercle":
        conv_data = np.zeros((*convolution_extent_list.shape,*window.shape))
        if verbose>=2: print('creating convolution list...')
        for i in range(convolution_extent_list.shape[0]):
            if convolution_extent_list[i] == 0:
                conv_data[i]=window.copy();continue
            else:
                ijc_list = np.array(fst_neigbors(convolution_extent_list[i],lon_pixel_size,lat_pixel_size,verbose=verbose)).astype(int)
                # print(ijc_list)
                # sleep(5)
                ijc_list [:,2]= convolution_function(ijc_list)
                conv_data[i]  = join_dt(window, ijc_list)
    elif mode == "box":
        conv_data = np.zeros((*convolution_extent_list.shape,*window.shape))
        clean_window = deNaN(window)
        for i in range(convolution_extent_list.shape[0]):
            if lat_pixel_size< lon_pixel_size:
                size=np.array([1,
                        1+(convolution_extent_list[i]),
                        1+(convolution_extent_list[i])* lat_pixel_size/ lon_pixel_size,
                        ],dtype=int)      
            else:
                size=np.array([1,
                        1+(convolution_extent_list[i])*lon_pixel_size/ lat_pixel_size,
                        1+(convolution_extent_list[i]),
                        ],dtype=int)
            # print("box size:",size)
            if verbose>=2: print('creating convolution list...')
            for i in range(convolution_extent_list.shape[0]):
                if convolution_extent_list[i] == 0:
                    conv_data[i]=window.copy();continue
                else:
                    conv_data[i] = reNaN(window,_sciunif(clean_window,size),size[1:]) 
    else: raise ValueError(f"mode:{mode} is not implemented or there is a misspelling")
    return conv_data
# @jit(nopython=True) #not tryed yet
def Preclean(cube):
    cube2 = cube.copy()
    # logic=np.logical_or(np.isinf(cube2),cube2<-10**10)
    logic=np.logical_or(cube2>490,cube2<-10)
    cube2[logic]=np.nan
    if False: #this part is for elemenating the cosmic effected values but it's not well done (see it again)
        mean_cube = np.nanmean(cube2, axis=1)*1000
        for i in range(cube2.shape[1]):
            cube2[:,i,:,:][cube2[:,i,:,:]>mean_cube] = np.nan
        
    return cube2
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
def gen_shmm(create = False,name=None,ndarray=None,size=0,shape=None,dtype=float):
    assert (type(ndarray)!=type(None) or size!=0) or type(name)!=type(None)
    return math.ceil(n * multiplier) / multiplier
    assert type
    (ndarray)!=type(None) or type(shape)!=type(None)
    size = size if type(ndarray) == type(None) else ndarray.nbytes
    shmm = SharedMemory(create = create,size=size,name=name)
    shmm_data = np.ndarray(shape = shape if type(ndarray)==type(None) else ndarray.shape  
                           , buffer = shmm.buf , dtype=dtype)
    
    if create and type(ndarray)!=type(None):
        shmm_data[:] = ndarray[:]
    elif create:
        shmm_data[:] = np.nan
        
    return shmm,shmm_data
def verbose_description(verbose):
    print(f"level {verbose:01d} verbosity")
    if verbose ==-2:
        print("On-screen information mode: Dead \nNo information including warnings  (CAREFUL DUDE!!)")
    elif verbose ==-1:
        print("On-screen information mode: Minimal\nHighly important ones only and wornings (Don't have a blind faith please)")
        
    elif verbose ==0:
        print("On-screen information mode: Normie\nBasic information any normie needs")
    
    elif verbose ==1:
        print("On-screen information mode: Extra\nmore detailed information for tracking and debugging")
    elif verbose ==2:
        print("On-screen information mode: Stupid\nUnless you are as stupid as the writer of this script, you don't need this much information for debugging an error")
    elif verbose ==3:
        print("On-screen information mode: Visual\nPlot extra figures in a ./tmp file with ")
def gen_velocity(doppler_data,quite_sun=[60,150,550,600],correction=False,verbose=0,get_0lbdd = False):
    qs = quite_sun
    mean_doppler = np.nanmean(doppler_data[qs[2]:qs[3],qs[0]:qs[1]]) 
    # print("mean_doppler",mean_doppler)
    results = (doppler_data-mean_doppler) / mean_doppler * 3*10**5
    if correction:
        if verbose >0: print("Correcting")
        hist,bins= gen_velocity_hist(results,bins=np.arange(-600,600,1),verbose=verbose)
        vel_corr,ref = correct_velocity(hist,bins,verbose=verbose)
        if verbose >0: print(f"The correction found the distribution was off by {ref}")
        results -= ref
        
    if verbose > 1:
        fig=plt.figure()
        plt.pcolormesh(results,cmap="twilight_shifted",vmax=80,vmin=-80)
        plt.plot([qs[1],qs[0],qs[0],qs[1],qs[1]],
                    [qs[2],qs[2],qs[3],qs[3],qs[2]],color="green",label="mean value {:06.1f}".format(mean_doppler))
        plt.legend()
        # plt.savefig('fig_test.jpg')
    return (results, (None if not correction else ref),(None if not get_0lbdd else mean_doppler)) 
def gen_velocity_hist(velocity_data,axis=None,bins = None,verbose=0):
    if type(bins)==type(None):bins = np.linspace(np.nanmin(velocity_data),np.nanmax(velocity_data),num=200)
    hist,bins = np.histogram(velocity_data,bins = bins) 
    bins = (bins[:-1] + bins[1:])/2
    if verbose>1:
        if type(axis) == type(None): 
            fig,axis = plt.subplots(1,1)        
        axis.step(bins,hist)
        axis.set_yscale('log',base=10)
        plt.axvline(0,ls="--",color='red',alpha=0.5)
    return hist, bins
def correct_velocity(velocity_hist,velocity_values,verbose=0):
    if verbose>1: print("correct_velocity<func>.velocity_hist.shape: {}\n,correct_velocity<func>.velocity_values.shape: {}".format(velocity_hist.shape,velocity_values.shape))
    ref_velocity = velocity_values[np.where(velocity_hist == np.nanmax(velocity_hist))[0]]
    ref_velocity = np.mean(ref_velocity)
    if verbose>0: print(f"the velocity reference was found at {ref_velocity}\n now it will be set to 0")
    velocity_values_corr = velocity_values-ref_velocity
    return velocity_values_corr,ref_velocity

# def get_celestial(raster):
 #    unq = spu.unique_windows(raster)
 #    lon = raster[unq[0]].celestial.data.lon.arcsec
 #    lat = raster[unq[0]].celestial.data.lat.arcsec
 #    lon[lon>180*3600]-=360*3600
 #    lat[lat>180*3600]-=360*3600
 #    return lon,lat
#def get_celestial(raster,include_time=False):
  #      if type(raster)==astropy.io.fits.hdu.hdulist.HDUList:
           
  #          shape = raster[0].data.shape
  #          wcs = WCS(raster[0].header)
  #          y   = np.arange(shape[2],dtype=np.int)
  #          x   = np.arange(shape[3],dtype=np.int)
           
  #          y = np.repeat(y,shape[3]).reshape(shape[2],shape[3])    
  #          x = np.repeat(x,shape[2]).reshape(shape[3],shape[2])
  #          x = x.T
  #          lon,lat,_,time = wcs.wcs_pix2world(x.flatten(),y.flatten(),0,0,0)
  #          time = time.reshape(shape[2],shape[3])
           
  #          lon[lon>180] -= 360
  #          lat[lat>180] -= 360   
           
  #          lon[lon<-180] += 360
  #          lat[lat<-180] += 360   
                
  #          lon  = lon.reshape(shape[2],shape[3])*3600
  #          lat  = lat.reshape(shape[2],shape[3])*3600
           
           
  #      elif type(raster) == ndcube.ndcollection.NDCollection:
  #          if include_time:raise ValueError("you can't get time with sunraster data set either change to astropy raster or set include_time to False")
  #          unq = spu.unique_windows(raster)
  #          lon = raster[unq[0]].celestial.data.lon.arcsec
  #          lat = raster[unq[0]].celestial.data.lat.arcsec
       
  #      elif type(raster) == tuple:
  #          header_dict = raster[4][0]
  #          shape = raster[0][0][0][0].shape
  #          wcs_dict = {}
  #          for key in header_dict:
  #              if (
  #                  "" != key and 
  #                  "\n"  not in str(header_dict[key]) and 
  #                  len(str(header_dict[key]))<=30 and 
  #                  len(key)<=10):
  #                  wcs_dict[key]=header_dict[key]
  #          wcs = WCS(wcs_dict)   
  #          y   = np.arange(shape[0],dtype=np.int)
  #          x   = np.arange(shape[1],dtype=np.int)
           
  #          y = np.repeat(y,shape[1]).reshape(shape[0],shape[1])    
  #          x = np.repeat(x,shape[0]).reshape(shape[1],shape[0])
  #          x = x.T
  #          lon,lat,_,time = wcs.wcs_pix2world(x.flatten(),y.flatten(),0,0,0)
  #          time = time.reshape(shape[0],shape[1])
           
  #          lon[lon>180] -= 360
  #          lat[lat>180] -= 360   
           
  #          lon[lon<-180] += 360
  #          lat[lat<-180] += 360   
                
  #          lon  = lon.reshape(shape[0],shape[1])*3600
  #          lat  = lat.reshape(shape[0],shape[1])*3600
                
  #      else:
  #          print(f"The raster passed doesn't match any known types: {type(raster)} but it has to be one of these types: \n{ndcube.ndcollection.NDCollection}\n{astropy.io.fits.hdu.hdulist.HDUList}")
  #          raise ValueError("inacceptable type")
  
  #      return (lon,lat,time) if include_time else (lon,lat)
def get_celestial(raster,include_time=False,**kwargs):
    if type(raster)==astropy.io.fits.hdu.hdulist.HDUList:
        
        shape = raster[0].data.shape
        wcs = WCS(raster[0].header)
        y   = np.arange(shape[2],dtype=int)
        x   = np.arange(shape[3],dtype=int)
        
        y = np.repeat(y,shape[3]).reshape(shape[2],shape[3])    
        x = np.repeat(x,shape[2]).reshape(shape[3],shape[2])
        x = x.T
        lon,lat,_,time = wcs.wcs_pix2world(x.flatten(),y.flatten(),0,0,0)
        time = time.reshape(shape[2],shape[3])
        
        lon[lon>180] -= 360
        lat[lat>180] -= 360   
        
        lon[lon<-180] += 360
        lat[lat<-180] += 360   
             
        lon  = lon.reshape(shape[2],shape[3])*3600
        lat  = lat.reshape(shape[2],shape[3])*3600
    
    elif type(raster) == ndcube.ndcollection.NDCollection:
        if include_time:raise ValueError("you can't get time with sunraster data set either change to astropy raster or set include_time to False")
        unq = spu.unique_windows(raster)
        
        lon = raster[unq[0]].celestial.spherical.lon.arcsec
        lat = raster[unq[0]].celestial.spherical.lat.arcsec
    
    elif type(raster) == tuple:
        header_dict = raster[4][0]
        shape = raster[0][0][0][0].shape
        wcs_dict = {}
        for key in header_dict:
            if (
                "" != key and 
                "\n"  not in str(header_dict[key]) and 
                len(str(header_dict[key]))<=30 and 
                len(key)<=10):
                wcs_dict[key]=header_dict[key]
        wcs = WCS(wcs_dict)   
        y   = np.arange(shape[0],dtype=int)
        x   = np.arange(shape[1],dtype=int)
        
        y = np.repeat(y,shape[1]).reshape(shape[0],shape[1])    
        x = np.repeat(x,shape[0]).reshape(shape[1],shape[0])
        x = x.T
        lon,lat,_,time = wcs.wcs_pix2world(x.flatten(),y.flatten(),0,0,0)
        time = time.reshape(shape[0],shape[1])
        
        lon[lon>180] -= 360
        lat[lat>180] -= 360   
        
        lon[lon<-180] += 360
        lat[lat<-180] += 360   
             
        lon  = lon.reshape(shape[0],shape[1])*3600
        lat  = lat.reshape(shape[0],shape[1])*3600
        
    elif isinstance(raster,WCS):
        shape = kwargs['shape']
        
        wcs = raster
        y   = np.arange(shape[0],dtype=int)
        x   = np.arange(shape[1],dtype=int)
        
        y = np.repeat(y,shape[1]).reshape(shape[0],shape[1])    
        x = np.repeat(x,shape[0]).reshape(shape[1],shape[0])
        x = x.T
        lon,lat,time = wcs.wcs_pix2world(x.flatten(),y.flatten(),0,0)
        time = time.reshape(shape[0],shape[1])
        
        lon[lon>180] -= 360
        lat[lat>180] -= 360   
        
        lon[lon<-180] += 360
        lat[lat<-180] += 360   
             
        lon  = lon.reshape(shape[0],shape[1])*3600
        lat  = lat.reshape(shape[0],shape[1])*3600
    else:
        print(f"The raster passed doesn't match any known types: {type(raster)} but it has to be one of these types: \n{ndcube.ndcollection.NDCollection}\n{astropy.io.fits.hdu.hdulist.HDUList}")
        raise ValueError("inacceptable type")
    return (lon,lat,time) if include_time else (lon,lat)
def quickview(
    RasterOrPath, 
    fig1 = None, imag_ax = None, 
    fig2 = None, spec_ax = None, 
              ):
    from pathlib import PosixPath,Path
    if type(RasterOrPath) in (str,PosixPath):
        raster = read_spice_l2_fits(str(RasterOrPath))
    else: raster = RasterOrPath
    unq = spu.unique_windows(raster)
    lon,lat = get_celestial(raster)
    n= 3
    m= len(unq)//3 +( 1 if len(unq)%3 != 0 else 0)
    
    if type(imag_ax) == type(None): fig1,ax1 = plt.subplots(m,n,figsize=(n*3,m*3),sharex=True,sharey=True);ax1 = ax1.flatten() 
    if type(spec_ax) == type(None): fig2,ax2 = plt.subplots(m,n,figsize=(n*3,m*3)                        );ax2 = ax2.flatten() 
    fig1.suptitle(raster[unq[0]].meta['DATE_EAR'])
    fig2.suptitle(raster[unq[0]].meta['DATE_EAR'])
    for i,kw in enumerate(unq):
        data = raster[unq[i]].data
        image = np.nanmean(data, axis =(0,1))
        spect = np.nanmean(data, axis =(0,2,3))
        spec_ax = raster[unq[i]].spectral_axis*10**10
        
        norm = ImageNormalize(data,
                              interval=AsymmetricPercentileInterval(1, 99),
                              stretch=SqrtStretch()
                              )
        
        ax1[i].pcolormesh(lon,lat,image,norm=norm)
        ax2[i].step(spec_ax,spect)
        ax1[i].set_title(kw)
        ax2[i].set_title(kw)
    return ((fig1,ax1),(fig2,ax2))
def getfiles(
    YEAR="all",
    MONTH="all",
    DAY="all",
    STD_TYP="all",
    STP_NUM="all",
    SOOP_NAM="all",
    MISOSTUD_NUM = "all",
    in_name=None,
    verbose = 0,
    L2_folder:str="/archive/SOLAR-ORBITER/SPICE/fits/level2/",
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
        if True:#reading years
            if type(YEAR) in [int,float]:
                years = [YEAR]
            elif type(YEAR) in  (list,np.ndarray):
                years = np.array(YEAR)
            elif YEAR in ["ALL","all"]:
                years = np.array([i for i in range(2018,2030)])
            else:
                raise ValueError("YEAR should be an integer, a list, or a string of value \"all\" not {}".format(type(YEAR))) 
        if True:#reading months
            if type(MONTH) in [int,float]:
                months = [MONTH]
            elif type(MONTH) in  (list,np.ndarray):
                months = np.array(MONTH)
            elif MONTH in ["ALL","all"]:
                months = np.array([i for i in range(1,13)])
            else:
                raise ValueError("MONTH shuld be an integer, a list, or a string of value \"all\" not {}".format(type(MONTH))) 
        if True:#reading days
            if type(DAY) in [int,float]:
                days = [DAY]
            elif type(DAY) in  (list,np.ndarray):
                days = np.array(DAY)
            elif DAY in ["ALL","all"]:
                days = np.array([i for i in range(1,32)])
            else:
                raise ValueError("DAY shuld be an integer, a list, or a string of value \"all\" not {}".format(type(DAY))) 
        for day in days:#combining the path to the targeted folders
            for month in months:
                for year in years:
                    searching_paths.append(path_l2/f"{year}"/f"{month:02d}"/f"{day:02d}")       
        if verbose >= 2:#for seeing what are the paths chosen 
            for i in searching_paths: print(i)
        for path in searching_paths:#combining the path to the targeted set and filtring study type
            if Path.exists(path):
                available_fits = os.listdir(path)
                # print(path,available_fits)
                for fits in available_fits:
                    if str(fits)[-5:]==".fits":
                        _sample = path/fits
                        # print("in_name",in_name)
                        if in_name is not None:
                            if type(in_name) == str:in_name = [in_name]
                            for name in in_name:
                                if name in str(fits): selected_fits.append(_sample)
                        else:   
                            # print('this is the data',_sample) 
                            data = fits_reader.open(_sample)
                            # PURPOSE = data[0].header['PURPOSE']
                            STUDY     = data[0].header['STUDY'].upper()
                            STP       = data[0].header['STP']
                            SOOPNAME  = data[0].header['SOOPNAME'].upper()
                            MISOSTUD  = data[0].header['MISOSTUD']
                            # print(STUDY   )      
                            # print(STP     )    
                            # print(SOOPNAME)         
                            # print(MISOSTUD)         
                            if True:
                                GOOD_STP = False
                                if (STP_NUM in ["ALL","all"]):
                                    GOOD_STP = True
                                elif (int(STP) == int(STP_NUM) ):
                                    GOOD_STP = True
                            if True: 
                                GOOD_MISOSTUD = False
                                if (MISOSTUD_NUM in ["ALL","all"]):
                                    GOOD_MISOSTUD = True
                                elif (int(MISOSTUD) == int(MISOSTUD_NUM) ):
                                    GOOD_MISOSTUD = True
                                
                            if (
                                ((STD_TYP in ["ALL","all"]) or (STD_TYP in STUDY        ))
                                and
                                GOOD_STP
                                and
                                ((SOOP_NAM in ["ALL","all"]) or (SOOP_NAM in SOOPNAME   ))
                                ): 
                            
                                selected_fits.append(_sample)
        if verbose >= 1:
            for i in (selected_fits): print(i,"***********")
        return(selected_fits)

def puke_template(where="./input_config_template.json"):
    PATH = pkg_resources.resource_filename("SlimPy", "manager/input_config_template.json")
    shutil.copy(PATH,where)
