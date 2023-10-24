# import matplotlib.pyplot as plt
# import matplotlib as mplt
# from astropy.visualization import SqrtStretch, AsymmetricPercentileInterval, ImageNormalize

# import pickle as p 
# from astropy.io import fits
# import numpy as np
# from astropy.wcs import WCS
# import os 
# from pathlib import Path
# from multiprocessing import Process,Lock
# import time

# from sunraster.instr.spice import read_spice_l2_fits
# from spice_utils.ias_spice_utils import utils as spu 

# from ..init_handler import getfiles,get_celestial
# from ..visualization import _plot_window_Miho
# from ..fit_models import flat_inArg_multiGauss 
# from ..utils import fst_neigbors,join_dt,gen_velocity,gen_shmm
# from ..utils.denoise import denoise_data
# from ..utils.error_finder import get_spice_errors


# class SPICE_raster():
#     def __init__(self):
#         self.L2_data = None
#         self.L2_data_astropy = None
#         self.L2_data_offsetCorr = None
#         self.L3_data = None
#         self.conv_data = None #L2 data with extra ax of convolution data 
#         self.conv_data_denoise = None #if data where denoised this will be deoised L2 data with extra ax of convolution data
#         self.real_L2_data = None #the real data that has been analysed (L2 data whether convoluted or denoised)
        
#         self.conv_sigmas = None #extract uncertainty from the L2 row data using Eric BUCHLIN script and get upper levels of convolution equivalent of uncertainty
#         self.real_sigmas = None #extract real uncertainty from the L2 row data using Eric BUCHLIN script and get uncertainty for convolved data 
#         self.X2_3D  = None #X2 for every pixel
#         self.X2_2D  = None #X2 for x and y 
         
#         self.L2_path            = None
#         self.L3_path            = None
#         self.L2_Path_offsetCorr = None
#         self.cache_path         = None
        
#         self.meta    = None
#         self.lon     = None
#         self.lat     = None
#         self.unq     = None
#         self.model   = flat_inArg_multiGauss
#         self.convolution_extent_list = None
#         self.quiet_sun = np.array([[0,-1],[0,-1]])
#         self.EXTNAMES = None
        
#         self.verbose = 0
        
#     def charge_data(self, L3_PathOrRaster,L2_PathOrRaster=None,cache =None):
#         print("charging data for the file: ",L3_PathOrRaster)
#         if type(L3_PathOrRaster) == list:
#             self.L3_data = L3_PathOrRaster
#         else:
#             self.L3_data = p.load(open(L3_PathOrRaster,"rb"))
#         try:
#             self.EXTNAMES = [self.L3_data[4][i]["EXTNAME"] for i in range(len(self.L3_data[4]))] 
#             meta = self.L3_data[4] if type(self.L3_data[4]) != list else self.L3_data[4][0]
#         except:
#             print("there is only one window")
#             # self.L3_data = list(self.L3_data)
#             # for (i,data_type) in enumerate(self.L3_data):
#             #     self.L3_data[i] = [data_type]
#             # self.EXTNAMES = [self.L3_data[4][i]["EXTNAME"] for i in range(len(self.L3_data[4]))]
            
        
#         if type(L2_PathOrRaster) == type(None):    
#             date     = self.L3_data[4][0]["DATE-OBS"]
#             year     = date[0:4]
#             month    = date[5:7]
#             day      = date[8:10]
#             hour     = date[11:13]
#             minute   = date[14:16]
#             seond    = date[17:19]
#             str_date = year +month +day +"T"+hour +minute +seond
#             print("reading L2 data for the date {}_{}_{}".format(year,month,day))
#             selected_date = getfiles(YEAR=int(year),
#                                     MONTH=int(month),
#                                     DAY=int(day),
#                                     STD_TYP="ALL",
#                                     verbose = 0)
#             selected_file = []
#             for file in selected_date:
#                 if str_date in str(file):
#                     selected_file.append(file)
#             if len(selected_file)>1: raise ValueError(f'found more than 1 selected file {selected_file}')
#             L2_PathOrRaster = selected_file[0]
        
#         self.L2_data_astropy = fits.open(str(L2_PathOrRaster)) 
#         if True: #apply correction
#             self.L2_Path_offsetCorr = Path('./tmp/offset_corr_fits/')/(Path(L2_PathOrRaster).name)
#             # print("os.path.exists(self.L2_Path_offsetCorr)",os.path.exists(self.L2_Path_offsetCorr))
#             if True:#not os.path.exists(self.L2_Path_offsetCorr):
#                 for i in range(len(self.L2_data_astropy)):
#                     try:
#                         dx, dy = self._rotate(self.L2_data_astropy[i].header['CRVAL1'], self.L2_data_astropy[i].header['CRVAL2'], np.deg2rad(+ self.L2_data_astropy[i].header['CROTA']))
#                         dx = dx - (- 83) + (0.46 * self.L2_data_astropy[i].header['T_GRAT'] - 85.0)
#                         dy = dy - (- 68) + (- 72.2)
#                         # print(dx,dy,"DELETE")
#                         self.L2_data_astropy[i].header['CRVAL1'], self.L2_data_astropy[i].header['CRVAL2'] = self._rotate(dx, dy, np.deg2rad(- self.L2_data_astropy[i].header['CROTA']))
#                     except:
#                         pass
#                 #save fits file:
#                 if not os.path.exists("./tmp"):
#                     os.mkdir("./tmp/")
#                 if not os.path.exists('./tmp/offset_corr_fits/'):
#                     os.mkdir('./tmp/offset_corr_fits/')
#                 self.L2_data_astropy.writeto(self.L2_Path_offsetCorr,overwrite=True)
        
#         convolution_extent_list = np.unique(self.L3_data[3])
#         convolution_extent_list = convolution_extent_list[np.logical_not(np.isnan(convolution_extent_list))]
#         self.convolution_extent_list = convolution_extent_list       
#         self.L2_data_offsetCorr = read_spice_l2_fits(str(self.L2_Path_offsetCorr))
#         self.L2_data = read_spice_l2_fits(str(L2_PathOrRaster)) 
        
#         self.lon, self.lat = get_celestial(self.L2_data_offsetCorr)
#         self.unq = spu.unique_windows(self.L2_data_offsetCorr)
#         self.L2_path = L2_PathOrRaster
#         self.L3_path = L3_PathOrRaster
        
#         if cache=="r":
#             self.cache(mode ='r')
#         else: 
#             print("generating convoluted data")
#             self.gen_convData()
#             # print("generating sigmas")
#             # self.gen_sigmas()
#             print("generating real sigmas data")
#             self.gen_real_sigmas()
#             print("generating real L2 data")
#             self.gen_real_L2_data()
#             print("real_L2 generated")
#             self.gen_X2()
#             if cache=="w":
#                 self.cache(mode = "w")
            
#     def cache(self,mode = "r",verbose=None):
#         L3_path = Path(self.L3_path)
#         file_path = Path(L3_path.parent) / ("."+L3_path.stem+".p") 
#         self.cache_path = file_path
#         if mode in "w" :
#             data = {
#                 "conv_data":None,
#                 "conv_data_denoise":None,
#                 "real_L2_data":None,
#                 "conv_sigmas":None,
#                 "real_sigmas":None,
#                 "X2_3D":None,
#                 "X2_2D":None,
#             }
#             data["conv_data"] = self.conv_data
#             data["conv_data_denoise"] = self.conv_data_denoise
#             data["real_L2_data"] = self.real_L2_data
#             data["conv_sigmas"] = self.conv_sigmas
#             data["real_sigmas"] = self.real_sigmas
#             data["X2_3D"] = self.X2_3D
#             data["X2_2D"] = self.X2_2D
#             if file_path.exists(): print(f"Warning: the file {str(file_path)} do exist")
#             p.dump(data,open(file_path,"wb"))
#             print("cache created in: {}".format(str(file_path)))         
#         elif mode == "r":
#             if file_path.exists():
#                 data =p.load(open(file_path,"rb"))
#                 self.conv_data = data["conv_data"] 
#                 self.conv_data_denoise = data["conv_data_denoise"] 
#                 self.real_L2_data = data["real_L2_data"] 
#                 self.conv_sigmas = data["conv_sigmas"] 
#                 self.real_sigmas = data["real_sigmas"] 
#                 self.X2_3D = data["X2_3D"] 
#                 self.X2_2D = data["X2_2D"] 
#                 print("cache loaded from {}".format(str(file_path)))
#             else: 
#                 print(f"the file: {file_path} has not been found\nComputing instead")
#                 print("generating convoluted data")
#                 self.gen_convData()
#                 print("generating real sigmas data")
#                 self.gen_real_sigmas()
#                 print("generating real L2 data")
#                 self.gen_real_L2_data()
#                 print("real_L2 generated")
#                 self.gen_X2()
#                 self.cache('w')
#         else: raise ValueError(f"the reading mode of cache should be 'r' for read or 'w' for write and not {mode}")
        
#     def gen_X2(self,force_building=False):
#         if ((type(self.X2_3D)==type(None))or force_building): 
#             self.gen_sigmas()
#             self.gen_real_L2_data()
#             self.X2_3D = []
#             self.X2_2D = []
#             if type(self.conv_sigmas)==None: print("self.conv_sigmas is not defined")
#             for i,kw in enumerate(self.EXTNAMES):
#                 X2_3D = np.zeros(self.L2_data[kw].data.shape)*np.nan
#                 wvl_ax = (self.L2_data[kw].spectral_axis.value).astype(float)
#                 params = self.L3_data[0][i]
#                 for i_y in range(self.L2_data[kw].data.shape[2]):
#                     for i_x in range(self.L2_data[kw].data.shape[3]):
#                         X2_3D[0,:,i_y,i_x] = (self.model(wvl_ax,*params[:,0,i_y,i_x])-self.real_L2_data[i][0,:,i_y,i_x])**2
#                 self.X2_3D.append(X2_3D/self.real_sigmas[i]**2/params.shape[0])
#                 self.X2_2D.append(np.nansum(self.X2_3D[i],axis=(0,1)))
#         elif self.verbose>=1: 
#             print("The X2 data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")
            
#     def gen_convData(self, force_building = False,verbose = None): #convlist handeling
#         if False:
#             if (type(self.conv_data)==type(None)
#                     or force_building):
#                 convolution_function = lambda lst:np.zeros_like(lst[:,2])+1
#                 lat_pixel_size= abs(np.nanmean(self.lat[1:,:]-self.lat[:-1,:]))
#                 lon_pixel_size= abs(np.nanmean(self.lon[:,1:]-self.lon[:,:-1]))
                
#                 print("lat/lon pixel size",lat_pixel_size,lon_pixel_size )
#                 sigma_denoise = self.L3_data[4][0]["Slim_denoise"]
                
#                 self.conv_data = []
#                 self.conv_data_denoise = []
#                 if type(sigma_denoise) == type(None):  sigma_denoise=[0,0] 
#                 self.conv_data_denoise = []
                
#                 for j,kw in enumerate(self.unq):
#                     conv_data = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[self.unq[j]].data.shape))
#                     # if type(sigma_denoise)!=type(None):
#                     conv_data_denoise = conv_data.copy()
#                     for i in range(self.convolution_extent_list.shape[0]):
#                         if self.convolution_extent_list[i] == 0:
#                             conv_data[i]=self.L2_data[self.unq[j]].data.copy()
#                         else:
#                             ijc_list = np.array(fst_neigbors(self.convolution_extent_list[i],lon_pixel_size,lat_pixel_size)).astype(int)
#                             ijc_list [:,2]= convolution_function(ijc_list)
#                             conv_data[i]  = join_dt((self.L2_data[self.unq[j]].data).astype(float), ijc_list)
#                             conv_data_denoise[i,0] = denoise_data(conv_data[i,0],denoise_sigma=sigma_denoise)
                        
                            
#                     self.conv_data.append(conv_data.copy())
#                     # if type(sigma_denoise) != type(None): 
#                     self.conv_data_denoise.append(conv_data_denoise.copy())
                    
#             elif self._is_True_verbose(verbose,1):
#                 print("The convoluted data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")
#         if False:
#             if (type(self.conv_data)==type(None)
#                     or force_building):
#                 def _task_conv_data(l_con,i_con,
#                                     raw,con,dns,
#                                     convolution_function,lat_pixel_size,lon_pixel_size,sigma_denoise):
                    
#                     shmm_raw,data_raw = gen_shmm(create=False,name=raw["name"],dtype=raw["type"],shape=raw["shape"]) 
#                     shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"]) 
#                     shmm_dns,data_dns = gen_shmm(create=False,name=dns["name"],dtype=dns["type"],shape=dns["shape"]) 
#                     start_time = time.time()
#                     ijc_list = np.array(fst_neigbors(l_con,lon_pixel_size,lat_pixel_size)).astype(int)
#                     ijc_list [:,2]= convolution_function(ijc_list)
#                     conv_data = join_dt(data_raw, ijc_list)
#                     dnys_data = denoise_data(conv_data[0],denoise_sigma=sigma_denoise)
#                     if self._is_True_verbose(verbose=verbose,value=2):print("running time for a task",time.time()-start_time,'convolution layers: ',len(ijc_list), "maxes", np.max(np.abs(ijc_list[:,:2]),axis=0))
                    
#                     lock.acquire()
#                     data_con[i_con] = conv_data
#                     data_dns[i_con] = dnys_data
#                     lock.release()
                    
#                 lock = Lock()
#                 convolution_function = lambda lst:np.zeros_like(lst[:,2])+1
#                 lat_pixel_size= abs(np.nanmean(self.lat[1:,:]-self.lat[:-1,:]))
#                 lon_pixel_size= abs(np.nanmean(self.lon[:,1:]-self.lon[:,:-1]))
#                 self.conv_data = []
#                 self.conv_data_denoise = []
                
#                 sigma_denoise = self.L3_data[4][0]["Slim_denoise"]
#                 if type(sigma_denoise) == type(None): sigma_denoise = [0,0]
                
#                 Processes = []
#                 all_shmm_con = []
#                 all_data_con = []
#                 all_shmm_dns = []
#                 all_data_dns = []
                
#                 Processes = []
#                 if self._is_True_verbose(verbose=verbose,value=2):print(f"convolution list length: {self.convolution_extent_list.shape}, number of windows: {len(self.unq)}\nTotal tasks:{len(self.unq)*self.convolution_extent_list.shape[0]}")
#                 for j,kw in enumerate(self.unq):
#                     #creating the sructure of conv_data
#                     data_con = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[kw].data.shape))
#                     data_dns = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[kw].data.shape))
#                     data_raw = self.L2_data[kw].data.astype(float)
                    
#                     all_data_con.append(data_con) 
#                     all_data_dns.append(data_dns) 
                    
#                     shmm_con, shmm_con_data = gen_shmm(create = True,ndarray = data_con)
#                     shmm_dns, shmm_dns_data = gen_shmm(create = True,ndarray = data_dns)
#                     shmm_raw, shmm_raw_data = gen_shmm(create = True,ndarray = data_raw)
                    
#                     all_shmm_con.append(shmm_con)
#                     all_shmm_dns.append(shmm_dns)
                                    
#                     shmm_raw_data[:] = data_raw[:]
#                     #preparing to convolute diffrent windows
#                     for i,i_con in enumerate(self.convolution_extent_list):
#                         keywords = {
#                             "l_con":i_con ,
#                             "i_con":i     ,
#                             'raw'  :{"name":shmm_raw.name         ,'type':data_raw.dtype,'shape':data_raw.shape},
#                             'con'  :{"name":all_shmm_con[j].name  ,'type':data_con.dtype,'shape':data_con.shape},
#                             'dns'  :{"name":all_shmm_dns[j].name  ,'type':data_dns.dtype,'shape':data_dns.shape},
#                             "convolution_function" : convolution_function,
#                             "lat_pixel_size"       : lat_pixel_size      ,
#                             "lon_pixel_size"       : lon_pixel_size      ,
#                             "sigma_denoise"        : sigma_denoise       ,
#                             }
#                         Processes.append(Process(target=_task_conv_data,kwargs=keywords))
#                         Processes[-1].start()
#                         if self._is_True_verbose(verbose=verbose,value=2):print("started task:",len(Processes))
                        
#                 for i,p in enumerate(Processes):
#                     Processes[i].join()
#                     if self._is_True_verbose(verbose=verbose,value=2):print("joining task",i )
#                 for j,kw in enumerate(self.unq):
#                     all_data_con[j] = np.ndarray(shape = all_data_con[j].shape,buffer=all_shmm_con[j].buf)
#                     all_data_dns[j] = np.ndarray(shape = all_data_dns[j].shape,buffer=all_shmm_dns[j].buf)
#                 self.conv_data = all_data_con
#                 self.conv_data_denoise = all_data_dns
#             elif self._is_True_verbose(verbose,1):
#                 print("The convoluted data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")
#         if True:
#             if (type(self.conv_data)==type(None)
#                     or force_building):
#                 def _task_conv_data(l_con,i_con,
#                                     raw,con,dns,
#                                     convolution_function,lat_pixel_size,lon_pixel_size,sigma_denoise):
                    
#                     shmm_raw,data_raw = gen_shmm(create=False,name=raw["name"],dtype=raw["type"],shape=raw["shape"]) 
#                     shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"]) 
#                     shmm_dns,data_dns = gen_shmm(create=False,name=dns["name"],dtype=dns["type"],shape=dns["shape"]) 
#                     start_time = time.time()
#                     ijc_list = np.array(fst_neigbors(l_con,lon_pixel_size,lat_pixel_size)).astype(int)
#                     ijc_list [:,2]= convolution_function(ijc_list)
#                     conv_data = join_dt(data_raw, ijc_list)
#                     dnys_data = denoise_data(conv_data[0],denoise_sigma=sigma_denoise)
#                     dnys_data[np.isnan(conv_data[0])]=np.nan
#                     if self._is_True_verbose(verbose=verbose,value=2):print("running time for a task",time.time()-start_time,'convolution layers: ',len(ijc_list), "maxes", np.max(np.abs(ijc_list[:,:2]),axis=0))
                    
#                     lock.acquire()
#                     data_con[i_con] = conv_data
#                     data_dns[i_con] = dnys_data
#                     lock.release()
#                 def _task_sig_data(l_con,i_con,
#                                     raw,sig,
#                                     convolution_function,lat_pixel_size,lon_pixel_size,verbose=None):
                    
#                     shmm_raw,data_raw = gen_shmm(create=False,name=raw["name"],dtype=raw["type"],shape=raw["shape"]) 
#                     shmm_sig,data_sig = gen_shmm(create=False,name=sig["name"],dtype=sig["type"],shape=sig["shape"]) 
#                     start_time = time.time()
#                     ijc_list = np.array(fst_neigbors(l_con,lon_pixel_size,lat_pixel_size)).astype(int)
#                     ijc_list [:,2]= convolution_function(ijc_list)
                    
#                     sig_data = join_dt(data_raw, ijc_list)
                    
#                     if self._is_True_verbose(verbose=verbose,value=2): print("running time for a task",time.time()-start_time,'convolution layers: ',len(ijc_list), "maxes", np.max(np.abs(ijc_list[:,:2]),axis=0))
                    
#                     lock.acquire()
#                     data_sig[i_con] = sig_data
#                     lock.release()
                    
#                 if self._is_True_verbose(verbose=verbose,value=2):print(f"convolution list length: {self.convolution_extent_list.shape}, number of windows: {len(self.unq)}\nTotal tasks:{len(self.unq)*self.convolution_extent_list.shape[0]}")
                
#                 lock = Lock()
#                 convolution_function = lambda lst:np.zeros_like(lst[:,2])+1
#                 lat_pixel_size= abs(np.nanmean(self.lat[1:,:]-self.lat[:-1,:]))
#                 lon_pixel_size= abs(np.nanmean(self.lon[:,1:]-self.lon[:,:-1]))
#                 self.conv_data = []
#                 self.conv_data_denoise = []
#                 self.conv_sigmas = []
                
#                 sigma_denoise = self.L3_data[4][0]["Slim_denoise"]
#                 if type(sigma_denoise) == type(None): sigma_denoise = [0,0]
                
#                 sigmas = get_spice_errors(self.L2_path,verbose=True if self.verbose>0 else False)
#                 sigmas = [i**2 for i in sigmas]
#                 Processes = []
#                 all_shmm_sig = []
#                 all_data_sig = []
#                 all_shmm_con = []
#                 all_data_con = []
#                 all_shmm_dns = []
#                 all_data_dns = []
                
#                 if self._is_True_verbose(verbose=verbose,value=2):print(f"convolution list length: {self.convolution_extent_list.shape}, number of windows: {len(self.unq)}\nTotal tasks:{len(self.unq)*self.convolution_extent_list.shape[0]}")
#                 for j,kw in enumerate(self.unq):
#                     #creating the sructure of conv_data
                    
#                     if True: #error part part
#                         data_sig = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[kw].data.shape))*np.nan
#                         data_raw_sig = sigmas[j].astype(float)
                        
#                         all_data_sig.append(data_sig) 
                        
#                         shmm_sig, shmm_sig_data = gen_shmm(create = True,ndarray = data_sig)
#                         shmm_raw_sig, shmm_raw_sig_data = gen_shmm(create = True,ndarray = data_raw_sig)
                        
#                         all_shmm_sig.append(shmm_sig)
#                         shmm_raw_sig_data[:] = data_raw_sig[:]
                    
#                     if True: #convolution part
#                         #creating the sructure of conv_data
#                         data_con = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[kw].data.shape))
#                         data_dns = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[kw].data.shape))
#                         data_raw = self.L2_data[kw].data.astype(float)
                        
#                         all_data_con.append(data_con) 
#                         all_data_dns.append(data_dns) 
                        
#                         shmm_con, shmm_con_data = gen_shmm(create = True,ndarray = data_con)
#                         shmm_dns, shmm_dns_data = gen_shmm(create = True,ndarray = data_dns)
#                         shmm_raw, shmm_raw_data = gen_shmm(create = True,ndarray = data_raw)
                        
#                         all_shmm_con.append(shmm_con)
#                         all_shmm_dns.append(shmm_dns)
                                        
#                         shmm_raw_data[:] = data_raw[:]

#                     for i,i_con in enumerate(self.convolution_extent_list):
#                         if True: #error part part
#                             keywords2 = {
#                                 "l_con":i_con ,
#                                 "i_con":i     ,
#                                 'raw'  :{"name":shmm_raw_sig.name         ,'type':data_raw_sig.dtype,'shape':data_raw_sig.shape},
#                                 'sig'  :{"name":all_shmm_sig[j].name  ,'type':data_sig.dtype,'shape':data_sig.shape},
#                                 "convolution_function" : convolution_function,
#                                 "lat_pixel_size"       : lat_pixel_size      ,
#                                 "lon_pixel_size"       : lon_pixel_size      ,
#                                 "verbose"              : verbose             ,
#                                 }
#                             Processes.append(Process(target=_task_sig_data,kwargs=keywords2))
#                             Processes[-1].start()
#                             if self._is_True_verbose(verbose=verbose,value=2): print("started error task:",i)
                        
#                         if True: #convolution part
#                             keywords = {
#                                 "l_con":i_con ,
#                                 "i_con":i     ,
#                                 'raw'  :{"name":shmm_raw.name         ,'type':data_raw.dtype,'shape':data_raw.shape},
#                                 'con'  :{"name":all_shmm_con[j].name  ,'type':data_con.dtype,'shape':data_con.shape},
#                                 'dns'  :{"name":all_shmm_dns[j].name  ,'type':data_dns.dtype,'shape':data_dns.shape},
#                                 "convolution_function" : convolution_function,
#                                 "lat_pixel_size"       : lat_pixel_size      ,
#                                 "lon_pixel_size"       : lon_pixel_size      ,
#                                 "sigma_denoise"        : sigma_denoise       ,
#                                 }
#                             Processes.append(Process(target=_task_conv_data,kwargs=keywords))
#                             Processes[-1].start()
#                             if self._is_True_verbose(verbose=verbose,value=2):print("started convo task:",i)
                        
#                 for i,p in enumerate(Processes):
#                     Processes[i].join()
#                     if self._is_True_verbose(verbose=verbose,value=2):print("joining task",i )
#                 for j,kw in enumerate(self.unq):
#                     all_data_con[j] = np.ndarray(shape = all_data_con[j].shape,buffer=all_shmm_con[j].buf).copy()
#                     all_data_dns[j] = np.ndarray(shape = all_data_dns[j].shape,buffer=all_shmm_dns[j].buf).copy()
#                     all_data_sig[j] = np.sqrt(np.ndarray(shape = all_data_sig[j].shape,buffer=all_shmm_sig[j].buf)).copy()
                    
#                 self.conv_data = all_data_con.copy()
#                 self.conv_data_denoise = all_data_dns.copy()
#                 self.conv_sigmas = all_data_sig.copy()
#             elif self._is_True_verbose(verbose,1):
#                 print("The convoluted data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")
            
#     def gen_real_L2_data(self,force_building = False,verbose = None):
#         if (force_building or type(self.real_L2_data)==type(None)):
#             self.real_L2_data = []
#             sigma_denoise = self.L3_data[4][0]["Slim_denoise"]
#             if type(sigma_denoise)==type(None):sigma_denoise=[0,0]
            
#             for i,kw in enumerate(self.EXTNAMES):
#                 real_data = np.zeros(self.L2_data[kw].data.shape)   
                
#                 sub_data = self.conv_data[i][0] if type(sigma_denoise)==type(None) else self.conv_data_denoise[i][0]
                
#                 for i_y in range(real_data.shape[2]):
#                     for i_x in range(real_data.shape[3]):
#                         i_con = self.L3_data[3][i][0,i_y,i_x]
#                         if np.isnan(i_con): i_con = self.convolution_extent_list[0]
#                         else: i_con = int(i_con)
#                         i_con = np.where(self.convolution_extent_list == i_con)
                        
#                         real_data[0,:,i_y,i_x] = (
#                             self.conv_data[i][i_con,0,:,i_y,i_x]
                        
#                             ).astype(float)
                        
#                 self.real_L2_data.append(real_data)
#         elif self._is_True_verbose(verbose,1):
#             print("The real_L2_data data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")
    
#     def gen_real_sigmas(self,force_building = False,verbose = None):
#         self.gen_sigmas()
#         if (force_building or type(self.real_sigmas)==type(None)):
#             self.real_sigmas = []
#             for i,kw in enumerate(self.EXTNAMES):
#                 real_sigmas = np.zeros(self.L2_data[kw].data.shape)*np.nan   
#                 # sub_data = self.conv_data[i][0] if type(sigma_denoise)==type(None) else self.conv_data_denoise[i][0]
                
#                 for i_y in range(real_sigmas.shape[2]):
#                     for i_x in range(real_sigmas.shape[3]):
#                         i_con = self.L3_data[3][i][0,i_y,i_x]
#                         if np.isnan(i_con): i_con = self.convolution_extent_list[0]
#                         else: i_con = int(i_con)
#                         i_con = np.where(self.convolution_extent_list == i_con)
                        
#                         real_sigmas[0,:,i_y,i_x] = self.conv_sigmas[i][i_con,0,:,i_y,i_x]
                            
#                 self.real_sigmas.append(real_sigmas.copy())
#         elif self._is_True_verbose(verbose,1):
#             print("The real_L2_data data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")
    
#     def gen_sigmas(self,force_building=False,verbose=None): 
#         if True:
#             if type(self.conv_sigmas)==type(None) or force_building:
#                 convolution_function = lambda lst:np.zeros_like(lst[:,2])+1
#                 lat_pixel_size= abs(np.nanmean(self.lat[1:,:]-self.lat[:-1,:]))
#                 lon_pixel_size= abs(np.nanmean(self.lon[:,1:]-self.lon[:,:-1]))
#                 sigmas = get_spice_errors(self.L2_path,verbose=True if self.verbose>0 else False)
#                 sigmas = [i**2 for i in sigmas]
#                 self.conv_sigmas = []
#                 for j,kw in enumerate(self.unq):
#                     conv_sigmas = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[self.unq[j]].data.shape))
#                     for i in range(self.convolution_extent_list.shape[0]):
#                         if self.convolution_extent_list[i] == 0:
#                             conv_sigmas[i]=sigmas[j].astype(float)
#                         else:
#                             ijc_list = np.array(fst_neigbors(self.convolution_extent_list[i],lon_pixel_size,lat_pixel_size)).astype(int)
#                             ijc_list [:,2]= convolution_function(ijc_list)
#                             conv_sigmas[i]  = np.sqrt(join_dt(sigmas[j].astype(float), ijc_list))
                            
#                     self.conv_sigmas.append(conv_sigmas.copy())
                    
#             elif self.verbose>=1: 
#                 print("The sigma data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")
#         if False:
#             if type(self.conv_sigmas)==type(None) or force_building:
#                 def _task_sig_data(l_con,i_con,
#                                     raw,sig,
#                                     convolution_function,lat_pixel_size,lon_pixel_size,verbose=None):
                    
#                     shmm_raw,data_raw = gen_shmm(create=False,name=raw["name"],dtype=raw["type"],shape=raw["shape"]) 
#                     shmm_sig,data_sig = gen_shmm(create=False,name=sig["name"],dtype=sig["type"],shape=sig["shape"]) 
#                     start_time = time.time()
#                     ijc_list = np.array(fst_neigbors(l_con,lon_pixel_size,lat_pixel_size)).astype(int)
#                     ijc_list [:,2]= convolution_function(ijc_list)
#                     sig_data = join_dt(data_raw, ijc_list)
#                     if self._is_True_verbose(verbose=verbose,value=2): print("running time for a task",time.time()-start_time,'convolution layers: ',len(ijc_list), "maxes", np.max(np.abs(ijc_list[:,:2]),axis=0))
                    
#                     lock.acquire()
#                     data_sig[i_con] = sig_data
#                     lock.release()
                    
#                 lock = Lock()
#                 convolution_function = lambda lst:np.zeros_like(lst[:,2])+1
#                 lat_pixel_size= abs(np.nanmean(self.lat[1:,:]-self.lat[:-1,:]))
#                 lon_pixel_size= abs(np.nanmean(self.lon[:,1:]-self.lon[:,:-1]))
#                 self.conv_sigmas = []
#                 sigmas = get_spice_errors(self.L2_path,verbose=True if self.verbose>0 else False)
#                 sigmas = [i**2 for i in sigmas]
                
#                 sigma_denoise = self.L3_data[4][0]["Slim_denoise"]
                
#                 Processes = []
#                 all_shmm_sig = []
#                 all_data_sig = []
                
#                 Processes = []
#                 print(f"convolution list length: {self.convolution_extent_list.shape}, number of windows: {len(self.unq)}\nTotal tasks:{len(self.unq)*self.convolution_extent_list.shape[0]}")
#                 for j,kw in enumerate(self.unq):
#                     #creating the sructure of conv_data
#                     data_sig = np.zeros((*self.convolution_extent_list.shape,*self.L2_data[kw].data.shape))
#                     data_raw = sigmas[j]
                    
#                     all_data_sig.append(data_sig) 
                    
#                     shmm_sig, shmm_sig_data = gen_shmm(create = True,ndarray = data_sig)
#                     shmm_raw, shmm_raw_data = gen_shmm(create = True,ndarray = data_sig)
                    
#                     all_shmm_sig.append(shmm_sig)
                                    
#                     shmm_raw_data[:] = data_raw[:]
#                     #preparing to convolute diffrent windows
#                     for i,i_con in enumerate(self.convolution_extent_list):
#                         keywords = {
#                             "l_con":i_con ,
#                             "i_con":i     ,
#                             'raw'  :{"name":shmm_raw.name         ,'type':data_raw.dtype,'shape':data_raw.shape},
#                             'sig'  :{"name":all_shmm_sig[j].name  ,'type':data_sig.dtype,'shape':data_sig.shape},
#                             "convolution_function" : convolution_function,
#                             "lat_pixel_size"       : lat_pixel_size      ,
#                             "lon_pixel_size"       : lon_pixel_size      ,
#                             "verbose"              : verbose             ,
#                             }
#                         Processes.append(Process(target=_task_sig_data,kwargs=keywords))
#                         Processes[-1].start()
#                         if self._is_True_verbose(verbose=verbose,value=2): print("started task:",len(Processes))
                        
#                 for i,p in enumerate(Processes):
#                     Processes[i].join()
#                     if self._is_True_verbose(verbose=verbose,value=2): print("joining task",i )
#                 for j,kw in enumerate(self.unq):
#                     all_data_sig[j] = np.sqrt(np.ndarray(shape = all_data_sig[j].shape,buffer=all_shmm_sig[j].buf))
#                 self.conv_sigmas = all_data_sig
#             elif self.verbose>=1: 
#                 print("The sigma data has been built already repeating the process is time consuming!\nif you want to do it anyway set force_building=True ")

#     def get_velocity(self,kwOri=0,ion_order=0,quiet_sun=None):
#         kw = kwOri if type(kwOri) == str else self.unq[kwOri] 
#         i  = kwOri if type(kwOri) == int else (self.unq).index(kwOri) 
#         if type(quiet_sun)==type(None) : quiet_sun = self.quiet_sun
        
#         wvl_data = self.L3_data[0][i][ion_order*3+1][0]
#         if quiet_sun[0,1] < 0: quiet_sun[0,1] = (wvl_data.shape[0])+quiet_sun[0,1]
#         if quiet_sun[1,1] < 0: quiet_sun[1,1] = (wvl_data.shape[1])+quiet_sun[1,1]
#         # print([quiet_sun[1,0],quiet_sun[1,1],quiet_sun[0,0],quiet_sun[0,1]])
#         res = gen_velocity(wvl_data,np.array([quiet_sun[1,0],quiet_sun[1,1],quiet_sun[0,0],quiet_sun[0,1]]),correction=True)
#         # plt.figure()
#         # plt.pcolormesh(res[0],vmin=-40,vmax=40,cmap="seismic")
#         # plt.figure()
#         # plt.hist(res[0])
#         return res[0]
    
#     def apply_conditions(self,kwOri,conditions,quiet_sun = None):
#         kw = kwOri if type(kwOri) == str else self.unq[kwOri] 
#         i  = kwOri if type(kwOri) == int else (self.unq).index(kwOri) 
#         paramlist = self.L3_data[0][i].copy()
        
#         for mode in conditions:
#             if "min_I" in mode:
#                 if ' s' in mode or ' all' in mode:
#                     paramlist[2:-1:3][paramlist[0:-1:3]<conditions[mode]] = np.nan 
#                 if ' x' in mode or ' all' in mode:
#                     paramlist[1:-1:3][paramlist[0:-1:3]<conditions[mode]] = np.nan 
#                 if ' I' in mode or ' all' in mode:
#                     paramlist[0:-1:3][paramlist[0:-1:3]<conditions[mode]] = np.nan 
#             elif "max_v" in mode:
#                 #get_velocity_first
#                 ion_order = mode.find("ion")+4
#                 if ion_order not in ["all","ALL"]:ion_order = int(ion_order)
#                 if type(ion_order) == int:
#                     ion_indices = [ ion_order  for i in range(paramlist.shape[0])]
#                 elif type(ion_order) == str:
#                     if ion_order in ["all","ALL"]:
#                         ion_indices = [ i  for i in range(paramlist.shape[0])]
#                     else: raise ValueError(f"{ion_order} not understood")                
#                 else: raise ValueError(f"{ion_order} not understood")
                
#                 for i_ind in range(paramlist.shape[0]//3):
#                     velocity_data = self.get_velocity(kwOri=kwOri,
#                                                       ion_order=i_ind,
#                                                       quiet_sun=quiet_sun)
                                                                                  
#                     if ' s' in mode or ' all' in mode:
#                         paramlist[i_ind*3+2][0][(np.abs(velocity_data))>conditions[mode]] = np.nan 
#                     if ' x' in mode or ' all' in mode:
#                         paramlist[i_ind*3+1][0][(np.abs(velocity_data))>conditions[mode]] = np.nan 
#                     if ' I' in mode or ' all' in mode:
#                         paramlist[i_ind*3+0][0][(np.abs(velocity_data))>conditions[mode]] = np.nan 
                                    
#             else: raise ValueError(f"{mode} not understood")    
#         return paramlist 
       
#     def plot_window(self,kwOri,conditions={},quiet_sun=None,offset_correction=True,
#                     visualize_saturation=True,
#                     ):
#         kw = kwOri if type(kwOri) == str else self.unq[kwOri] 
#         i  = kwOri if type(kwOri) == int else (self.unq).index(kwOri) 
        
#         window = (self.L2_data[kw].data).astype(float).copy()
#         paramlist = self.L3_data[0][i].copy()
#         paramlist = self.apply_conditions(kwOri=kwOri,conditions=conditions,quiet_sun=quiet_sun).copy()
#         for _i,params in enumerate(paramlist):
#             if np.isnan(params).all():
#                 print(f"found NaNed list in {_i}")
                
#         fig,axis,_,_ = _plot_window_Miho(
#                 spectrum_axis = (self.L2_data[kw].spectral_axis *10**10).astype(float),                    
#                 window        =  window,           
#                 paramlist     =  paramlist,
#                 quentity      =  self.L3_data[2][i],
#                 convlist      =  self.L3_data[3][i],
#                 raster        =  self.L2_data_offsetCorr[kw] if offset_correction else self.L2_data[kw],
#                 suptitle      =  kw,
#                 quite_sun     = ([quiet_sun[1,0],quiet_sun[1,1],quiet_sun[0,0],quiet_sun[0,1]]) if type(quiet_sun)!=type(None) else np.array([0,-1,0,-1]),
#                 visualize_saturation=visualize_saturation)
#         return fig,axis

#     def plot_all_windows(self,conditions={},quiet_sun=None,offset_correction=True):
#         for i,kw in enumerate(self.unq):
#             self.plot_window(kwOri=kw,conditions=conditions,quiet_sun=quiet_sun,offset_correction=offset_correction)
            
#     def _get_random_positions(self,number,window_size = np.array([[0,-1],[0,-1]])):
#         data_shape = self.L2_data[self.unq[0]].data.shape
#         if window_size[0,1] < 0 :window_size[0,1] = data_shape[2] + window_size[0,1] 
#         if window_size[1,1] < 0 :window_size[1,1] = data_shape[3] + window_size[1,1]
#         random_positions = []
#         while True:
#             rand_y = np.random.randint(*window_size[0])
#             rand_x = np.random.randint(*window_size[1])
#             if [rand_y, rand_x] not in random_positions: random_positions.append([rand_y, rand_x]) 
#             if len(random_positions) >= number: break
#         return np.array(random_positions)
        
#     def plot_selected_pixels(self,
#                              positions_list,
#                              kwOri = 0,
#                              fig_axis=None,
#                              plot_convolution_steps=True,):
#         kw = kwOri if type(kwOri) == str else self.unq[kwOri] 
#         i  = kwOri if type(kwOri) == int else (self.unq).index(kwOri)
#         self.gen_convData()
#         self.gen_real_L2_data()
#         self.gen_sigmas()
#         number = len(positions_list)
#         if type(fig_axis)==type(None):
#             ratio = 6
#             n = int(np.sqrt(number))
#             m = int(number//n) + (1 if number%n!=0 else 0)
#             params = {
#                     'legend.fontsize': 10*ratio/4,
#                     # 'figure.figsize': (15, 5),
#                     'axes.labelsize' : 10*ratio/2 ,
#                     'axes.titlesize' : 10*ratio/2 ,
#                     'xtick.labelsize': 10*ratio/2 ,
#                     'ytick.labelsize': 10*ratio/2 ,
#                     'lines.linewidth':0.5*ratio
#                     }
#             plt.rcParams.update(params)

#             fig,axis  = plt.subplots(n,m,figsize=(ratio*m,ratio*n),sharex=True)
#             axis = axis.flatten()
#         else:
#             fig,axis = fig_axis
#         specaxis = (self.L2_data[kw].spectral_axis.value).astype(float)*10**10
#         for j,indices in enumerate(positions_list):
#             specdata = (self.L2_data[kw].data[0,:,indices[0],indices[1]]).astype(float)
#             params = self.L3_data[0][i][:,0,indices[0],indices[1]]
#             conv_level = self.L3_data[3][i][0,indices[0],indices[1]]
#             conv_level = conv_level if not np.isnan(conv_level) else self.convolution_extent_list[0]
#             is_denoised = type(self.L3_data[4][i]["Slim_denoise"])!=type(None)
#             sigma = self.conv_sigmas[i][0][0,:,indices[0],indices[1]]
#             sigma[np.isnan(sigma)]=0
#             if not (np.all(np.isnan(specdata))):
#                 axis[j].errorbar(x=specaxis,y=specdata,yerr=sigma,marker=".",alpha=0.3,color=(0,0.5,0.5))
#             axis[j].step(specaxis,specdata, label="raw",color=(0,0.5,0.5),alpha=1)
            
#             axis[j].plot(specaxis,self.model(specaxis,*params),label="fit{}".format("(lost)" if np.isnan(params).any() else ""))

            
#             high_conv_index = np.where(self.convolution_extent_list==conv_level)[0][0]
#             if conv_level!=0:
#                 for k,conv_ind in enumerate(self.convolution_extent_list):
#                     if conv_ind == 0: continue
#                     if high_conv_index<k:break
#                     # print(plot_convolution_steps , high_conv_index,k,plot_convolution_steps or high_conv_index==k)
#                     if (plot_convolution_steps or high_conv_index==k):
#                         axis[j].step(specaxis,self.conv_data[i][k][0,:,indices[0],indices[1]],
#                                     label=f"conv l:{int(conv_ind):01d}",where="mid",
#                                     )
#                         if high_conv_index==k:
#                             sigma = self.conv_sigmas[i][k][0,:,indices[0],indices[1]]
#                             sigma[np.isnan(sigma)]=0
#                             if not (np.all(np.isnan(specdata))):
#                                 axis[j].errorbar(x=specaxis,
#                                                  y=self.conv_data[i][k][0,:,indices[0],indices[1]],
#                                               yerr=sigma,marker=".",alpha=0.3,color=(0.5,0,0.5),label=f"conv l:{int(conv_ind):01d} error")
#             if is_denoised:
#                 axis[j].step(specaxis,self.conv_data_denoise[i]
#                              [high_conv_index][0,:,indices[0],indices[1]],
#                              label=f"denoised conv l:{int(self.convolution_extent_list[high_conv_index]):01d}",
#                              alpha=1,where="mid",)
                
#             axis[j].legend()  
#             axis[j].set_title(f"px_x{indices[1]},px_y{indices[0]}")
        
        
#         plt.tight_layout()
#         return fig,axis
       
#     def plot_random_pixels(self, 
#         number = 1, 
#         fraction=None, 
#         kwOri = 0,
#         window_size = np.array([[0,-1],[0,-1]]),
#         plot_convolution_steps=True,
#         selected_pixels = None,
#         fig_axis = None,
        
#         ):
#         if True:
#             assert type(number) != type(None) or type(fraction) != type(None)
#             kw = kwOri if type(kwOri) == str else self.unq[kwOri] 
#             i  = kwOri if type(kwOri) == int else (self.unq).index(kwOri) 
            
#             if number == type(None):
#                 number = fraction* self.L3_data[0][0][0][
#                     np.logical_not(np.isnan(self.L3_data[0][0][0]))
#                                                         ]
#             random_numbers = number - (0 if type(selected_pixels)==type(None) else len(selected_pixels))
#             if random_numbers <=0 : 
#                 random_numbers=0
#                 number = len(selected_pixels)
#             # print(random_numbers)
#             random_positions = self._get_random_positions(random_numbers,window_size=window_size)
#             if type(selected_pixels)==type(None): selected_pixels = []
#             all_positions = np.zeros((number,2),dtype=int)
#             if len(selected_pixels)!=0: all_positions[:len(selected_pixels)] = selected_pixels
#             all_positions[len(selected_pixels):] = random_positions
            
#             fig, axis = self.plot_selected_pixels(positions_list=all_positions,
#                                               kwOri=i,plot_convolution_steps=plot_convolution_steps,
#                                               fig_axis = fig_axis,
#                                               ) 
#             return fig,axis,all_positions           
#         if False:
#             assert type(number) != type(None) or type(number) != type(None)
#             kw = kwOri if type(kwOri) == str else self.unq[kwOri] 
#             i  = kwOri if type(kwOri) == int else (self.unq).index(kwOri) 
            
#             if number == type(None):
#                 number = fraction* self.L3_data[0][0][0][
#                     np.logical_not(np.isnan(self.L3_data[0][0][0]))
#                                                         ]
#             random_positions = self._get_random_positions(number,window_size=window_size)
            
#             fig, axis = self.plot_selected_pixels(positions_list=random_positions,
#                                                 kwOri=i,plot_convolution_steps=plot_convolution_steps,
#                                                 fig_axis = fig_axis,
#                                                 )
#         if False:        
            
#             ratio = 3
#             n = int(np.sqrt(number))
#             m = int(number//n) + (1 if number%n!=0 else 0)
#             params = {
#                     'legend.fontsize': 10*ratio/4,
#                     # 'figure.figsize': (15, 5),
#                     'axes.labelsize':  10*ratio/2 ,
#                     'axes.titlesize':  10*ratio/2 ,
#                     'xtick.labelsize': 10*ratio/2 ,
#                     'ytick.labelsize': 10*ratio/2 ,
#                     'lines.linewidth':0.5*ratio
#                     }
#             plt.rcParams.update(params)

#             fig,axis  = plt.subplots(n,m,figsize=(ratio*m,ratio*n),sharex=True)
#             axis = axis.flatten()
            
#             specaxis = (self.L2_data[kw].spectral_axis*10**10).astype(float)
#             for j,indices in enumerate(random_positions):
#                 specdata = (self.L2_data[kw].data[0,:,indices[0],indices[1]]).astype(float)
#                 params = self.L3_data[0][i][:,0,indices[0],indices[1]]
#                 conv_level = self.L3_data[3][i][0,indices[0],indices[1]]
#                 axis[j].step(specaxis,specdata, label="raw")
#                 axis[j].plot(specaxis,self.model(specaxis,*params),label="fit{}".format("(lost)" if np.isnan(params).any() else ""))
#                 if conv_level!=0:
#                     for k,conv_ind in enumerate(self.convolution_extent_list):
#                         if conv_ind == 0: continue
#                         axis[j].plot(specaxis,self.conv_data[i][k][0,:,indices[0],indices[1]],
#                                     label=f"conv l:{int(conv_ind):01d}")            
#                 axis[j].legend()    
#             for j in range(j+1,n*m):
#                 axis[j].remove()
            
#             plt.tight_layout()
    
#     def plot_worse_plots():
#         pass
    
#     def plot_X2(self,axis=None,origine=False,convolution=False,kwOri=0,
#             hist_kwargs= {"alpha":1,"bins":10**np.linspace(-4,2,num=100)},select_pixel=None):
        
#         i,kw = self._getkwANDi_(kwOri)   
#         self.gen_X2()
#         X2_2D = self.X2_2D[i]
#         X2_3D = self.X2_3D[i][0]
        
#         if type(select_pixel)==type(None):
#             ref_X2_2D = X2_2D
#         else:
#             ref_X2_2D = X2_3D[select_pixel]
        
#         if type(axis) == type(None):
#             i = 2
#             if origine: i+=1
#             if convolution: i+=1
#             fig,axis = plt.subplots(1,i,figsize=(7*i+3,7))
#             axis = axis.flatten()
#         ii=2
        
#         try:
#             norm = ImageNormalize(ref_X2_2D,interval = AsymmetricPercentileInterval(1,99),stretch = SqrtStretch())
#             im = axis[0].pcolormesh(ref_X2_2D,norm=norm,cmap="BuPu",rasterized=True)
#         except:
#             im = axis[0].pcolormesh(ref_X2_2D,cmap="BuPu",rasterized=True)
            
#         plt.colorbar(im,ax=axis[0]) 
#         axis[1].hist(X2_2D.flatten(),**hist_kwargs)
#         axis[1].set_yscale('log')
#         axis[1].set_xscale('log')
#         if origine:
#             i,kw = self._getkwANDi_(kwOri)
#             data = self.L2_data[kw].data 
#             if type(select_pixel) == type(None):
#                 data = np.nanmean(data,axis=(0,1))
#             else:
#                 data = data[0,select_pixel]
                
#             norm = ImageNormalize(data,interval = AsymmetricPercentileInterval(1,99),stretch = SqrtStretch())
#             im = axis[ii].pcolormesh(data,norm=norm,cmap="magma",rasterized=True)        
#             plt.colorbar(im,ax=axis[ii]) 
#             ii+=1
            
#         if convolution:
#             i,kw = self._getkwANDi_(kwOri)
#             data = self.L3_data[3][i][0]    
#             # norm = ImageNormalize(data,interval = AsymmetricPercentileInterval(1,99.99),stretch = SqrtStretch())
#             im = axis[ii].pcolormesh(data,cmap="rainbow",rasterized=True)        
#             plt.colorbar(im,ax=axis[ii]) 
#             ii+=1
#         try:return fig,axis
#         except: return axis

#     def _getkwANDi_(self,kwOri):
#         if type(kwOri)==str:
#             kw = kwOri;i=kwOri.index(kw)
#         else:
#             i = kwOri;kw= self.EXTNAMES[i]
#         return i,kw
    
#     def _is_True_verbose(self,verbose=None,value=0):
#         if type(verbose)==type(None):
#             return self.verbose>=value
#         else:
#             return verbose>=value
    
#     def _rotate(self,vx, vy, theta):
#         return (np.cos(theta) * vx - np.sin(theta) * vy,
#                 np.sin(theta) * vx + np.cos(theta) * vy)
    
#     def __getitem__(self, kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         output = []
#         for _dat in self.L3_data:
#             if type(_dat) == list:
#                 if len(_dat) == len(self.EXTNAMES): 
#                     output.append(_dat[i])
                    
#                 else:output.append(_dat)
#             else:output.append(_dat)
            
#         return(output)  
        
#     def get_parameters(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.L3_data[0][i]
    
#     def get_covariance(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.L3_data[1][i]
    
#     def get_quentities(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.L3_data[2][i]
    
#     def get_convolution(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.L3_data[3][i]    
    
#     def get_meta(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.L2_data_astropy[i].header
    
#     def get_WCS(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         meta = self.get_meta(i)
#         meta2 = {}

#         for i,key in enumerate(meta):
#             if (key=="" or " " in key or "\n" in key or "\n" in str(meta[key])):
#                 continue
#             meta2[key] = meta[key]

#         SPICE_WCS = WCS(meta2)
#         SPICE_WCS2 = SPICE_WCS.dropaxis(2) 

#         return SPICE_WCS2
 
#     def get_window_name(self,kwOri = "All"):
#         if kwOri in ["All","all"]:  return self.EXTNAMES
#         else: 
#             i,kw = self._getkwANDi_(kwOri)
#             return kw

# class Fit4Anna():
#     def __init__(self,L3_file_path, verbose = 1):
#         if verbose >= 1:print("Starting the data reading")
#         self.data = p.load(open(L3_file_path,"rb"))
#         self.EXTNAMES = [self.data[4][i]["EXTNAME"] for i in range(len(self.data[4]))] 
    
#     def _getkwANDi_(self,kwOri):
#         if type(kwOri)==str:
#             kw = kwOri;i=kwOri.index(kw)
#         else:
#             i = kwOri;kw= self.EXTNAMES[i]
#         return i,kw
    
#     def __getitem__(self, kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         output = []
#         for _dat in self.data:
#             if type(_dat) == list:
#                 if len(_dat) == len(self.EXTNAMES): 
#                     output.append(_dat[i])
                    
#                 else:output.append(_dat)
#             else:output.append(_dat)
            
#         return(output)  
        

#     def get_parameters(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.data[0][i]
    
#     def get_covariance(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.data[1][i]
    
#     def get_quentities(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.data[2][i]
    
#     def get_convolution(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.data[3][i]    
    
#     def get_meta(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         return self.data[4][i]    
    
#     def get_WCS(self,kwOri):
#         i,kw = self._getkwANDi_(kwOri)
#         meta = self.get_meta(i)
#         meta2 = {}

#         for i,key in enumerate(meta):
#             if (key=="" or " " in key or "\n" in key or "\n" in str(meta[key])):
#                 continue
#             meta2[key] = meta[key]

#         SPICE_WCS = WCS(meta2)
#         SPICE_WCS2 = SPICE_WCS.dropaxis(2) 

#         return SPICE_WCS2
 
#     def get_window_name(self,kwOri = "All"):
#         if kwOri in ["All","all"]:  return self.EXTNAMES
#         else: 
#             i,kw = self._getkwANDi_(kwOri)
#             return kw


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
from analysers import normit, get_coord_mat,suppress_output,FIP_error
from collections.abc import Iterable
from astropy.visualization import SqrtStretch,PowerStretch,LogStretch, AsymmetricPercentileInterval, ImageNormalize, MinMaxInterval, interval,stretch
from sunpy.map import Map
from fiplcr import Line
from fiplcr import LinearComb as lc
from fiplcr import fip_map
import astropy.units as u

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
    if isinstance(hdul_or_path,(str, pathlib.PosixPath, pathlib.WindowsPath,HDUList)):raise ValueError('The hdul_or_path sould be a list of 3')
    for val in hdul_or_path:
      if isinstance(val, (str, pathlib.PosixPath, pathlib.WindowsPath)):
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