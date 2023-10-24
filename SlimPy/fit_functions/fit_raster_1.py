import warnings
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import os 
import pickle
from time import sleep
import multiprocessing as mp
from multiprocessing import Process,Lock
from pathlib import PosixPath, Path

from ..spice_utils.ias_spice_utils import utils as spu #look at this later the path isn't set well yet
from astropy.visualization import SqrtStretch, AsymmetricPercentileInterval, ImageNormalize
from ndcube import NDCollection
from sunraster.instr.spice import read_spice_l2_fits

from ..visualization.graphs import plot_window_Miho,plot_error,_plot_window_Miho
from .fit_window import fit_window,_fit_window
from ..utils.utils import prepare_filenames,gen_shmm,verbose_description
from ..utils.denoise import denoise_raster, denoise_data
from ..utils.error_finder import get_spice_errors

_recent_version = 1
def fit_raster(*args,**kwargs):
    if "version" in kwargs.keys():
        if kwargs["version"] == 0:
            print(f"""
            You asked fit_raster to use old function it's unstable and it will be deprecated soon. 
            The most recent is version {_recent_version} 
            If you are aware you keep it else be prepared for errors (:3)""")
            del kwargs["version"]
            return multi_windows_fit(*args,**kwargs)
        if kwargs["version"] == "v1":
            return _fit_raster(*args,**kwargs)
    else:
        print(f"""
        You asked fit_raster to use old function it's unstable and it will be deprecated soon. 
        The most recent is version {_recent_version} 
        If you are aware you keep it else be prepared for errors (:3) """)
        return multi_windows_fit(*args,**kwargs)
        
    
#becareful this is in  the testing phase
def     _fit_raster(
                path_or_raster          :str or NDCollection                                           ,                                                            
                init_params             :list                                                          ,                                                      
                fit_func                :callable                                                      ,                                                      
                quentities              :list                                                          ,                                                      
                bounds                  :np.array               = np.array([np.nan])                   , 
                window_size             :np.ndarray             = np.array([[500,510],[60,70]])        ,
                convolution_function    :callable               = lambda lst:np.zeros_like(lst[:,2])+1 ,
                convolution_threshold   :float                  = np.array([0.1,10**-4,0.1,100])       ,
                convolution_extent_list :np.array               = np.array([0,1,2,3,4,5])              ,
                mode                    :str                    = "cercle"                             ,
                weights                 :bool                   = True                                 ,
                denoise                 :list or None           = None                                 ,
                clipping_sigma          :float                  = 2.5                                  ,           
                clipping_med_size       :list                   = [6,3,3]                              ,              
                clipping_iterations     :int                    = 3                                    ,                
                counter_percent         :float                  = 10                                   ,
                preclean                :bool                   = True                                 ,
                preadjust               :bool                   = True                                 , 
                save_data               :bool                   = True                                 ,
                save_plot               :bool                   = True                                 ,           
                prefix                  :str                    = None                                 ,
                plot_filename           :str                    = None                                 ,
                data_filename           :str                    = None                                 ,
                quite_sun               :np.array               = np.array([0,-1,0,-1])                ,
                data_save_dir           :str                    = "./.p/"                              ,
                plot_save_dir           :str                    = "./imgs/"                            ,
                plot_kwargs             :dict                   = {}                                   ,
                show_ini_infos          :float                  = True                                 ,
                forced_order            :int                    = None                                 , 
                Jobs                    :dict                   = {"windows":1,"pixels":1}             ,
                verbose                 :int                    = 0                                    ,
                describe_verbose        :bool                   = True                                      
                ):
    
    if True: #assertion
        #assert that jobs is non zero int
        pass
    lock = Lock()
    if verbose<=-2: warnings.filterwarnings("ignore");sleep(5)
    verbose_description(verbose)
    
    
    if True: #checking arguments adequacy 
        if verbose > 0: print("checking adequacy of given parameters")
        if type(quentities[0]) != list: raise ValueError(f"queities sould be a list of lists\n quentities given {quentities}")
        for i in range(len(init_params)):
            if len(init_params[i])!=len(quentities[i]):
                raise ValueError(f'initial parameters provided are not aligned with the quentities\nindex: {i}\ninit parms: {init_params[i]}\nquentities: {quentities}')
        if verbose > 0: print("passed tests: the parameters are initially right")
        
    if True: #reading data
        if verbose>1: print("reading data")
        if type(path_or_raster) in (str,PosixPath):
            if verbose>1: print(f"data is given as path:  {path_or_raster}")
            raster = read_spice_l2_fits(str(path_or_raster))
        else: 
            raise ValueError("In the new analysis function the path is needed ")
            if verbose>1: print(f"data is given as NDCube:{path_or_raster}")
            raster = path_or_raster
        unq = spu.unique_windows(raster)
        KW = [unq[i] for i in range(len(unq))]
        paramlist2 = []
        covlist2 =   []
        quentity2 =  []
        convlist2 =  []
        if verbose>1: print(f"data imported\nWindows:{KW}\nDATE SUN:{raster[KW[0]].meta['DATE_SUN']}")
        
    if True: #preparing the path names for futur saves 
        filename,filename_a,filename_b = prepare_filenames(prefix, data_filename, plot_filename, data_save_dir, plot_save_dir, forced_order,verbose=verbose )
        
        if verbose>1: print("Save files:\n{}\n{}\n{}".format(filename,filename_a,filename_b))
        
    if show_ini_infos and False: #this will be showing initial state of the algotrithm before fitting
        save_info = True
        dir = "./tmp/"
        if not os.path.isdir(dir):
            os.mkdir(dir)
        dir_list = os.listdir(dir); j=0
        for file in dir_list:
            try:
                j2 = int(file[:-4])
                    
                if j2>=j:
                    j=j2+1
            except Exception:
                pass
        file_plot = f'{j:06d}.jpg'
        if verbose >0: 
            print(f"""
                  show_ini_infos is set to True this will show a 
                  plot of the initial stat of tha algorithm.
                  It will save a copy as well in the a tmp dirctory in {dir+file_plot}""")
        n_windows = len(KW)
        m = 3; n= int(n_windows/3 + (1 if n_windows%m!= 0 else 0))
        fig,axis = plt.subplots(n*2,m,figsize = (4*m,6*n))
        axis = axis.flatten()
        A_axis= axis[:(n_windows)+(0 if n_windows%3==0 else 1)]
        B_axis= axis[(n_windows)+(0 if n_windows%3==0 else 1):]
        for i,k in enumerate(KW):
            #Window plotting
            data = np.nanmean(raster[k].data,axis=(0,1))
            atad = np.nanmean(raster[k].data,axis=(0,2,3))
            
            ang_lat = raster[k].celestial.data.lat.deg
            ang_lon = raster[k].celestial.data.lon.deg
            ang_lon[ang_lon<=180] = ang_lon[ang_lon<=180]+360 
            norm = ImageNormalize(data,
                                    interval=AsymmetricPercentileInterval(1, 99),
                                    stretch=SqrtStretch())
            im = A_axis[i].pcolormesh(ang_lon,ang_lat,data,norm=norm,cmap="magma")
            # im = A_axis[i].pcolormesh(data[110:720],norm=norm,cmap="magma")
            B_axis[i].plot((raster[k]).spectral_axis*10**10,atad)
            B_axis[i].set_title(k);A_axis[i].set_title(k)
            A_axis[i].grid();B_axis[i].grid()
            plt.colorbar(im,ax= A_axis[i])
            
            
            #ini params and segmentation
            init_params2 = init_params[i]
            if len(bounds.shape)!=1:
                bounds2 = bounds[i]
            else:
                bounds2 = bounds
            
            B = init_params2[-1] 
            B_axis[i].axhline(B,ls=":",label="BG",color='black')
            for j in range(int(len(init_params2)/3)):
                color = np.random.rand(3)
                color = 0.8 * color/np.sqrt(np.sum(color**2))
                I = init_params2[j*3]; x = init_params2[j*3+1]; s = init_params2[j*3+2] 
                print(j,I,x,s)
                B_axis[i].scatter([x],[I+B],color = color)
                B_axis[i].axvline(x,ls=":",color = color)
                B_axis[i].step([x-s/2,x+s/2],[I*np.exp(-1/2)+B ,I*np.exp(-1/2)+B],color = color,label="line {}".format(j) )
            
            
            
            #window selection
            ws = window_size.copy()
            up_leftx     =  ang_lon[ws[0,1],ws[1,0]]
            up_lefty     =  ang_lat[ws[0,1],ws[1,0]]
            up_rightx    =  ang_lon[ws[0,1],ws[1,1]]
            up_righty    =  ang_lat[ws[0,1],ws[1,1]]
            down_rightx  =  ang_lon[ws[0,0],ws[1,1]]
            down_righty  =  ang_lat[ws[0,0],ws[1,1]]
            down_leftx   =  ang_lon[ws[0,0],ws[1,0]]
            down_lefty   =  ang_lat[ws[0,0],ws[1,0]]
            A_axis[i].plot([up_leftx   ,up_rightx ,down_rightx,down_leftx ,up_leftx],
                           [up_lefty   ,up_righty ,down_righty,down_lefty ,up_lefty],
                           color='green',lw=2,label='window size')
            ws = np.array(
                [
                    quite_sun[2:],
                    quite_sun[:2]
                    ]
                )
            up_leftx     =  ang_lon[ws[0,1],ws[1,0]]
            up_lefty     =  ang_lat[ws[0,1],ws[1,0]]
            up_rightx    =  ang_lon[ws[0,1],ws[1,1]]
            up_righty    =  ang_lat[ws[0,1],ws[1,1]]
            down_rightx  =  ang_lon[ws[0,0],ws[1,1]]
            down_righty  =  ang_lat[ws[0,0],ws[1,1]]
            down_leftx   =  ang_lon[ws[0,0],ws[1,0]]
            down_lefty   =  ang_lat[ws[0,0],ws[1,0]]
            A_axis[i].plot([up_leftx   ,up_rightx ,down_rightx,down_leftx ,up_leftx],
                           [up_lefty   ,up_righty ,down_righty,down_lefty ,up_lefty],
                           color='red',lw=2,label='quite sun reference')
            #A_axis[i].legend()
            B_axis[i].legend()
        fig.suptitle("Pre-analysis info \nStudied window: Green / Quite sun reference: Red")
        plt.tight_layout()
        plt.savefig(dir+file_plot)
        plt.show()
    
    if True: #SHOW TIME! now we have to use a function that analyses the window ver a number of jobs   
        Processes = []
        all_shmm_raw  = [] #list of shared memory pointers for raw data 
        all_shmm_par  = [] #list of shared memory pointers for par data
        all_shmm_cov  = [] #list of shared memory pointers for cov data
        all_shmm_con  = [] #list of shared memory pointers for con data
        
        all_data_par = [] #list of the final results of all the analysis for par data
        all_data_cov = [] #list of the final results of all the analysis for cov data
        all_data_con = [] #list of the final results of all the analysis for con data
        if weights:
            fits_data_wgt = get_spice_errors(path_or_raster)
            all_shmm_wgt = []
            all_data_wgt = []
                             
        is_done = []
        unc_i = 0 #for the uncertainty 
        for i in range(len(KW)):
            kw = KW[i]
            kw2 = kw.replace("/","_")
            kw2 = kw2.replace(" ","")
            window = raster[kw]
            meta=window.meta
            
            ang_lat = raster[kw].celestial.data.lat.arcsec
            ang_lon = raster[kw].celestial.data.lon.arcsec
            ang_lon[ang_lon>180*3600] -= 360*3600 
            ang_lat[ang_lat>180*3600] -= 360*3600 
            lon_pixel_size= raster[kw].meta["CDELT1"]#abs(np.nanmean(ang_lat[1:,:]-ang_lat[:-1,:]))
            lat_pixel_size= raster[kw].meta["CDELT2"]#abs(np.nanmean(ang_lon[:,1:]-ang_lon[:,:-1]))
            if True: #check whether data already exists
                if verbose> -1: print("checking for the file ",filename.format(i,kw2,window.meta["DATE_SUN"]))
                if os.path.isfile(filename.format(i,kw2,window.meta["DATE_SUN"])):
                    if verbose> -1: print("the {} file exists!".format(filename.format(i,kw2,window.meta["DATE_SUN"])))
                    done = True
                else:
                    if verbose> -1: print("the {} file doesn't exists".format(filename.format(i,kw2,window.meta["DATE_SUN"])))
                    done = False
            is_done.append(done)
            if not is_done[i]:
                if True: # preparing arguments
                    x = (window.spectral_axis*10**10).astype(float)
                    sub_init_params = init_params[i]
                    sub_quentities = quentities[i]
                    
                    #preparing output matrices 
                    shape_raw = window.data.shape 
                    data_raw = (window.data).astype(float) 
                    data_par = np.zeros((sub_init_params.shape[0],                         shape_raw[0],shape_raw[2],shape_raw[3]))*np.nan
                    data_cov = np.zeros((sub_init_params.shape[0],sub_init_params.shape[0],shape_raw[0],shape_raw[2],shape_raw[3]))*np.nan
                    data_con = np.zeros((                                                  shape_raw[0],shape_raw[2],shape_raw[3]))*np.nan
                    data_wgt = np.zeros((                                     shape_raw[0],shape_raw[1],shape_raw[2],shape_raw[3]))+1
                    
                    
                    all_data_par.append(data_par.copy())
                    all_data_cov.append(data_cov.copy())
                    all_data_con.append(data_con.copy())
                    
                    
                    shmm_raw, shmm_raw_data = gen_shmm(create = True,ndarray = data_raw)
                    shmm_par, shmm_par_data = gen_shmm(create = True,ndarray = data_par)
                    shmm_cov, shmm_cov_data = gen_shmm(create = True,ndarray = data_cov)
                    shmm_con, shmm_con_data = gen_shmm(create = True,ndarray = data_con)
                    shmm_wgt, shmm_wgt_data = gen_shmm(create = True,ndarray = data_wgt)
                    
                    indx =0
                    # print("data size: ",uncertainty.shape[1])
                    if weights:
                        while True:
                            # print(f"uncertainty {unc_i} size {uncertainties[unc_i].shape[1]}")
                            shmm_wgt_data[:,
                                        indx:indx+fits_data_wgt[unc_i].shape[1],
                                        :,
                                        :] = fits_data_wgt[unc_i] 
                            indx += fits_data_wgt[unc_i].shape[1] 
                            unc_i+=1
                            if indx == shmm_wgt_data.shape[1]:
                                break
                        
                    shmm_raw_data[:] = data_raw[:]
                    shmm_par_data[:] = np.nan
                    shmm_cov_data[:] = np.nan
                    shmm_con_data[:] = np.nan
                    
                    
                    all_shmm_raw.append(shmm_raw)
                    all_shmm_par.append(shmm_par)
                    all_shmm_cov.append(shmm_cov)
                    all_shmm_con.append(shmm_con)
                    
                    
                    if verbose>1: print(f"""
                    fit raster input:
                    shmm_raw: {{'name':{all_shmm_raw[i].name},'type':{data_raw.dtype},'shape':{data_raw.shape} }}
                    fit raster output:
                    shmm_par: {{'name':{all_shmm_par[i].name},'type':{all_data_par[i].dtype},'shape':{all_data_par[i].shape} }}
                    shmm_cov: {{'name':{all_shmm_cov[i].name},'type':{all_data_cov[i].dtype},'shape':{all_data_cov[i].shape} }}
                    shmm_con: {{'name':{all_shmm_con[i].name},'type':{all_data_con[i].dtype},'shape':{all_data_con[i].shape} }}
                    """)
                CTH = convolution_threshold 
                if type(convolution_threshold[0])==np.ndarray: CTH=convolution_threshold[i]
                print(len(all_shmm_par),all_shmm_par, "DELETE")
                keywords ={ "x"                      : x,
                            'raw' :{"name"           : all_shmm_raw[i].name  ,
                                    'type'           : data_raw.dtype,
                                    'shape'          : shape_raw},
                            "quentities"             : sub_quentities,
                            'meta'                   : meta,
                            'init_params'            : sub_init_params,
                            'fit_func'               : fit_func,
                            'lat_pixel_size'         : lat_pixel_size, 
                            'lon_pixel_size'         : lon_pixel_size, 
                            'bounds'                 : bounds if len(bounds.shape)==1 else bounds[i],
                            'window_size'            : window_size,
                            # 'adaptive'             : adaptive, It's always adaptive
                            'convolution_function'   : convolution_function,
                            'convolution_threshold'  : CTH,
                            'convolution_extent_list': convolution_extent_list,
                            "mode"                   : mode,    
                            'weights'                : weights,
                            'denoise'                : denoise,
                            'clipping_sigma'         : clipping_sigma,                 
                            'clipping_med_size'      : clipping_med_size,                
                            'clipping_iterations'    : clipping_iterations,               

                            'counter_percent'        : counter_percent,
                            'preclean'               : preclean,
                            'preadjust'              : preadjust,
                            'njobs'                  : Jobs["pixels"],
                            "verbose"                : verbose,
                            "lock"                   : lock,
                            "describe_verbose"       : False,
                            
                            "par" : {'name':all_shmm_par[i].name,'type':all_data_par[i].dtype,'shape':all_data_par[i].shape},
                            "cov" : {'name':all_shmm_cov[i].name,'type':all_data_cov[i].dtype,'shape':all_data_cov[i].shape},
                            "con" : {'name':all_shmm_con[i].name,'type':all_data_con[i].dtype,'shape':all_data_con[i].shape},
                            "wgt" : {'name':    shmm_wgt   .name,'type':    data_wgt   .dtype,'shape':    data_wgt   .shape} if weights==True else None,
                            
                    }
                Processes.append(Process(target=task_fit_window,kwargs=keywords))
            elif is_done[i]:
                Processes.append(0)
                all_shmm_raw.append(0)
                all_shmm_par.append(0)
                all_shmm_cov.append(0)
                all_shmm_con.append(0)
    
    if True: #Starting analysis
        last_joined_i = -1
        for i,p in enumerate(Processes):
            if not is_done[i]:
                JobNum = (i+1)%Jobs["windows"]
                if verbose>1: print(f"Starting process job { JobNum } on raster fits")
                p.start()
                if ((i+1)%Jobs["windows"]==0 or i+1==len(Processes)): #to run only njobs for windows not all
                    for j in range(last_joined_i+1,i+1):
                        # print("is_done;",is_done[i]," i=",i," j=",j," last_joined_i",last_joined_i)
                        if is_done[j]: continue
                        Processes[j].join()
                        all_data_par[j] = np.ndarray(shape = all_data_par[j].shape,buffer=all_shmm_par[j].buf)  
                        all_data_cov[j] = np.ndarray(shape = all_data_cov[j].shape,buffer=all_shmm_cov[j].buf)  
                        all_data_con[j] = np.ndarray(shape = all_data_con[j].shape,buffer=all_shmm_con[j].buf)
                        
                        if save_data:   
                            kw = KW[j]
                            kw2 = kw.replace("/","_")
                            kw2 = kw2.replace(" ","")  
                            pickle.dump((
                                        all_data_par[j],
                                        all_data_cov[j],
                                        quentities  [j],
                                        all_data_con[j],
                                        dict(raster[KW[j]][0].meta)
                                        ),
                                        open(filename.format(j,kw2,raster[KW[j]][0].meta["DATE_SUN"]),"wb"))
                            #these maxis work only with DYNamics of 2/04
                            maxI = np.array([0.80,2.00,10.0,60.0,25.0,30.0   ])
                            maxB = np.array([0.08,1.50,0.60,2.00,1.50,2.00  ])
                            try:
                                _plot_window_Miho(
                                            (raster[kw].spectral_axis).astype(float)*10**10,
                                            (raster[kw].data).astype(float),
                                            paramlist=all_data_par[j],
                                            quentity =quentities[j],
                                            convlist =all_data_con[j],
                                            suptitle =kw,
                                            window_size=window_size,
                                            quite_sun=quite_sun,
                                            save=save_plot,
                                            filename=filename_a.format(i,kw2,window.meta["DATE_SUN"]),
                                            # min_x=-80,max_x=80,
                                            # min_I=0,max_I=maxI[j],
                                            # min_s=0.3,max_s=0.6,
                                            # min_B=0,max_B=maxB[j],
                                            raster=raster[kw],
                                            visualize_saturation = False)
                            except:pass
                    last_joined_i = i
                    
    if verbose>-1: print(f'The raster is fitted for the Date: {raster[KW[0]][0].meta["DATE_SUN"]}')
    for i in range(len(KW)):#ctype to python getting data in the ight type and shape
        kw = KW[i]
        kw2 = kw.replace("/","_")
        kw2 = kw2.replace(" ","")
        
        if is_done[i]:
                #you should read data here                
                all_data_par[i],
                all_data_cov[i],
                _              ,
                all_data_con[i],
                _              = pickle.load(open(filename.format(i,kw2,raster[KW[i]][0].meta["DATE_SUN"]),"rb"))
                
                
        else:
            pass
    
    metas  =[dict(raster[kw][0].meta) for kw in KW]
    for i in range(len(metas)):
        metas[i]["Slim_denoise"] = denoise
        metas[i]["Slim_error"] = weights
        
    pickle.dump((all_data_par,
                 all_data_cov,
                 quentities,
                 all_data_con,
                 metas
                            ),
                            open(filename.format(len(KW),"all",raster[KW[0]][0].meta["DATE_SUN"]),"wb"))
    for i in range(len(KW)): 
        all_shmm_par[i]
        all_shmm_cov[i]
        all_shmm_con[i]
        all_shmm_raw[i]
    return (np.array(all_data_par).copy(),
            np.array(all_data_cov).copy(),
            quentities,
            np.array(all_data_con).copy(),
            [dict(raster[kw][0].meta) for kw in KW])

def task_fit_window(  
                        x                        :np.ndarray, 
                        raw                      :np.ndarray,
                        meta                     :dict,
                        init_params              :np.ndarray,
                        quentities               :list,
                        fit_func                 :callable,
                        bounds                   :np.ndarray       = np.array([np.nan]),
                        lat_pixel_size           :float            = 1,
                        lon_pixel_size           :float            = 1,
                        window_size              :np.ndarray       = np.array([[210,800],[0,-1]]),
                        # adaptive:bool  True, It's always adaptif
                        convolution_function     :callable         = lambda lst:np.zeros_like(lst[:,2])+1,
                        convolution_threshold    :np.ndarray       = np.array([0.1,10**-4,0.1,100]),
                        convolution_extent_list  :np.array         = np.array([0,1,2,3,4,5,6,7,8,9,10]),
                        mode                     :str              = "cercle",
                        weights                  :str              = None,
                        denoise                  :list             = None,
                        clipping_sigma           :float            = 2.5 ,                                         
                        clipping_med_size        :list             = [6,3,3],                                         
                        clipping_iterations      :int              = 3,  
                        counter_percent          :float            = 10,
                        preclean                 :bool             = True,
                        preadjust                :bool             = True,
                        njobs                    :int              = 1,
                        verbose                  :int              = 0,
                        describe_verbose         :bool             = True,
                        lock                                       = None,
                        
                        par                      :dict             = None,
                        cov                      :dict             = None,
                        con                      :dict             = None,
                        wgt                      :dict             = None,
                   ):
        
         
        _fit_window(x=x,
                    WindowOrShmm = raw,
                    init_params = init_params,
                    quentities=quentities,
                    fit_func = fit_func,
                    bounds=bounds,
                    lat_pixel_size=lat_pixel_size,
                    lon_pixel_size=lon_pixel_size,
                    window_size=window_size,
                    meta=meta,
                    #  adaptive = adaptive, It's always adaptive
                    convolution_function = convolution_function,
                    convolution_threshold=convolution_threshold,
                    convolution_extent_list = convolution_extent_list,
                    mode                    = mode,
                    weights=weights,
                    denoise=denoise,
                    counter_percent=counter_percent,
                    preclean=preclean,
                    preadjust=preadjust,
                    njobs=njobs,
                    verbose=verbose,
                    describe_verbose=describe_verbose,
                    lock = None,
                                 
                    par = par,
                    cov = cov,
                    con = con,
                    wgt = wgt,
                                )    
          
def multi_windows_fit(raster,
                     init_params,
                     fit_func,
                     bounds=np.array([np.nan]),
                     segmentation:np.ndarray = np.array([0,np.inf]),
                     window_size:np.ndarray = np.array([[500,510],[60,70]]),
                     adaptive:bool = True, 
                     convolution_function :callable   = lambda lst:np.zeros_like(lst[:,2])+1,
                     convolution_threshold:float      = np.array([0.1,10**-4,0.1,100]),
                     convolution_extent_list:np.array = np.array([0,1,2,3,4,5]),
                     mode                   :str      = "cercle",
                     weights:str = None,
                     counter_percent:float = 10,
                     preclean:bool=True,
                     preadjust:bool = True, 
                     save_data=True,
                     save_plot=True,           
                     prefix = None,
                     plot_filename=None,
                     quite_sun= np.array([0,-1,0,-1]),
                     min_vel= -100,
                     data_save_dir = "./.p/"  ,
                     plot_save_dir = "./imgs/" ,
                     max_vel= 100,
                     show_ini_infos = True,
                     i=None):
    a ="""The documentation isn't right yet

    Args:
        raster (_type_): _description_
        init_params (_type_): _description_
        fit_func (_type_): _description_
        bounds (_type_, optional): _description_. Defaults to np.array([np.nan]).
        counter_percent (int, optional): _description_. Defaults to 10.
        preclean (bool, optional): _description_. Defaults to True.
        preadjust (bool, optional): _description_. Defaults to True.
        prefix (str, optional): _description_. Defaults to "./.p/01_".

    Returns:
        _type_: _description_
        
        
        x:np.ndarray,
        window:np.ndarray,
        init_params:np.ndarray,
        fit_func:callable,
        bounds:np.ndarray=np.array([np.nan]),
        segmentation:np.ndarray = np.array([0,np.inf]),
        window_size:np.ndarray = np.array([[210,800],[0,-1]]),
        adaptive:bool = True,
        convolution_function :callable   = lambda lst:np.zeros_like(lst[:,2])+1,
        convolution_threshold:float      = 1.,
        convolution_extent_list:np.array = np.array([0,1,2,3,4,5]),
        weights:str = None,
        counter_percent:float = 10,
        preclean:bool=True,
        preadjust:bool = True
        )->[np.ndarray,np.ndarray,np.ndarray,np.ndarray]
    """
    
    unq = spu.unique_windows(raster)
    KW = [unq[i] for i in range(len(unq))]
    paramlist2 = []
    covlist2 = []
    quentity2 = []
    convlist2 = []
    
    
    if type(prefix)==str:
        
        filename = prefix+"window_{:03d}_"+"{:}.p"
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
        print("working with file with prefix i={:03d} ".format(j))
        #------------------------
        filename_a = dir + "{:03d}_".format(j)+"plot_{:03d}_"+"{}_{}.jpg"
        filename_b = dir + "{:03d}_".format(j)+"hits_{:03d}_"+"{}_{}.jpg"
        filename = dir2+"{:03d}_".format(j)+"window_{:03d}_"+"{}_{}.p"
    
    if show_ini_infos:
        n_windows = len(KW)
        m = 3; n= int(n_windows/3 + (1 if n_windows%m!= 0 else 0))
        fig,axis = plt.subplots(n*2,m,figsize = (4*m,6*n))
        axis = axis.flatten()
        A_axis= axis[:n_windows]
        B_axis= axis[n_windows:]
        for i,k in enumerate(KW):
            #Window plotting
            data = np.nanmean(raster[k].data,axis=(0,1))
            atad = np.nanmean(raster[k].data,axis=(0,2,3))
            
            ang_lat = raster[k].celestial.data.lat.deg
            ang_lon = raster[k].celestial.data.lon.deg
            ang_lon[ang_lon<=180] = ang_lon[ang_lon<=180]+360 
            norm = ImageNormalize(data,
                                    interval=AsymmetricPercentileInterval(1, 99),
                                    stretch=SqrtStretch())
            im = A_axis[i].pcolormesh(ang_lon,ang_lat,data,norm=norm,cmap="magma")
            # im = A_axis[i].pcolormesh(data[110:720],norm=norm,cmap="magma")
            B_axis[i].plot((raster[k]).spectral_axis*10**10,atad)
            B_axis[i].set_title(k);A_axis[i].set_title(k)
            A_axis[i].grid();B_axis[i].grid()
            plt.colorbar(im,ax= A_axis[i])
            
            
            #ini params and segmentation
            init_params2 = init_params[i]
            if len(bounds.shape)!=1:
                bounds2 = bounds[i]
            else:
                bounds2 = bounds
            if type(segmentation[0]) not in [np.ndarray,list]:
                segmentation2 = segmentation
            else:
                segmentation2 = segmentation[i]
            if len(segmentation2.shape) == 1:
                 segmentation2 = np.array([segmentation2])
            
            for seg in segmentation2:
                color = np.random.rand(3)
                color = 0.8 * color/np.sqrt(np.sum(color**2))
                B_axis[i].axvspan(seg[0], seg[1], alpha=.3,color = color)
            
            B = init_params2[-1] 
            B_axis[i].axhline(B,ls=":",label="BG",color='black')
            for j in range(int(len(init_params2)/3)):
                color = np.random.rand(3)
                color = 0.8 * color/np.sqrt(np.sum(color**2))
                I = init_params2[j]; x = init_params2[j+1]; s = init_params2[j+2] 
                B_axis[i].scatter([x],[I+B],color = color)
                B_axis[i].axvline(x,ls=":",color = color)
                B_axis[i].plot([x-s/2,x+s/2],[I*np.exp(-1/2)+B ,I*np.exp(-1/2)+B],color = color,label="line {}".format(j) )
            
            
            
            #window selection
            ws = window_size.copy()
            up_leftx     =  ang_lon[ws[0,1],ws[1,0]]
            up_lefty     =  ang_lat[ws[0,1],ws[1,0]]
            up_rightx    =  ang_lon[ws[0,1],ws[1,1]]
            up_righty    =  ang_lat[ws[0,1],ws[1,1]]
            down_rightx  =  ang_lon[ws[0,0],ws[1,1]]
            down_righty  =  ang_lat[ws[0,0],ws[1,1]]
            down_leftx   =  ang_lon[ws[0,0],ws[1,0]]
            down_lefty   =  ang_lat[ws[0,0],ws[1,0]]
            A_axis[i].plot([up_leftx   ,up_rightx ,down_rightx,down_leftx ,up_leftx],
                           [up_lefty   ,up_righty ,down_righty,down_lefty ,up_lefty],
                           color='green',lw=2,label='window size')
            ws = np.array(
                [
                    quite_sun[2:],
                    quite_sun[:2]
                    ]
                )
            up_leftx     =  ang_lon[ws[0,1],ws[1,0]]
            up_lefty     =  ang_lat[ws[0,1],ws[1,0]]
            up_rightx    =  ang_lon[ws[0,1],ws[1,1]]
            up_righty    =  ang_lat[ws[0,1],ws[1,1]]
            down_rightx  =  ang_lon[ws[0,0],ws[1,1]]
            down_righty  =  ang_lat[ws[0,0],ws[1,1]]
            down_leftx   =  ang_lon[ws[0,0],ws[1,0]]
            down_lefty   =  ang_lat[ws[0,0],ws[1,0]]
            A_axis[i].plot([up_leftx   ,up_rightx ,down_rightx,down_leftx ,up_leftx],
                           [up_lefty   ,up_righty ,down_righty,down_lefty ,up_lefty],
                           color='red',lw=2,label='quite sun reference')
            #A_axis[i].legend()
            B_axis[i].legend()
        fig.suptitle("Pre-analysis info \nStudied window: Green / Quite sun rference: Red")
        plt.tight_layout()
        plt.show()
    
    
              
    for i in range(len(KW)):
        # if i!=4: continue
        kw = KW[i]
        kw2 = kw.replace("/","_")
        kw2 = kw2.replace(" ","")
        window = raster[kw]
        print(kw)
        
        data = window.data 
        x = window.spectral_axis
        
        print("checking for the file ",filename.format(i,kw2,window.meta["DATE_SUN"]))
        if os.path.isfile(filename.format(i,kw2,window.meta["DATE_SUN"])):
            print("the {} file exists!".format(filename.format(i,kw2,window.meta["DATE_SUN"])))
            done = True
        else:
            print("the {} file doesn't exists".format(filename.format(i,kw2,window.meta["DATE_SUN"])))
            done = False
        
        
        ang_lat = window.celestial.data[window_size[0,0]:window_size[0,1],
                                        window_size[1,0]:window_size[1,1]].lat.arcsec
        ang_lon = window.celestial.data[window_size[0,0]:window_size[0,1],
                                        window_size[1,0]:window_size[1,1]].lon.arcsec
        ang_lon[ang_lon>180*3600] -= 360*3600 
        ang_lat[ang_lat>180*3600] -= 360*3600 
        lat_pixel_size= abs(np.nanmean(ang_lat[1:,:]-ang_lat[:-1,:]))
        lon_pixel_size= abs(np.nanmean(ang_lon[:,1:]-ang_lon[:,:-1]))
        init_params2 = init_params[i]
        if len(bounds.shape)!=1:
            bounds2 = bounds[i]
        else:
            bounds2 = bounds
        if type(segmentation[0]) not in [np.ndarray,list]:
            segmentation2 = segmentation
        else:
            segmentation2 = segmentation[i]
        
        
        if not done:  
            paramlist, covlist, quentity, convlist = fit_window(
                    x                              = np.array(x*10**10).astype('float32'),
                    window                         = np.array(data).astype('float32'),
                    init_params                    = init_params2,
                    counter_percent                = counter_percent,
                    bounds                         = bounds2,
                    fit_func                       = fit_func,
                    segmentation                   = segmentation2,
                    window_size                    = window_size,
                    adaptive                       = adaptive,
                    convolution_function           = convolution_function,
                    convolution_threshold          = convolution_threshold,
                    convolution_extent_list        = convolution_extent_list,
                    mode                           = mode,
                    weights                        = weights,
                    preclean                       = preclean,
                    preadjust                      = preadjust
                    )
            
            if save_data:     
                pickle.dump((paramlist, 
                             covlist, 
                             quentity, 
                             convlist,
                             window.meta["DATE_SUN"]),
                            open(filename.format(i,kw2,window.meta["DATE_SUN"]),"wb"))
        
        else:
            paramlist, covlist, quentity,convlist,window.meta["DATE_SUN"] = pickle.load(open(filename.format(i,kw2,window.meta["DATE_SUN"]),"rb"))
             
        paramlist2.append(paramlist.copy())
        covlist2.append(covlist.copy())
        quentity2.append(quentity.copy())
        convlist2.append(convlist.copy())
        
        if not done:
            #these maxis work only with DYNamics of 2/04
            maxI = np.array([0.80,2.00,10.0,60.0,25.0,30.0   ])
            maxB = np.array([0.08,1.50,0.60,2.00,1.50,2.00  ])
            plot_window_Miho(x*10**10,
                        data,
                        paramlist=paramlist,
                        quentity=quentity,
                        convlist=convlist,
                        suptitle=kw,
                        window_size=window_size,
                        segmentation=segmentation2,
                        quite_sun=quite_sun,
                        save=save_plot,
                        filename=filename_a.format(i,kw2,window.meta["DATE_SUN"]),
                        # min_x=-80,max_x=80,
                        # min_I=0,max_I=maxI[i],
                        # min_s=0.3,max_s=0.6,
                        # min_B=0,max_B=maxB[i],
                        raster=raster[kw],
                        visualize_saturation = False)
            plot_error(
                covlist = covlist,
                paramlist = paramlist,
                quentity = quentity,
                save=save_plot,
                filename=filename_b.format(i,kw2,window.meta["DATE_SUN"])
                )
    if save_data:     
            pickle.dump((paramlist2, covlist2, quentity2, convlist2,window.meta["DATE_SUN"]),open(filename.format(len(KW),"all",window.meta["DATE_SUN"]),"wb"))
    return paramlist2, covlist2, quentity2, convlist2 

