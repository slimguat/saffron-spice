import datetime
from time import sleep
from SlimPy.utils.utils import round_up
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os 

from ..utils import fst_neigbors,join_dt,Preclean,gen_shmm,verbose_description
from .fit_pixel import fit_pixel as fit_pixel_multi

import multiprocessing as mp 
from multiprocessing import Process,Lock
from multiprocessing.shared_memory import SharedMemory

_recent_version = 1

def fit_window(*args,**kwargs,):
    #this part will adapt to select the old and the new fast format in the future
    #for now it works with the old algorithm
    if "version" in kwargs.keys():
        if kwargs["version"] == 0:
            print(f"""
            You asked fit_raster to use old function it's unstable and it will be deprecated soon. 
            The most recent is version {_recent_version} 
            If you are aware you keep it else be prepared for errors (:3)""")
            del kwargs["version"]
            return fit_window_multi(*args,**kwargs)
        if kwargs["version"] == "v1":
            return _fit_window(*args,**kwargs)
    else:
        print(f"""
        You asked fit_raster to use old function it's unstable and it will be deprecated soon. 
        The most recent is version {_recent_version} 
        If you are aware you keep it else be prepared for errors (:3) """)
        return fit_window_multi(*args,**kwargs)

  
     
def _fit_window(x:np.ndarray,
                WindowOrShmm:np.ndarray or dict("sharedMempory"),
                init_params:np.ndarray,
                quentities:list[str],
                fit_func:callable,
                bounds:np.ndarray=np.array([np.nan]),
                # segmentation:np.ndarray = np.array([0,np.inf]),
                window_size:np.ndarray = np.array([[210,800],[0,-1]]),
                meta=None,
                lon_pixel_size= 1,
                lat_pixel_size= 1,
                #  adaptive:bool = True, It's always adaptif
                convolution_function :callable   = lambda lst:np.zeros_like(lst[:,2])+1,
                convolution_threshold:np.ndarray      = np.array([0.1,10**-4,0.1,100]),
                convolution_extent_list:np.array = np.array([0,1,2,3,4,5,6,7,8,9,10]),
                weights:str = None,
                counter_percent:float = 10,
                preclean:bool=True,
                preadjust:bool = True,
                njobs = 1,
                verbose=0,
                describe_verbose=True,
                lock = None,
                
                par = None,
                cov = None,
                con = None,
                     )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray] or None:
    """Fitting a whole window of dimension (n_t x n_lmb x n_y x n_x) and        creating a map of fitting parameters

        Args:
            x (np.ndarray): spectrum_axis of dimension (n_lmb x 1)
            window (np.ndarray): window data cube of dimension (n_t x n_lmb x n_y x n_x)
            init_params (np.ndarray): parameters list as an array of dimension ((number_of_gaussians * 3 + 1) x 1)
            fit_func (collable): the fitting function to use 
            bounds (np.ndarray, optional): boundary list that contains boundaries of everyparameter. Defaults to np.array([np.nan]).
            segmentation (np.ndarray, optional): if the window is needed to be segmented in multiple windows or smaller one. Defaults to np.array([0,np.inf]).
            window_size (np.ndarray, optional): if the window size is needed to be smaller. Defaults to np.array([[210,800],[0,-1]]).
            window_size (metaObject|dict, optional): meta data 
            adaptive (bool, optional): if we want the window fitting to be convoluting. Defaults to True.
            convolution_function (collable, optional): function to convolute in case the convolution is needed. Defaults to lambda lst:np.zeros_like(lst[:,2])+1.
            convolution_threshold (float, optional):the minimu relative error of the parameters else the pixel will be convoluted to the next level. Defaults to 1.
            convolution_extent_list (np.array, optional): the list of levels of convolution. Defaults to np.array([0,1,2,3,4,5]).
            weights (str, optional): string ["I": for a linear weight depend on the value of intensity]. Defaults to None.
            counter_percent (int, optional): percentile counter for visulization. Defaults to 10.
            preclean (bool, optional): True to clean data from saturated pixels and negative values. Defaults to True.
            preadjust (bool, optional): first fit of the whole window in order to predict positions. Defaults to True.

        Returns:
            paramlist,(np.ndarry): data fit map of dimension (shapeinitParam x n_t x n_y x n_x)
            covlist  ,(np.ndarry): covariance matrix of dimension (? x ? x n_t x n_y x n_x)
            quentity ,(np.ndarry): the array of the first dimension content of paramlist |I => intensity| x=> peak position| s=> sigma| 
            convlist ,(np.ndarry): convolution positions levels for every pixel of dimension (n_x x n_y)

        """
    if lock == None:
        lock = Lock()
    # lock.acquire()
    if verbose<=-2: warnings.filterwarnings("ignore")
    if describe_verbose: verbose_description(verbose);sleep(5)
    if verbose >0: print(f"The window fit is called for {meta['DATE_SUN']}")    
    
    if type(None) in [type(par),type(cov),type(con)]:return_data = True
    else:return_data = False
    
    if verbose >1:print(f"The output shared memory have {'not' if return_data else ''} been given")    

    if True: #assertion part 
        assert (counter_percent<=100 and counter_percent>=0)
        
    
    if True:#handling input data
        if type(WindowOrShmm) == dict:
            shmm_raw,data_raw = gen_shmm(create=False,
                                    name =WindowOrShmm["name"], 
                                    dtype =WindowOrShmm["type"], 
                                    shape =WindowOrShmm["shape"] )
        else: 
            shmm_raw,data_raw = gen_shmm(create=True,
                                        ndarray=WindowOrShmm)
        if verbose>1: print(f"""
                    fit Window input:
                    shmm_raw: {{'name':{shmm_raw.name},'type':{data_raw.dtype},'shape':{data_raw.shape} }}
                    """)
    if True: #making sure that the Window size array is integer like and contains rather index than relative ones 
        ws = (window_size.copy()).astype(int)
        for i in range(ws.shape[0]):
            for j in range(ws.shape[1]):
                if ws[i,j] <0: ws[i,j] = data_raw[0,0].shape[i]+1+ws[i,j]
        if verbose > 1: print(f"the window_size have been checked\n{ws}\n{data_raw[0,0].shape}")
    
    dshape  = np.array([data_raw.shape[0],
                        data_raw.shape[2],
                        data_raw.shape[3],
                       ])
    
    if True: #prepring the counter (To be adapted with the new pipeline)
        _counter = 0
        _imsize = dshape[0]*dshape[1]*dshape[2]
        _next = 0
    
    if preclean: window = Preclean(data_raw.copy()) #clean data from all saturations
     
    if True: #creating pixel convolution matrices (always adaptive) 
        
        # if len(np.array(segmentation).shape) != 1:
            # raise Exception("""
            # The new pipeline doesn't support multiple segments anymore
            # the segmentation should be one list""")
        
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
        if verbose>=2: print('convolution list created')

    # now the data_conv is a window that has an additional convolution ax
    shmm_data_conv, data_conv = gen_shmm(create = True, ndarray=conv_data)
    # del conv_data
    war = {"name":shmm_data_conv.name,"type":data_conv.dtype,"shape":data_conv.shape}
    if verbose>1: print(f"""
                    fit window input:
                    shmm_war: {{'name':{war["name"]},'type':{war["type"]},'shape':{war["shape"]} }}
                    """)
    
    if preadjust:# mke sure t review this after correcting pixel_fit
        fig = plt.figure()
        init_params2,var =fit_pixel_multi(x,
                                            np.nanmean(window[:,
                                                            :,
                                                            ws[0,0]:ws[0,1],
                                                            ws[1,0]:ws[1,1]]
                                                    ,axis=(0,2,3)),
                                            init_params,
                                            quentities=quentities,
                                            fit_func=fit_func,plotit=False,
                                            plot_title_prefix = "preadjust",
                                            weights=weights,verbose=verbose)
        if verbose>=2: print(f"parameters were Preadjusted\nOriginal: {init_params}\nAdjusted:{init_params2}\n,",var)
        
        if not((np.isnan(init_params2)).any()):
            init_params = init_params2
            if verbose>=0: print('Preadjust was successful')
        else:
            if verbose>=0: print('Preadjust was not successful')
            
        
        dtime = str(datetime.datetime.now())[:19].replace(":","-")
        plt.savefig(f"./tmp/preadjust_{dtime}.jpg")
    if True: #generating pixel index positions for linear treatement 
        def index_list(njobs, ws=ws,verbose=verbose):
            index_list = np.zeros(((ws[0,1] - ws[0,0]) * 
                                   (ws[1,1] - ws[1,0]) , 
                                   2),dtype=int
                                  )
            inc = 0
            for i in range(ws[0,0],ws[0,1]):
                for j in range(ws[1,0],ws[1,1]):
                    index_list[inc] = [i,j]
                    inc+=1
            Job_index_list = []
            nPerJob = (len(index_list)//njobs)
            reste = len(index_list)%njobs
            if verbose >=2: print("N pixels per job",nPerJob)
            if verbose >=2: print("reste for first job",reste)
            for i in range(njobs):#distributing pixels over jobs
                Job_index_list.append(index_list[
                    i*nPerJob+(reste if i!=0 else 0):
                    min((i+1)*nPerJob +reste,len(index_list))
                                                ])
            return Job_index_list
        Job_index_list = index_list(njobs,ws=ws,verbose=verbose)
        if verbose>=2:
            for i,j in enumerate(Job_index_list):
                print(i,len(j)) 
                
    if True: #handeling output matrix
        if return_data:
            data_par  = np.zeros((init_params.shape[0],                     *dshape))* np.nan
            data_cov  = np.zeros((init_params.shape[0],init_params.shape[0],*dshape))* np.nan
            data_con  = np.zeros(                                            dshape )* np.nan
            
            shmm_par,data_par = gen_shmm(create=True,ndarray=data_par) 
            shmm_cov,data_cov = gen_shmm(create=True,ndarray=data_cov) 
            shmm_con,data_con = gen_shmm(create=True,ndarray=data_con) 
            par = {"name":shmm_par.name ,"type":data_par.dtype ,"shape":data_par.shape }
            cov = {"name":shmm_cov.name ,"type":data_cov.dtype ,"shape":data_cov.shape }
            con = {"name":shmm_con.name ,"type":data_con.dtype ,"shape":data_con.shape }
        if False:
            
            shmm_par,data_par = gen_shmm(create=False,name=par["name"],dtype=par["type"],shape=par["shape"]) 
            shmm_cov,data_cov = gen_shmm(create=False,name=cov["name"],dtype=cov["type"],shape=cov["shape"])
            shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"])
            
            
        if verbose>1: print(f"""
                    fit window output:
                    shmm_par: {{'name':{par["name"]},'type':{par["type"]},'shape':{par["shape"]} }}
                    shmm_cov: {{'name':{cov["name"]},'type':{cov["type"]},'shape':{cov["shape"]} }}
                    shmm_con: {{'name':{con["name"]},'type':{con["type"]},'shape':{con["shape"]} }}
                    """)

    Processes =[]
    for i in range(njobs): #preparing processes:
        keywords = {
                "x"                      : x,
                "list_indeces"           : Job_index_list[i],
                "war"                    : war,
                "par"                    : par,
                "cov"                    : cov,
                "con"                    : con,
                "ini_params"             : init_params,
                "quentities"             : quentities,
                "fit_func"               : fit_func,
                "bounds"                 : bounds,
                "weights"                : weights,
                "convolution_threshold"  : convolution_threshold,
                "convolution_extent_list": convolution_extent_list,
                "verbose"                : verbose,
                "describe_verbose"       : False,
                "lock"                   : lock,
        }
                
        Processes.append(Process(target=task_fit_pixel,kwargs=keywords))
        Processes[i].start()
        if verbose>=1: print(f"Starting process job: {i+1:02d} on raster fits\nJob list contains: {len(Job_index_list[i])} pixel")
    for i in range(njobs): #join all processes
        Processes[i].join()
        
    if return_data: #returning if par/co/con haven't been given
        shmm_par,data_par = gen_shmm(create=False,name=par["name"],dtype=par["type"],shape=par["shape"]) 
        shmm_cov,data_cov = gen_shmm(create=False,name=cov["name"],dtype=cov["type"],shape=cov["shape"]) 
        shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"]) 
        return (
            data_par.copy(),
            data_cov.copy(),
            data_con.copy())
    else: 
        shmm_par,data_par = gen_shmm(create=False,name=par["name"],dtype=par["type"],shape=par["shape"]) 
        shmm_cov,data_cov = gen_shmm(create=False,name=cov["name"],dtype=cov["type"],shape=cov["shape"])
        shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"])
        
        
def task_fit_pixel(x:np.ndarray,
                    list_indeces:np.ndarray,
                    war: dict,
                    par: dict,
                    cov: dict,
                    con: dict,
                    ini_params:np.ndarray,
                    quentities: list[str],
                    fit_func:callable,
                    bounds:np.ndarray=[np.nan],
                    weights: str = None,
                    convolution_threshold = None,
                    convolution_extent_list = None, 
                    verbose=0,
                    lock=None,
                    describe_verbose = False,
                    **kwargs
):
    
    shmm_war,data_war = gen_shmm(create=False,name=war["name"],dtype=war["type"],shape=war["shape"]) 
    shmm_par,data_par = gen_shmm(create=False,name=par["name"],dtype=par["type"],shape=par["shape"]) 
    shmm_cov,data_cov = gen_shmm(create=False,name=cov["name"],dtype=cov["type"],shape=cov["shape"]) 
    shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"]) 
    for index in list_indeces:
        i_y,i_x = index
        if verbose>=2: print(f"fitting pixel [{i_y},{i_x}]")
        i_ad = -1   
        best_cov = np.zeros((*ini_params.shape,*ini_params.shape)) * np.nan
        best_par = np.zeros((*ini_params.shape,)                 ) * np.nan
        if verbose>2: print(f"y data: {data_war[i_ad,0,:,i_y,i_x]}")
        
        while True: #this will break only when the convolution threshold is met or reached max allowed convolutions
            i_ad +=1                                        #                 |
            if i_ad == len(convolution_extent_list): break  #<----------------'
            
            last_par,last_cov = fit_pixel_multi(x =x,
                                                y=data_war[i_ad,
                                                            0,
                                                            :,
                                                            i_y,
                                                            i_x].copy(),
                                                ini_params=ini_params,
                                                quentities=quentities,
                                                fit_func=fit_func,
                                                bounds=bounds,
                                                weights=weights,
                                                verbose=verbose,
                                                describe_verbose=describe_verbose,
                                                plot_title_prefix=f"{i_y:04d},{i_x:03d},{i_ad:03d}",
                                                )
            if np.isnan(last_par).all():
                best_con = convolution_extent_list[i_ad]
            elif ((np.sqrt(np.diag(last_cov)))/last_par < convolution_threshold).all():
                best_cov   = last_cov
                best_par   = last_par
                best_con   = convolution_extent_list[i_ad]
                break
            else:
                if (np.isnan(best_par)).all():
                    best_cov   = last_cov
                    best_par   = last_par
                    best_con   = convolution_extent_list[i_ad]
                elif np.nansum(
                        (np.sqrt(np.diag(last_cov)))/last_par/convolution_threshold
                    )<np.nansum(
                        (np.sqrt(np.diag(best_cov)))/best_par/convolution_threshold
                                                                                    ):
                    best_cov = last_cov
                    best_par = last_par
                    best_con = convolution_extent_list[i_ad]
        if verbose>=2 : print(f"best_par: {best_par}\nbest_con: {best_con}")
        
        lock.acquire()
        data_par[  :,0,i_y,i_x] = best_par #the result UUUUUUgh finally it's here every pixel will be here
        data_cov[:,:,0,i_y,i_x] = best_cov #the result UUUUUUgh finally it's here every pixel will be here
        data_con[    0,i_y,i_x] = best_con #the result UUUUUUgh finally it's here every pixel will be here
        lock.release()


    
def fit_window_multi(x:np.ndarray,
                     window:np.ndarray,
                     init_params:np.ndarray,
                     fit_func:callable,
                     bounds:np.ndarray=np.array([np.nan]),
                     segmentation:np.ndarray = np.array([0,np.inf]),
                     window_size:np.ndarray = np.array([[210,800],[0,-1]]),
                     adaptive:bool = True,
                     convolution_function :callable   = lambda lst:np.zeros_like(lst[:,2])+1,
                     convolution_threshold:np.ndarray      = np.array([0.1,10**-4,0.1,100]),
                     convolution_extent_list:np.array = np.array([0,1,2,3,4,5,6,7,8,9,10]),
                     weights:str = None,
                     counter_percent:float = 10,
                     preclean:bool=True,
                     preadjust:bool = True
                     )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Fitting a whole window of dimension (n_t x n_lmb x n_y x n_x) and creating a map of fitting parameters

    Args:
        x (np.ndarray): spectrum_axis of dimension (n_lmb x 1)
        window (np.ndarray): window data cube of dimension (n_t x n_lmb x n_y x n_x)
        init_params (np.ndarray): parameters list as an array of dimension ((number_of_gaussians * 3 + 1) x 1)
        fit_func (collable): the fitting function to use 
        bounds (np.ndarray, optional): boundary list that contains boundaries of everyparameter. Defaults to np.array([np.nan]).
        segmentation (np.ndarray, optional): if the window is needed to be segmented in multiple windows or smaller one. Defaults to np.array([0,np.inf]).
        window_size (np.ndarray, optional): if the window size is needed to be smaller. Defaults to np.array([[210,800],[0,-1]]).
        adaptive (bool, optional): if we want the window fitting to be convoluting. Defaults to True.
        convolution_function (collable, optional): function to convolute in case the convolution is needed. Defaults to lambda lst:np.zeros_like(lst[:,2])+1.
        convolution_threshold (float, optional):the minimu relative error of the parameters else the pixel will be convoluted to the next level. Defaults to 1.
        convolution_extent_list (np.array, optional): the list of levels of convolution. Defaults to np.array([0,1,2,3,4,5]).
        weights (str, optional): string ["I": for a linear weight depend on the value of intensity]. Defaults to None.
        counter_percent (int, optional): percentile counter for visulization. Defaults to 10.
        preclean (bool, optional): True to clean data from saturated pixels and negative values. Defaults to True.
        preadjust (bool, optional): first fit of the whole window in order to predict positions. Defaults to True.

    Returns:
        paramlist,(np.ndarry): data fit map of dimension (shapeinitParam x n_t x n_y x n_x)
        covlist  ,(np.ndarry): covariance matrix of dimension (? x ? x n_t x n_y x n_x)
        quentity ,(np.ndarry): the array of the first dimension content of paramlist |I => intensity| x=> peak position| s=> sigma| 
        convlist ,(np.ndarry): convolution positions levels for every pixel of dimension (n_x x n_y)
        
    """
    assert (counter_percent<=100 and counter_percent>=0)
    concat_window = window[:,:,window_size[0,0]:window_size[0,1],
                               window_size[1,0]:window_size[1,1]]
    dshape  = np.array([concat_window.shape[0],
                        concat_window.shape[2],
                        concat_window.shape[3],
                       ])
    
    _counter = 0
    _imsize = dshape[0]*dshape[1]*dshape[2]
    _next = 0
    
    if preclean: window = Preclean(window)

     
    if adaptive: #creating matrix convolutions
        
        convlist = np.zeros((1 if len(segmentation.shape) == 1 else segmentation.shape[0],*dshape))
        
        conv_data = np.zeros((*convolution_extent_list.shape,*window.shape))
        for i in range(convolution_extent_list.shape[0]):
            if convolution_extent_list[i] == 0:
                conv_data[i]=window.copy();continue
            else:
                ijc_list = fst_neigbors(convolution_extent_list[i])
                ijc_list [:,2]= convolution_function(ijc_list)
                conv_data[i]  = join_dt(window, ijc_list)
    else:
        conv_data = np.zeros((1,*window.shape))
        conv_data[0]=window 
        convlist = np.zeros((1 if segmentation.shape == 1 else segmentation.shape[0],*dshape))
    conv_data = conv_data[:,:,:,window_size[0,0]:window_size[0,1],
                                window_size[1,0]:window_size[1,1]]
    window = concat_window
    
    if len(segmentation.shape) == 1: #we apply segmentation here
        segmentation = np.array([segmentation])
       
    quentity = [] #In the futur the quentity will differ in shape as not all the pics will be obligatory having all the 3 parameters
    sub_windows = []
    sub_xs = []
    sub_init_params = []
    for i in range(segmentation.shape[0]):
        lim_lbda = segmentation[i]
        assert lim_lbda[0]<lim_lbda[1]
        sub_xs.append(x[np.logical_and(x>=lim_lbda[0],x<lim_lbda[1])]) 
        
        sub_windows.append(conv_data[:,:,
                                    np.logical_and(x>=lim_lbda[0],x<lim_lbda[1]),
                                    :,
                                    :
                                        ])
        sub_inits = []    
        for j in range(int((len(init_params)-1)/3)):
            x0 = init_params[j*3+1]
            if x0>lim_lbda[0] and x0<lim_lbda[1]:
                sub_inits.append(init_params[j*3])
                sub_inits.append(init_params[j*3+1])
                sub_inits.append(init_params[j*3+2])   
                quentity.append('I');quentity.append('x');quentity.append('s') 
        if len(quentity)==0:
            raise Exception (
                """Found an empty sub set 
                when segmenting there is no need to add empty segments that contains no peak inside 
                weird segment: {}
                init_params {}""".format(lim_lbda,init_params[np.arange(len(init_params) )%3==1] ))
        sub_inits.append(init_params[-1])          
        quentity.append('B') 
        sub_init_params.append(np.array(sub_inits))
                        
    paramlist = np.zeros((init_params.shape[0]+len(sub_windows)-1, 
                          *dshape))
    
    covlist  = np.zeros((init_params.shape[0]+len(sub_windows)-1, 
                         init_params.shape[0]+len(sub_windows)-1, 
                          *dshape))
   
    if preadjust:        
        for i_seg in range(len(sub_windows)): 
            if preadjust:
                plt.figure()
                sub_init_params[i_seg],var =fit_pixel_multi(sub_xs[i_seg],
                                                np.nanmean(sub_windows[i_seg][0],axis=(0,2,3)),
                                                sub_init_params[i_seg],
                                                fit_func=fit_func,plotit=True,
                                                plot_title_prefix = "preadjust",
                                                weights=weights)

    for i_t in range(dshape[0]):
        for i_y in range(dshape[1]):
            for i_x in range(dshape[2]):
                index = 0
                for i_seg in range(len(sub_windows)):
                    sub_x = sub_xs[i_seg]
                    sub_window = sub_windows[i_seg]
                    sub_inits = sub_init_params[i_seg]
                    conv_thresh = sub_inits.copy()*0
                    conv_thresh[-1] = convolution_threshold[-1]
                    for i_q in range(int(len(sub_inits[:-1])/3)):
                        conv_thresh[i_q+0] = convolution_threshold[0]
                        conv_thresh[i_q+1] = convolution_threshold[1]
                        conv_thresh[i_q+2] = convolution_threshold[2]
                    coeff2 = np.zeros_like(sub_inits) *np.nan
                    var2 = np.zeros((sub_inits.shape[0],
                                     sub_inits.shape[0])) * np.nan
                    best_i = 0
                    for i_ad in range(sub_window.shape[0]):
                        
                        coeff,var = fit_pixel_multi(x =sub_x,
                                            y=sub_window[i_ad,
                                                         i_t,
                                                         :,
                                                         i_y,
                                                         i_x],
                                            ini_params=sub_inits,
                                            fit_func=fit_func,
                                            bounds=bounds,
                                            weights=weights
                                            )
                        if False: #i_ad>4:
                            plt.figure()
                            plt.plot(sub_x,sub_window[i_ad,
                                                    i_t,
                                                    :,
                                                    i_y,
                                                    i_x])
                            plt.plot(sub_x,flat_inArg_multiGauss(sub_x,*coeff))
                            
                            print(np.sqrt(np.diag(var)))
                            print(i_ad,conv_thresh,np.sqrt(np.diag(var))/coeff)
                            plt.show()
                            input()
                        if all(np.isnan(coeff)):
                            pass
                        elif all ((np.sqrt(np.diag(var)))/coeff < conv_thresh):
                            var2 = var
                            coeff2 = coeff
                            best_i = i_ad
                            break
                        else:
                            if all (np.isnan(coeff2)):
                                var2 = var
                                coeff2 = coeff
                                best_i = i_ad
                            elif np.nansum((np.sqrt(np.diag(var )))/coeff/conv_thresh)<np.nansum((np.sqrt(np.diag(var2)))/conv_thresh):
                                var2 = var
                                coeff2 = coeff
                                best_i = i_ad
                            
                    coeff = coeff2                        
                    var = var2
                    convlist[i_seg,i_t,i_y,i_x] = best_i                        
                    index2 = len(coeff) + index
                    paramlist[index:index2,i_t,i_y,i_x] = coeff
                    covlist  [index:index2,index:index2,i_t,i_y,i_x] = var
                    index = index2              
                if _next<= _counter and counter_percent<100:
                    print( "{:05.2f}% generated".format(_counter/_imsize*100))
                    _next = int(counter_percent/100 * _imsize + _counter)
                _counter+=1
    return paramlist, covlist, quentity, convlist



def _fit_window_locking(
                x:np.ndarray,
                WindowOrShmm            :np.ndarray or dict("sharedMempory"),
                
                init_params_lock        :np.ndarray,
                convert_to_unlock       :callable  ,
                quentities_lock         :list[str] ,
                quentities_unlock       :list[str] ,
                fit_func_lock           :callable  ,
                fit_func_unlock         :callable  ,
                bounds_lock             :np.ndarray     = np.array([np.nan]),
                bounds_unlock           :np.ndarray     = np.array([np.nan]),
                
                unlock_condition        :callable       = lambda params: True if (params[0]>0.1) else False,
                
                segmentation            :np.ndarray     = np.array([0,np.inf]),
                window_size             :np.ndarray     = np.array([[210,800],[0,-1]]),
                meta                    :dict           = None,
                convolution_function    :callable       = lambda lst:np.zeros_like(lst[:,2])+1,
                convolution_threshold   :np.ndarray     = np.array([0.1,10**-4,0.1,100]),
                convolution_extent_list :np.array       = np.array([0,1,2,3,4,5]),
                weights                 :str            = None,
                preclean                :bool           = True,
                preadjust               :bool           = True,
                njobs                   :int            = 1,
                verbose                 :int            = 0,
                describe_verbose        :bool           = True, 
                lock                                    = None,
                               
                par                     :dict           = None,
                cov                     :dict           = None,
                con                     :dict           = None,
                loc                     :dict           = None,
                     )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray] or None:
    """Fitting a whole window of dimension (n_t x n_lmb x n_y x n_x) and        creating a map of fitting parameters

        Args:
            x (np.ndarray): spectrum_axis of dimension (n_lmb x 1)
            window (np.ndarray): window data cube of dimension (n_t x n_lmb x n_y x n_x)
            init_params (np.ndarray): parameters list as an array of dimension ((number_of_gaussians * 3 + 1) x 1)
            fit_func (collable): the fitting function to use 
            bounds (np.ndarray, optional): boundary list that contains boundaries of everyparameter. Defaults to np.array([np.nan]).
            segmentation (np.ndarray, optional): if the window is needed to be segmented in multiple windows or smaller one. Defaults to np.array([0,np.inf]).
            window_size (np.ndarray, optional): if the window size is needed to be smaller. Defaults to np.array([[210,800],[0,-1]]).
            window_size (metaObject|dict, optional): meta data 
            adaptive (bool, optional): if we want the window fitting to be convoluting. Defaults to True.
            convolution_function (collable, optional): function to convolute in case the convolution is needed. Defaults to lambda lst:np.zeros_like(lst[:,2])+1.
            convolution_threshold (float, optional):the minimu relative error of the parameters else the pixel will be convoluted to the next level. Defaults to 1.
            convolution_extent_list (np.array, optional): the list of levels of convolution. Defaults to np.array([0,1,2,3,4,5]).
            weights (str, optional): string ["I": for a linear weight depend on the value of intensity]. Defaults to None.
            counter_percent (int, optional): percentile counter for visulization. Defaults to 10.
            preclean (bool, optional): True to clean data from saturated pixels and negative values. Defaults to True.
            preadjust (bool, optional): first fit of the whole window in order to predict positions. Defaults to True.

        Returns:
            paramlist,(np.ndarry): data fit map of dimension (shapeinitParam x n_t x n_y x n_x)
            covlist  ,(np.ndarry): covariance matrix of dimension (? x ? x n_t x n_y x n_x)
            quentity ,(np.ndarry): the array of the first dimension content of paramlist |I => intensity| x=> peak position| s=> sigma| 
            convlist ,(np.ndarry): convolution positions levels for every pixel of dimension (n_x x n_y)

        """

    if type(lock) == type(None):
        lock = Lock()
    if verbose<=-2: warnings.filterwarnings("ignore")
    if describe_verbose: verbose_description(verbose);sleep(5)
    if verbose >0: print(f"The window fit is called for {meta['DATE_SUN']}")    
    
    if type(None) in [type(par),type(cov),type(con)]:return_data = True
    else:return_data = False
    
    if verbose >1:print(f"The output shared memory have {'not' if return_data else ''} been given")    

    if True: #assertion part 
        pass
        
    
    if True: #handling input data
        if type(WindowOrShmm) == dict:
            shmm_raw,data_raw = gen_shmm(create=False,
                                    name =WindowOrShmm["name"], 
                                    dtype =WindowOrShmm["type"], 
                                    shape =WindowOrShmm["shape"] )
        else: 
            shmm_raw,data_raw = gen_shmm(create=True,
                                        ndarray=WindowOrShmm)
        if verbose>1: print(f"""
                    fit Window input:
                    shmm_raw: {{'name':{shmm_raw.name},'type':{data_raw.dtype},'shape':{data_raw.shape} }}
                    """)
    
    if True: #making sure that the Window size array is integer like and contains rather index than relative ones 
        ws = (window_size.copy()).astype(int)
        for i in range(ws.shape[0]):
            for j in range(ws.shape[1]):
                if ws[i,j] <0: ws[i,j] = data_raw[0,0].shape[i]+1+ws[i,j]
        if verbose > 1: print(f"the window_size have been checked\n{ws}\n{data_raw[0,0].shape}")
    
    dshape  = np.array([data_raw.shape[0],
                        data_raw.shape[2],
                        data_raw.shape[3],
                       ])
    
    if preclean: window = Preclean(data_raw.copy()) #clean data from all saturations
     
    if True: #creating pixel convolution matrices (always adaptive) 
        if len(np.array(segmentation).shape) != 1:
            raise Exception("""
            The new pipeline doesn't support multiple segments anymore
            the segmentation should be one list""")
        
        conv_data = np.zeros((*convolution_extent_list.shape,*window.shape))
        if verbose>=2: print('creating convolution list...')
        for i in range(convolution_extent_list.shape[0]):
            if convolution_extent_list[i] == 0:
                conv_data[i]=window.copy();continue
            else:
                ijc_list = fst_neigbors(convolution_extent_list[i])
                ijc_list [:,2]= convolution_function(ijc_list)
                conv_data[i]  = join_dt(window, ijc_list)
        if verbose>=2: print('convolution list created')
 
    # now the data_conv is a window that has an additional convolution ax
    shmm_data_conv, data_conv = gen_shmm(create = True, ndarray=conv_data)
    # del conv_data
    war = {"name":shmm_data_conv.name,"type":data_conv.dtype,"shape":data_conv.shape}
    if verbose>1: print(f"""
                    fit window input:
                    shmm_war: {{'name':{war["name"]},'type':{war["type"]},'shape':{war["shape"]} }}
                    """)
    
    if preadjust:# mke sure t review this after correcting pixel_fit
        init_params2,var =fit_pixel_multi(x,
                                            np.nanmean(window[:,
                                                            :,
                                                            ws[0,0]:ws[0,1],
                                                            ws[1,0]:ws[1,1]]
                                                    ,axis=(0,2,3)),
                                            init_params_lock,
                                            quentities=quentities_lock,
                                            fit_func=fit_func_lock,
                                            plotit=True,
                                            plot_title_prefix = "preadjust",
                                            weights=weights,
                                            verbose=verbose)
        if verbose>=2: print(f"parameters were Preadjusted\nOriginal: {fit_func_lock}\nAdjusted:{init_params2}\n,",var)
        
        if not((np.isnan(init_params2)).any()):
            init_params_lock = init_params2
            if verbose>=0: print('Preadjust was successful')
        else:
            if verbose>=0: print('Preadjust was not successful')
    
    if True: #generating pixel index positions for linear treatement 
        def index_list(njobs, ws=ws,verbose=verbose):
            index_list = np.zeros(((ws[0,1] - ws[0,0]) * 
                                   (ws[1,1] - ws[1,0]) , 
                                   2),dtype=int
                                  )
            inc = 0
            for i in range(ws[0,0],ws[0,1]):
                for j in range(ws[1,0],ws[1,1]):
                    index_list[inc] = [i,j]
                    inc+=1
            Job_index_list = []
            nPerJob = (len(index_list)//njobs)
            reste = len(index_list)%njobs
            if verbose >=2: print("N pixels per job",nPerJob)
            if verbose >=2: print("reste for first job",reste)
            for i in range(njobs):#distributing pixels over jobs
                Job_index_list.append(index_list[
                    i*nPerJob+(reste if i!=0 else 0):
                    min((i+1)*nPerJob +reste,len(index_list))
                                                ])
            return Job_index_list
        Job_index_list = index_list(njobs,ws=ws,verbose=verbose)
        if verbose>=2:
            for i,j in enumerate(Job_index_list):
                print(i,len(j)) 
                
    if True: #handeling output matrix
        if return_data:
            init_params_unlock = convert_to_unlock(init_params_lock)
            data_par  = np.zeros((init_params_unlock.shape[0],                            *dshape))* np.nan
            data_cov  = np.zeros((init_params_unlock.shape[0],init_params_unlock.shape[0],*dshape))* np.nan
            data_con  = np.zeros(                                                          dshape )* np.nan
            data_loc  = np.zeros(                                                          dshape )* np.nan
            
            shmm_par,data_par = gen_shmm(create=True,ndarray=data_par) 
            shmm_cov,data_cov = gen_shmm(create=True,ndarray=data_cov) 
            shmm_con,data_con = gen_shmm(create=True,ndarray=data_con) 
            shmm_loc,data_loc = gen_shmm(create=True,ndarray=data_loc) 
            par = {"name":shmm_par.name ,"type":data_par.dtype ,"shape":data_par.shape }
            cov = {"name":shmm_cov.name ,"type":data_cov.dtype ,"shape":data_cov.shape }
            con = {"name":shmm_con.name ,"type":data_con.dtype ,"shape":data_con.shape }
            loc = {"name":shmm_loc.name ,"type":data_loc.dtype ,"shape":data_loc.shape }
        if verbose>1: print(f"""
                    fit window output:
                    shmm_par: {{'name':{par["name"]},'type':{par["type"]},'shape':{par["shape"]} }}
                    shmm_cov: {{'name':{cov["name"]},'type':{cov["type"]},'shape':{cov["shape"]} }}
                    shmm_con: {{'name':{con["name"]},'type':{con["type"]},'shape':{con["shape"]} }}
                    shmm_loc: {{'name':{loc["name"]},'type':{loc["type"]},'shape':{loc["shape"]} }}
                    """)

    Processes =[]
    for i in range(njobs): #preparing processes:
        keywords = {
                "x"                       : x,
                "list_indeces"            : Job_index_list[i],
                "war"                     : war,
                "par"                     : par,
                "cov"                     : cov,
                "con"                     : con,
                "loc"                     : loc,
                 
                "init_params_lock"        : init_params_lock,
                "convert_to_unlock"        : convert_to_unlock,
                "quentities_lock"         : quentities_lock,
                "quentities_unlock"       : quentities_unlock,
                "fit_func_lock"           : fit_func_lock,
                "fit_func_unlock"         : fit_func_unlock,
                "bounds_lock"             : bounds_lock,
                "bounds_unlock"           : bounds_unlock,
                "unlock_condition"        : unlock_condition,
                
                "weights"                 : weights,
                "convolution_threshold"   : convolution_threshold,
                "convolution_extent_list" : convolution_extent_list,
                "verbose"                 : verbose,
                "describe_verbose"        : False,
                "lock"                    : lock
        }

        Processes.append(Process(target=task_fit_pixel_lock,kwargs=keywords))
        Processes[i].start()
        if verbose>=1: print(f"Starting process job: {i+1:02d} on raster fits\nJob list contains: {len(Job_index_list[i])} pixel")
    for i in range(njobs): #join all processes
        Processes[i].join()
        
    if return_data: #returning if par/co/con haven't been given
        shmm_par,data_par = gen_shmm(create=False,name=par["name"],dtype=par["type"],shape=par["shape"]) 
        shmm_cov,data_cov = gen_shmm(create=False,name=cov["name"],dtype=cov["type"],shape=cov["shape"]) 
        shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"]) 
        shmm_loc,data_loc = gen_shmm(create=False,name=loc["name"],dtype=loc["type"],shape=loc["shape"]) 
        # lock.release()
        return (
            data_par.copy(),
            data_cov.copy(),
            data_con.copy(),
            data_loc.copy())
    
def task_fit_pixel_lock(
                    x:np.ndarray,
                    list_indeces:np.ndarray,
                    war: dict,
                    par: dict,
                    cov: dict,
                    con: dict,
                    loc: dict,
                    
                    init_params_lock        :np.ndarray ,
                    convert_to_unlock        :callable   ,
                    quentities_lock         :callable   ,
                    quentities_unlock       :np.ndarray ,
                    fit_func_lock           :callable  ,
                    fit_func_unlock         :callable   ,
                    bounds_lock             :np.ndarray ,
                    bounds_unlock           :np.ndarray ,
                    unlock_condition        :callable   ,
                    
                    weights: str = None,
                    convolution_threshold = None,
                    convolution_extent_list = None, 
                    verbose=0,
                    lock=None,
                    describe_verbose = False,
                    **kwargs
):
    
    shmm_war,data_war = gen_shmm(create=False,name=war["name"],dtype=war["type"],shape=war["shape"]) 
    shmm_par,data_par = gen_shmm(create=False,name=par["name"],dtype=par["type"],shape=par["shape"]) 
    shmm_cov,data_cov = gen_shmm(create=False,name=cov["name"],dtype=cov["type"],shape=cov["shape"]) 
    shmm_con,data_con = gen_shmm(create=False,name=con["name"],dtype=con["type"],shape=con["shape"]) 
    shmm_loc,data_loc = gen_shmm(create=False,name=loc["name"],dtype=loc["type"],shape=loc["shape"]) 
    
    for index in list_indeces:
        i_y,i_x = index
        if verbose>=2: print(f"fitting pixel [{i_y},{i_x}]")
        i_ad            = -1   
        locked          = 1
        
        best_cov_lock   = np.zeros((*init_params_lock.shape  ,*init_params_lock.shape  )) * np.nan
        best_par_lock   = np.zeros((*init_params_lock.shape  ,)                         ) * np.nan

        if verbose>2: print(f"y data: {data_war[i_ad,0,:,i_y,i_x]}")
        
        while True: #this will break only when the convolution threshold is met or reached max allowed convolutions
                                                              #                 |
            if i_ad == len(convolution_extent_list)-1:        #<----------------'
                # print(i_y,i_x,i_ad, 'no shit found'); 
                # best_cov_lock   = last_cov_lock
                # best_par_lock   = last_par_lock
                # best_con        = convolution_extent_list[i_ad]
                # best_i_ad       = i_ad
                break                         
            i_ad +=1
            
            # print(f"fit_func_lock: {fit_func_lock}");sleep(20)
            last_par_lock,last_cov_lock = fit_pixel_multi(
                                                x =x,
                                                y=data_war[i_ad,
                                                            0,
                                                            :,
                                                            i_y,
                                                            i_x].copy(),
                                                ini_params=init_params_lock,
                                                quentities=quentities_lock,
                                                fit_func=fit_func_lock,
                                                bounds=bounds_lock,
                                                weights=weights,
                                                verbose=verbose,
                                                describe_verbose=describe_verbose,
                                                plot_title_prefix=f"{i_y:04d},{i_x:03d},{i_ad:03d}",
                                                )
            # print( np.sqrt(np.diag(last_cov_lock))/last_par_lock < convolution_threshold)
            # print((np.sqrt(np.diag(last_cov_lock))/last_par_lock < convolution_threshold).all())
            # print(i_ad)
            if (np.isnan(last_par_lock)).any():
                best_con  = convolution_extent_list[i_ad]
                best_i_ad = i_ad
                # print(i_y,i_x,i_ad, 'par is nan')
            elif ((np.sqrt(np.diag(last_cov_lock)))/last_par_lock < convolution_threshold).all():
                best_cov_lock   = last_cov_lock
                best_par_lock   = last_par_lock
                best_con        = convolution_extent_list[i_ad]
                best_i_ad       = i_ad
                
                # print(i_y,i_x,i_ad, 'Found fit')
                break
            else:
                if (np.isnan(best_par_lock)).all():
                    best_cov_lock   = last_cov_lock
                    best_par_lock   = last_par_lock
                    best_con        = convolution_extent_list[i_ad]
                    best_i_ad       = i_ad
                    # print(i_y,i_x,i_ad, 'previous best par are nans replacing them')
                    
                elif np.nansum(
                        (np.sqrt(np.diag(last_cov_lock)))/last_par_lock/convolution_threshold
                    )<np.nansum(
                        (np.sqrt(np.diag(best_cov_lock)))/best_par_lock/convolution_threshold
                                                                              ):
                    # print(i_y,i_x,i_ad, 'wiinning by sum')
                    best_cov_lock = last_cov_lock
                    best_par_lock = last_par_lock
                    best_con = convolution_extent_list[i_ad]
                    best_i_ad = i_ad
                else: 
                    best_cov_lock = last_cov_lock
                    best_par_lock = last_par_lock
                    best_con = convolution_extent_list[i_ad]
                    best_i_ad = i_ad
                    # print('something is wrong') 
        
        if verbose>=2 : print(f"best_par_lock: {best_par_lock}\nbest_con_lock: {best_con}")
        i_ad = best_i_ad 
        # print("XXXX",best_con,convolution_extent_list[i_ad])

        if unlock_condition(best_par_lock,best_cov_lock):
            
            if False: #Only for debugging purposes
                best_par_lock,best_cov_lock = fit_pixel_multi(
                        x =x,
                        y=data_war[i_ad,
                                    0,
                                    :,
                                    i_y,
                                    i_x].copy(),
                        ini_params=init_params_lock,
                        quentities=quentities_lock,
                        fit_func=fit_func_lock,
                        bounds=bounds_lock,
                        weights=weights,
                        verbose=-3,
                        describe_verbose=describe_verbose,
                        plot_title_prefix=f"{i_y:04d},{i_x:03d},{i_ad:03d}_a",
                        )
            
            init_params_unlock = convert_to_unlock(best_par_lock)
            best_par_unlock,best_cov_unlock = fit_pixel_multi(
                    x          = x,
                    y          = data_war[i_ad,
                                0,
                                :,
                                i_y,
                                i_x].copy(),
                    ini_params = init_params_unlock,
                    fit_func   = fit_func_unlock,
                    quentities = quentities_unlock,
                    bounds     = bounds_unlock,
                    weights    = weights,
                    verbose    = verbose,
                    describe_verbose=describe_verbose,
                    plot_title_prefix=f"{i_y:04d},{i_x:03d},{i_ad:03d}_b",
                    )
            locked = 0
            
            # print(f'\n\n\n\n\n\n\n')
            # print(f"best_par_unlock\n {best_par_unlock}")
            # print(f"init_params_unlock\n {init_params_unlock}")
            # print(f"best_par_lock\n {best_par_lock}")
            # print(f"i_ad\n {i_ad}")
            # print(f"convolution_extent_list[i_ad]\n {convolution_extent_list[i_ad]}")
            # print(f"best_con\n {best_con}")
            # print(f"fitting pixel [{i_y},{i_x}]")
            # sleep(3600)
            if (np.isnan(best_par_unlock)).any():
                best_par_unlock = convert_to_unlock(best_par_lock)
                best_cov_unlock = convert_to_unlock(best_cov_lock)
                locked = 1
                
        else:
            best_par_unlock = convert_to_unlock(best_par_lock)
            best_cov_unlock = convert_to_unlock(best_cov_lock)
            locked = 1            
        
        lock.acquire()
        data_par[  :,0,i_y,i_x] = best_par_unlock #the result UUUUUUgh finally it's here every pixel will be here
        data_cov[:,:,0,i_y,i_x] = best_cov_unlock #the result UUUUUUgh finally it's here every pixel will be here
        data_con[    0,i_y,i_x] = best_con        #the result UUUUUUgh finally it's here every pixel will be here
        data_loc[    0,i_y,i_x] = locked          #the result UUUUUUgh finally it's here every pixel will be here
        lock.release()
   
   
   
