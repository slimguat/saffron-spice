# SlimPy
## (Unpackaged version)
Data fitting pipeline adapted to SPICE instrument onboard SolaOrbiter. I will apreceate any suggestions to improve the quality of the content

## Requirements
1. Python<=3.9.15 (I suggest to use pyenv to change versions easily )
2. Chianti database ([Download]([https://link-url-here.org](https://www.chiantidatabase.org/chianti_download.html)https://www.chiantidatabase.org/chianti_download.html))
3. Add the variable `XUVTOP` to the path of the database (After extraction)<br>

Linux :
```export XUVTOP=/home/../pathTo/Chianti_Database```<br>
Windows :
```
search Environment Variables in start panel
In the Environment Variables window, under the "System variables" section, scroll down to find the "Path" variable.
Click on "New..." to add a new system variable.
Set Variable Name and Value:

In the "New System Variable" dialog, set the variable name as XUVTOP.
In the "Variable value" field, enter the path or database value you want to assign to XUVTOP.
Save the Variable:

Click "OK" to save the new system variable.
```
python :
```python
import os

# Set the XUVTOP environment variable, will be gone by the end of the script
os.environ['XUVTOP'] = 'path/to/your/database'
```
## Install
1. Install the right version of python 

```bash
pyenv install 3.9.15
```
2. Create and activate your environment

```bash
pyenv virtualenv 3.9.15 SPICE_SlimPy
pyenv activate SPICE_SlimPy
```

3. install requirement

The library hasn't bee packaged yet so it's better to put in the parent folder to be able to use it. This is going to be changed eventually.

Inside SlimPy folder do:
```
pip install -r requirements.txt 
```
## Tutorial
### Basic Fitting 
```python
from SlimPy.manager.manager import Manager
from SlimPy.utils import get_input_template
get_input_template(where='./input.json')
session = Manager("./input.json")
# print(session) if you want
session.build_files_list()
session.build_rasters()
# change sessions.rasters[i].init_params if not good
session.run_preparations() #clean,convolve,denoise,despike,estimate error
session.fit_manager() #fit and save
```


#### Code details
The odds are that you will need no major adjustments for the fitting as the input parameters are most likely tuned out.

**JSON Input**
In the beginning you will need a input json file that contains all the parameters needed for the analysis part.
```python
from SlimPy.utils import get_input_template
get_input_template(where='./input.json')
```
Inside the json the most important keys are: 

1. "SELECTION_MODE": list, folder, date intervale
2. "files"         : list of files in case SELECTION_MODE is a list 
3. "data_filename" : "use ::SAMENAME to replace the placeholder, ::SAMENAMEL2.5 is to change L2 to L2.5 by the name of the fits input file or ::CONV to replace by the convolution level(s) with - as separator or ::TIME to replace it by fitting start time"
4. "data_save_dir" : The folder to save data in ('.'=in the current directory)
4. "window_size"   : the size of the window to fitted in y,x (not x,y) in pixel coordinates. [0,Null] = all the pixel in a given direction.
5. "Jobs"          : Number jobs while fitting (It's 30 by default so make sure to change it to the right number) 

**Initiating and preparing Manager**
```python
from SlimPy.manager import Manager
session = Manager("./input.json")
session.build_files_list()
```
- You can visualize and change this list by accessing the variable session.selected_fits

**Building rasters (\<RasterFit\>)**
Building rasters mean you will have a list of rasters that each have their parameters passed from the \<Manager\>. It's also auto calculating the initial parameters based on a catalogue the library. If you want to see the results of initial parameters set the verbose for geninits_verbose >3 or <-2. the plots are going to be in a `./tmp` folder with dates of creation in their names.
```python
session.build_rasters()
``` 
Now you have a new variable in your \<Manager\> (session.rasters) that have a list of \<RasterFit\> objects. if you are not satisfied of the initial parameters algorithm you can adjust it manually.
you only have to access these 2 parameters:<br> 
(raster.init_params): list of array for each window contains all the parameter for a gaussian [int,wvl,wid,int,wvl,wid....,Bg] <br>
(raster.quantities) the same list of the same size as init_params however, it's a list of characters for each parameter. 'I' for intensity, "x" for wavelength, "s" for width <br>

Inside each raster there is a list of windows as \<WindowFit\> object, but, it's going to be explained later

**data cleaning**
denoising, error estimation (for fitting weights), despising (remove cosmics), convolving (improving S2N ratio) is not yet applied so we are going to do so by calling. 
```python
session.run_preparations()
```
this is also possible by calling the same method inside one of the rasters or even one of the windows 
```python
session.run_preparations()
session.rasters[i].run_preparations()
session.rasters[i].windows[j].run_preparations()
```
Data preparation is a little time consuming so it will not run if it's called twice in the second time. unless if you set redo=True as an argument for .run_preparations()

**Fitting data**
```python
session.fit_manager()
```
I tried a progress bar buuut Hmmm it's not yet well coded -_- . So it looks stupid. But, it runs independently and doesn't affect the fitting processes.

### Managing Line Catalog
TODO
### Locking the lines
TODO
### Adaptive Convolution
TODO
### How Denoise works?
TODO

Credit:  [Frédéric Auchère]([frederic.auchere@universite-paris-saclay.fr])
### How Despike works?
TODO

Credit:  [Frédéric Auchère]([frederic.auchere@universite-paris-saclay.fr])
### How Sigma is estimated?
Credit:  [Eric Buchlin]([eric.buchlin@universite-paris-saclay.fr])
1. Estimate errors on data: Ask Eric Buchlin.
2. Estimation after convolution: $\frac{\sqrt{\Sigma_{Conv\_pixels}{I_{val}^2 }}}{N_Conv\_pixels}$
3. with after denoise: Nothing done. That means we overshoot error values a bit because of denoise

### Postprocessing (Composition maps)
TODO