
# SAFFRON
<a href="https://pypi.org/project/saffron-spice"><img alt="Latest version" src="https://badge.fury.io/py/saffron-spice.svg"></a>

## Overview

`SAFFRON` is a Python module designed for fitting spectral data using various models and functions. It provides tools for fitting individual pixels, spectral windows, and entire rasters, handling initial parameters, and managing data post-processing. This module is particularly useful for applications in solar physics and spectral analysis, composition analysis using data from [SPICE instrument](https://spice.ias.u-psud.fr/) onboard [Solar Orbiter](https://sci.esa.int/web/solar-orbiter).

## Requirements

1. python = ">=3.9.15,<4.0"

2. The module requires the following dependencies:
```text
astropy
colorama
docutils = ">=0.14,<0.21"
fiplcr
ipympl
ipyparallel #Chiantipy needs it but it is not in the dependencies
ipywidgets
matplotlib
multiprocess
ndcube
numba
numpy
opencv-python
pandas
requests
rich
roman
scipy
setuptools
sospice
sunpy
tqdm
watroo
```
3. Chiati database [Optional if using elemental composition analysis] 

## Installation

A. **Installing the module**

To install the module, follow these steps:

```bash
pip install git+https://github.com/yourusername/spice-saffron.git
```
Or, you can install directly from the GitHub repository:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spice-saffron.git
    ```

2. Navigate to the project directory:
    ```bash
    cd spice-saffron
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

B. **Chianti database setup**

Inside the terminal run the command `setup-chianti` than follow instructions to download extract and set the variable parameter (XUVTOP) 


## Tutorial
### Basic Fitting 
```python
from SlimPy.manager.manager import Manager
from SlimPy.utils import get_input_template
get_input_template(where='./input.json')
#Go inside JSON file and put the list of L2 files to use
session = Manager("./input.json")
# print(session) if you want
session.build_files_list()
#you can do session.fitinits = 4 before calling the next command if you want to save the graphs of initial parameters preparation in side ./tmp/ as png
session.build_rasters()
# change sessions.rasters[i].init_params if not good or 
# change the interval if there are artifacts in the borders of spectral windows (you can see if there are by using SAFFRAN.utils.quickview(L2_path))
session.run_preparations() #clean,convolve,denoise,despike,estimate error
session.fit_manager() #fit and save
# the fitting will be saved in the directory that is specified in input.json under the keyword "data_save_dir" and the naming structure is under the keyword "data_filename"  
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
5. "Jobs"          : Number jobs while fitting (It's 30 by default so make sure to change it to the right number). Else your CPU will pop out of the machine and sit next to you.

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
session.run_preparations() #Loop over all rasters of the session that will loop all over the windows
session.rasters[i].run_preparations() #loop over all the windows of the raster i
session.rasters[i].windows[j].run_preparations() #Apply only on the window j of the raster i 
```
Data preparation is a little time consuming depend on the convolution selection and whether denoise and despike were activated so it will not run if it's called twice in the second time. unless if you set redo=True as an argument for .run_preparations()

**Fitting data**
```python
session.fit_manager()
``` 
you can also run fitting for only one raster or even one window
```python
session.rasters[i].fit_raster() #i index of the raster you want to fit
session.rasters[i].windows[j].fit_window() #j index of the window inside raster of index i you want to fit
```


I tried a progress bar buuut Hmmm it's not yet well coded -_- . So it looks stupid. But, it runs independently and doesn't affect the fitting processes.


### Managing Line Catalog

SAFFRON employs a dynamic approach to identify line parameters essential for initiating the fitting process. This functionality is anchored in an integral component of SAFFRON: the line catalog. Our module not only comes with a pre-configured internal catalog but also offers the flexibility to customize or even replace it with your catalog, catering to all lines you deem necessary, whether they currently exist or not. The existing catalog predominantly features main lines optimized for composition lines ...

#### Getting Started with the LineCatalog Class

The `LineCatalog` class is your gateway to interacting with the line catalog. Here's how you can get started:

```python
from saffron.catalog import LineCatalog
# Initialize the catalog; you can specify a custom file location for your catalog
catalog = LineCatalog(file_location="path/to/your/catalog.json") #if not specified it will upload the default catalog
```

#### Loading and Viewing the Catalog

Upon initialization, the catalog is automatically loaded. You can directly interact with it to view the lines and spectral windows it contains.

```python
# Access the catalog data
lines = catalog.get_catalog_lines()
windows = catalog.get_catalog_windows()
```

#### Modifying the Catalog

SAFFRON's `LineCatalog` facilitates various operations to tailor the catalog to your needs, including adding or removing lines and spectral windows.
You can change the catalogue manually by going to `saffron.catalog.SPICE_SpecLines.json` or with code

- **Adding a New Line**

  ```python
  catalog.add_line(name="NewLine", wvl=123.45)
  ```

- **Removing a Line**

  ```python
  catalog.remove_line(name="OldLine")
  ```

- **Adding a Spectral Window**
  **Important Note:** If there is a line that doesn't exist yet you can not use it to create a new window. the line must be first added to the list of lines with its wavelength value 
  ```python
  catalog.add_window(lines=["Line1", "Line2"], max_line="Line2")
  ```

- **Removing a Spectral Window**

  ```python
  catalog.remove_window(lines=["Line1", "Line2"])
  ```

#### Saving Changes

Any modifications made can be persisted back to the JSON file, ensuring your catalog stays up-to-date.

```python
catalog.dump(new_path="path/to/save/your/updated_catalog.json")
```
#### Using your catalog
In `Manager.build_raster` add the argument `catalog_location` that has the location of your catalog. If you modified the default catalog, no need to specify the loaction

#### Important to know

- Ensure all new lines added have a unique name within the catalog.
- When adding a spectral window, the `max_line` should be the most intense line within that window.
- Use descriptive names for lines to maintain clarity and ease of identification.

With SAFFRON's line catalog management, you're equipped to customize your spectral line analysis to fit your unique requirements, enhancing the flexibility and accuracy of your scientific exploration.

### Locking Technique of Spectral Lines
#### Overview
The locking technique can be an essential part of the spectral line fitting in some cases in SAFFRON module, designed to enhance the accuracy of distinguishing between blended lines. This method is particularly useful in complex spectra where lines such as S_iv 750 and mg_ix 749 are blended in each other. Thus, we try to lock the position of mg_ix 706 with mg_ix 749 (and sometimes: S_iv 750 with S_iv 748) to increase the accuracy on the position. This would also cause a better estimation on the other parameters

#### Application
if the lines to lock are in two different windows we should first **fuse those windows** 
Example: (Approximate wavelengths)
- session.raster[i],window[0] have
 
  0 . O <span style="font-variant: small-caps;">iii</span> ~702.7 &Aring;.  

  1 . O <span style="font-variant: small-caps;">iii</span> ~703.2 &Aring;.

  2 . Mg <span style="font-variant: small-caps;">ix</span> ~706 &Aring;.

- session.raster[i],window[1] have

  0 . S <span style="font-variant: small-caps;">iv</span> ~748 &Aring;.

  1 . Mg <span style="font-variant: small-caps;">ix</span> ~749.5 &Aring;.

  2 . S <span style="font-variant: small-caps;">iv</span> ~750 &Aring;. 

If we want to lock the two sulfur lines or oxygen lines we don't need to fuse the windows but locking mg_ix lines need to fuse window[0] with window[1] 

##### Windows Fusion
```python
session.fuse_windows([0,1]) #0 is windiow 0 and 1 is window 1  
```
if shape of the argument is ($N_{selected \_files}$x2) there will be fusions different each file in the session

you can also fuse by selecting a raster directly
```python
session.rasters[i].fuse_windows(0,1) #0 is windiow 0 and 1 is window 1  
```
Now that you have fused the windows you can access this fused window by calling ```session.rasters[i].fused_windows[0]``` 
**Important note:** Once you fuse two windows they are automatically excluded from fitting and the resulting fused window is automatically included. However the component windows are not deleted you can specially fit them explicitly by running ```session.rasters[i].windows[0<or >].fit_window()```

##### Window locking 
After Fusion the index of each line in the fused window will be: 
- session.raster[i],window[0] have
  0 . O <span style="font-variant: small-caps;">iii</span> ~702.7 &Aring;.
   
  1 . O <span style="font-variant: small-caps;">iii</span> ~703.2 &Aring;.

  2 . Mg <span style="font-variant: small-caps;">ix</span> ~706 &Aring;.

  3 . S <span style="font-variant: small-caps;">iv</span> ~748 &Aring;.

  4 . Mg <span style="font-variant: small-caps;">ix</span> ~749.5 &Aring;.

  5 . S <span style="font-variant: small-caps;">iv</span> ~750 &Aring;. 

and now we use 
```Manager.set_lock_protocol(window_type,window_index,lock_line1_index,lock_line2_index,loc_distance)``` to specify the locking protocol we want to use.

```window_type```: "fuse" if the protocol in a fused window, "solo" if the protocol is in a single window. 

```window_index```: once selected window type. you select the order of the selected window in their respective list.

```lock_line1_index```: the index of the leading line for locking.

```lock_line2_index```: the index of the following line for locking into the leading one.

```loc_distance```: The distance in between the two lines that it will remain the same during fitting. 

```python 
session.set_lock_protocol("fuse", 0, 2, 4, (749.54-706.02))
session.set_lock_protocol("fuse", 0, 3, 5, (750.22-748.40))
```

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
2. Estimation after convolution: $\frac{\sqrt{\Sigma_{Conv\_pixels}{\Delta I_{val}^2 }}}{N_Conv\_pixels}$
3. with after denoise: Nothing done. That means we overshoot error values a bit because of denoise

### Postprocessing (Composition maps)
This part is adapted from [fiplcr](https://pypi.org/project/fiplcr/) module to SAFFRON output data. The FIP maps are generated using [Linear Combination Method (LCR)](https://doi.org/10.1051/0004-6361/201834735)

#### Using \<SPICEL3Raster\>
```python 
L3data_dir = './<output_data>/'
LFLines = ('s_4',750),('s_5',786),      #Low  Fip lines (needed to compute the FIP maps using LCR the choice is conditioned read the paper for more infos) 
HFLines = ('n_4',765.15),('n_3',991.59) #High Fip lines (needed to compute the FIP maps using LCR the choice is conditioned read the paper for more infos) 


from saffron.postprocessing import SPICEL3Raster
L3_raster = SPICEL3Raster(folder_path = _con_L3_data_dir) #Generating L3 raster: loading outputs and computing the radiance maps and the error.
L3_raster.gen_compo_LCR(LFLines =LFLines,HFLines =HFLines)# optimizing the linear combination and computing FIP maps and the error values. Add "suppressOutput=True" if you want no graphs 
```

Finally you have FIP maps

```L3_raster.FIP``` 2D numpy array FIP map.

```L3_raster.FIP_err``` 2D numpy array FIP error map.

```L3_raster.show_all_wvls()``` return all available lines' wavelengths in order.

```L3_raster.find_line(wvl)``` return the a line as ```saffron.postprocessing.SPECLine``` object with the closest wavelength to wvl.

```L3_raster.search_lines(ion=None, wavelength=None, closest_wavelength=None) -> line_selection```

This method searches for spectral lines based on specified criteria such as ion, wavelength, or the closest wavelength.
- **ion**: The ion to search for, specified in the Chianti structure naming convention (e.g., "fe_18", "o_6").
- **wavelength**: The exact wavelength to search for.
- **closest_wavelength**: The wavelength to find the closest match for. Only one of `wavelength` or `closest_wavelength` can be specified.
- **Output**: Returns a list of lines that match the search criteria.

```L3_raster.correct_doppler_gradient(direction="x", reference={"ion": "ne_8", "closest_wavelength": 770}, verbose=0)```

This method performs a primitive Doppler gradient correction for the entire raster based on a reference line's gradient. To revert to the original Doppler values, use:
```L3_raster.lines[i].reset_doppler()```
- **direction**: The direction of the gradient correction. Can be `'x'`, `'y'`, or `'xy'`. Default is `'x'`.
- **reference**: A dictionary specifying the reference line for the Doppler correction. Default is `{"ion": "ne_8", "closest_wavelength": 770}`.
- **verbose**: Controls the verbosity of the output. Default is `0`.

```L3_raster.lines``` list of lines of type ```saffron.postprocessing.SPECLine``` in the raster.

```L3_raster.lines[i].wavelength```  Return the wavelength  of the line i.

```L3_raster.lines[i].observatory``` Return the observatory of the line i.

```L3_raster.lines[i].instrument```  Return the instrument  of the line i.

```L3_raster.lines[i].ion```         Return the ion of the line i.

```L3_raster.lines[i].line_id```     Return the line_id of the line i.  

```L3_raster.lines[i].obs_date```     Return the observation dat of the line i.  

```L3_raster.lines[i][par]``` 2D Return array parameter of line i with par in `['int' and/or 'wav' and/or 'wid' and/or 'rad' and/or 'int_err' and/or 'wav_err' and/or 'wid_err' and/or 'rad_err' ]`.

```L3_raster.lines[i].header[par]``` Return astropy fits header object of line i with par in `['int' and/or 'wav' and/or 'wid' and/or 'int_err' and/or 'wav_err' and/or 'wid_err' ]` (no "rad" neither "rad_err").

```L3_raster.lines[i].plot(params='all',axes =None,add_keywords = False)``` Plot a parameter or a set of parameters `['int' and/or 'wav' and/or 'wid' and/or 'rad' and/or 'int_err' and/or 'wav_err' and/or 'wid_err' and/or 'rad_err' ]`. If axes is not None than it should be a 1D Iterable with size equal to the number of parameters to plot.

```L3_raster.lines[i].get_map(param)``` get generic map object of the selected parameter 

```L3_raster.lines[i].correct_doppler_gradient(self, direction="x", verbose=0, coeff=None) -> coeffs, errors```

This method performs a primitive Doppler gradient correction on line `i`. The corrected Doppler values are automatically applied to the line. To revert to the original Doppler values, use:
```L3_raster.lines[i].reset_doppler()```
- **Direction**: Can be `'x'`, `'y'`, or `'xy'`. Specifies the direction(s) for the gradient correction.
- **coeff Argument**: If the coefficients are already known, provide them using this argument to apply the gradient correction without recomputing the coefficients.
- **Output**: Returns a list of coefficients for each direction and the computed errors on the Doppler fit.

```L3_raster.get_coord_mat(as_skycoord=False) -> coord_matrix``` or ```L3_raster.lines[i].get_coord_mat(as_skycoord=False) -> coord_matrix```

This method retrieves the coordinate matrix from the raster data.

- **as_skycoord**: If set to `True`, the coordinates will be returned as SkyCoord objects. Default is `False`.
- **Output**: Returns the coordinate matrix (longitude,latitude).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use SAFFRON in your research, please cite the following:

Slimane MZERGUAT. SAFFRON: Spectral Analysis Fitting Framework, Reduction Of Noise, Version [Used_version](https://github.com/slimguat/saffron-spice/tags), 2024. Available at: https://github.com/slimguat/saffron-spice.

But hey, if that sounds like too much work, feel free to skip it and just enjoy the magic of SAFFRON! 😉