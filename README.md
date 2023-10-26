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
```
import os

# Set the XUVTOP environment variable, will be gone by the end of the script
os.environ['XUVTOP'] = 'path/to/your/database'
```
## Install
1. Install the right version of python 

```
pyenv install 3.9.15
```
2. Create and activate your environment

```
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

1. Fastest way 

The odds is that you will need no major adjustments for the fitting as the input parameters are most likely tuned out.
In the beginning you will need a input json file 
...
