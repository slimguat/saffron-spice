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


## Tutorial
...
