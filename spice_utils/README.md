# IAS SPICE UTILS

Basic tools for handling SPICE data at IAS. Used e.g. for creating preview notebooks.

## pre-installation

pip3 install setuptools

## Installation

from spice_utils directory
     `python3 -m build`
     `pip3 install .`


## usage, making your own notebok update

Once installed as described above.

export or setenv `SPICE_ARCHIVE_DATA_PATH` (default is /archive/SOLAR-ORBITER/SPICE)
The directory structure under the root is assumed to follow that of the UiO repository, i.e. `SPICE_ARCHIVE_DATA_PATH/fits/level2/YYYY/MM/DD/spice_file.fits`.

export or setenv NOTEBOOKS_ROOT (default is /home/nb/pre-analysis)
Where the notebooks will be created

export or setenv NOTEBOOK_TEMPLATE (default current_dir/ias_spice_utils/STUDY-NAME_template.ipynb)
Template file to use.

The notebooks assume that the files corresponding to the relevant study are available locally.
(`SPICE_ARCHIVE_DATA_PATH` environment variable)


You must initially clone the repository using the links provided in the "Clone" menu above. Then each time you want to modify a notebook:

- Pull the repository
- Set SPICE_ARCHIVE_DATA_PATH variable to point to the root directory of your local SPICE archive
- Pull or git clone https://git.ias.u-psud.fr/spice/pre-analysis.git
- Set the NOTEBOOKS_ROOT to this repository

If notebooks for an STP do not exist yet, they can be generated with (example for STP173):

```python
from utils.utils import make_notebooks_for_stp
make_notebooks_for_stp('173')
```

or

- Make your modifications to existing notebook, 
- Push your modifications to the notebook repository

## ATTENTION
Changing this package may break the automatic generation of the notebook done for each complete STP
from pre-analysis package.
