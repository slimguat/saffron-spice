import os
import platform

def find_xuvtop():
    xuvtop = os.getenv('XUVTOP')

    if xuvtop:
        print(f"XUVTOP is already set to: {xuvtop}")
    elif platform.system() == 'Linux':
        bashrc_path = os.path.expanduser('~/.bashrc')
        if os.path.exists(bashrc_path):
            with open(bashrc_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'export XUVTOP' in line:
                        xuvtop_value = line.split('=')[1].strip().strip('"')
                        os.environ['XUVTOP'] = xuvtop_value
                        print(f"XUVTOP found in .bashrc and set to: {xuvtop_value}")
                        return
        print("XUVTOP not found in environment variables or .bashrc.")
    else:
        print("XUVTOP not found and manual search is only supported on Linux.")

find_xuvtop()

from . import fit_functions
from . import fit_models
from . import init_handler
from . import line_catalog
from . import manager
from . import postprocessing
from . import utils

__all__ = (
    fit_functions,
    fit_models,
    init_handler,
    line_catalog,
    manager,
    postprocessing,
    utils,
)

