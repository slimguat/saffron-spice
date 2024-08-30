from .gauss import (
    Gauss,
    multiGauss,
    flat_inArg_multiGauss,
    flat_multiGauss,
    gauss_Fe18_C3,
    gauss_LyB_Fe10,
)
from .Model import ModelFactory
print("fit_models initiated")
__all__ = (
    "Gauss",
    "multiGauss",
    "flat_inArg_multiGauss",
    "flat_multiGauss",
    "gauss_Fe18_C3",
    "gauss_LyB_Fe10",
    'ModelFactory',
)
