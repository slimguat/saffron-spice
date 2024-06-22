
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

## Installation

To install the module, follow these steps:

1. Clone the repository:
    \`\`\`bash
    git clone <repository-url>
    \`\`\`

2. Navigate to the project directory:
    \`\`\`bash
    cd saffron-spice
    \`\`\`

3. Install the dependencies:
    \`\`\`bash
    pip install -r requirements.txt
    \`\`\`

## Usage

Provide examples of how to use the main functionalities of your module. Include code snippets for initializing and using classes and functions.

### Example: Fitting a Pixel

\`\`\`python
import numpy as np
from saffron.fit_functions import fit_pixel

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Perform fit
params = fit_pixel(x, y)
print(params)
\`\`\`

### Example: Fitting a Raster

\`\`\`python
from saffron.fit_functions import fit_raster

# Example raster data
raster_data = np.random.rand(100, 100)

# Perform raster fit
raster_fit = fit_raster(raster_data)
print(raster_fit)
\`\`\`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/fooBar`).
3. Commit your changes (`git commit -am 'Add some fooBar'`).
4. Push to the branch (`git push origin feature/fooBar`).
5. Create a new Pull Request.

## Acknowledgements

Special thanks to all contributors and the open-source community for their invaluable support and contributions.
