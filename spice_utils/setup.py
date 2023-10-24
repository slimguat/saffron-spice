from setuptools import setup

NAME = 'ias_spice_utils'
DESCRIPTION = 'IAS SPICE utilities',
URL = ''
EMAIL = 'spice-ops@universite-paris-saclay.fr'
AUTHOR = 'SPICE IAS'
REQUIRES_PYTHON = '>=3.9.0'
VERSION = '0.0.1'

REQUIRED = [
    'numpy',
    'astropy',
    'pandas',
    'matplotlib',
    'nbconvert',
    'nbformat',
    'pillow',
    'sunraster'
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIRED,
    license='LGPL-v3',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
    zip_safe=False,
    ext_modules=None,
    packages=['ias_spice_utils'],
    package_data={'ias_spice_utils': ['STUDY-NAME_template.ipynb'], }
)
