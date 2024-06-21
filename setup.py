from setuptools import setup, find_packages
import os

# Read the requirements from requirement.txt
with open('requirement.txt') as f:
    requirements = f.read().splitlines()

# Read the contents of your README file
with open('README.md') as f:
    long_description = f.read()

# Filter out URL dependencies and handle them separately
url_dependencies = [req for req in requirements if '://' in req]
standard_dependencies = [req for req in requirements if '://' not in req]

setup(
    name='SAFFRON',
    version='1.0.0',
    description='SAFFRON: Spectral Analysis Fitting Framework, Reduction Of Noise.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Slimane MZERGUAT',
    author_email='slimane.mzerguat@u-psud.fr,mzerguat.sl@gmail.com',
    url='https://github.com/slimguat/SAFFRON',
    packages=find_packages(include=['SAFFRON', 'SAFFRON.*']),
    install_requires=standard_dependencies,
    dependency_links=url_dependencies,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'SAFFRON=SAFFRON.__main__:main',
        ],
    },
    include_package_data=True,
    package_data={
        'SAFFRON': [
            'line_catalog/SPICE_SpecLines.json',
            'init_handler/*.depr',
            'manager/input_config_template.json',
        ],
    },
)

# Install URL dependencies if they exist
if url_dependencies:
    os.system('pip install ' + ' '.join(url_dependencies))
