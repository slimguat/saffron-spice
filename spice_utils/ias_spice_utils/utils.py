import os
import json
from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import SqrtStretch, AsymmetricPercentileInterval, ImageNormalize
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

__all__ = [
           'unique_headers',
           'full_file_path',
           'read_uio_cat',
           'make_summary',
           'read_studies_files_list_for_stp',
           'write_studies_files_list_for_stp',
           'make_notebook_from_template',
           'make_notebooks_for_stp',
           'process_notebooks_for_stp'
          ]


def extract_keyvals(header, keys):
    """
    Returns a subset of header key:value pairs
    """
    return {k: header[k] for k in keys}


def match_header(header, keyvals):
    """
    Returns True if dictionary of {keys:value} pairs is in header
    """
    if keyvals is None:
        return True
    header_keyvals = extract_keyvals(header, keyvals)
    return all(v == keyvals[k] for k, v in header_keyvals.items())


def unique_headers(headers, keys=None, keys_only=False):

    checked_keys = [k for k in headers[0]] if keys is None else keys
    seen_set = set()
    uniq = []
    for h in headers:
        checked_keyvals = {k: h[k] for k in checked_keys}
        if all(not match_header(dict(zip(checked_keys, keyvals)), checked_keyvals) for keyvals in seen_set):
            if keys_only:
                uniq.append(checked_keyvals)
            else:
                uniq.append(h)
            seen_set.add(tuple([h[k] for k in checked_keys]))
    return uniq


def full_file_path(file_name, base):
    splt = file_name.split('_')
    level = 'level' + splt[1][1]
    year = splt[3][0:4]
    month = splt[3][4:6]
    day = splt[3][6:8]
    return str(Path(base) / 'fits' / level / year / month / day / file_name)


def read_uio_cat(data_path):
    """
    Read UiO text table SPICE FITS files catalog
    http://astro-sdc-db.uio.no/vol/spice/fits/spice_catalog.txt

    Return
    ------
    pandas.DataFrame
        Table

    Example queries that can be done on the result:

    * `df[(df.LEVEL == "L2") & (df["DATE-BEG"] >= "2020-11-17") & (df["DATE-BEG"] < "2020-11-18") & (df.XPOSURE > 60.)]`
    * `df[(df.LEVEL == "L2") & (df.STUDYDES == "Standard dark for cruise phase")]`
    """
    cat_file = Path(data_path) / "fits" / "spice_catalog.txt"
    columns = list(pd.read_csv(cat_file, nrows=0, low_memory=False).keys())
    date_columns = ['DATE-BEG', 'DATE', 'TIMAQUTC']
    df = pd.read_table(cat_file, skiprows=1, names=columns, na_values="MISSING",
                       parse_dates=date_columns, warn_bad_lines=True,low_memory=False )
    df.LEVEL = df.LEVEL.apply(lambda string: string.strip())
    df.STUDYTYP = df.STUDYTYP.apply(lambda string: string.strip())
    return df


def write_studies_files_list_for_stp(stp_number, data_path, out_path):
    """
    Writes to files_list.txt a list of studies ran during a given STP and the corresponding file names.
    The file format is:
        STUDY1
        file1.fits
        file2.fits
        ....
        STUDY2
        file3.fits
        file4.fits
        ....

    :param stp_number: string, in the format XXX
    :param catalogue_path: location of the UiO catalogue
    :param out_path: location of output files_list.txt file
    :return: nothing
    """

    df = read_uio_cat(data_path)
      
    l2 = df[(df.LEVEL == "L2") &
            (df["STP"] == stp_number)]
    
    if l2['STUDY'].size == 0:
        print("An error occured")
        raise Exception ("there is nothing on STP{}".format(stp_number))
        return 1 #error

    stp_dir = "STP"+str(stp_number)
    out_path = Path(out_path) / stp_dir
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / 'files_list.txt', 'w') as f:
        for study in l2['STUDY'].unique():
            f.write("%s\n" % study)
            for file_name in l2[l2["STUDY"] == study]['FILENAME'].values:
                f.write("%s\n" % file_name)
    return 0 #ok


def read_studies_files_list_for_stp(file='files_list.txt'):
    """
    Reads in content of files_list.txt (or otherwise specified) created by write_studies_files_list_for_stp into
    a dictionary of STUDY_NAMES:list_of_files.

    :return: dictionary. Each key is a study_name, values are lists of files for each corresponding study.
    """
    studies_files = {}
    with open(file, 'r') as f:
        current_study = f.readline().strip()
        if current_study != '':  # null string should occur only for empty files_lists.txt
            studies_files[current_study] = []
            for line in f:
                line = line.strip()
                if line.endswith('.fits'):
                    studies_files[current_study].append(line)
                else:
                    current_study = line
                    studies_files[current_study] = []
    return studies_files


def make_notebook_from_template(study, stp_number, out_path):
    """
    Creates a Jupyter notebook from the template file referenced by NOTEBOOK_TEMPLATE

    :param study: string, study name
    :param stp_number: string, in the format XXX
    :param out_path: location of output file
    :return: nothing
    """
    out_path.mkdir(parents=True, exist_ok=True)
    notebook_name = study + '.ipynb'
    output_notebook = out_path / notebook_name

    default = Path(__file__).resolve().parent / 'STUDY-NAME_template.ipynb'
    notebook_template = os.getenv('NOTEBOOK_TEMPLATE',default=default)

    with open(notebook_template, 'r') as file:
        json_data = json.load(file)
        for item in json_data['cells']:
            item['source'] = [src.replace('STUDY-NAME', study) for src in item['source']]

    with open(output_notebook, 'w') as file:
        json.dump(json_data, file)


def make_notebooks_for_stp(stp_number):
    """
    Creates all Jupyter notebooks for a given STP

    :param stp_number: string, in the format XXX
    :return: nothing
    """

    notebooks_root = os.getenv("NOTEBOOKS_ROOT",default='/home/nb/pre-analysis')
    data_path = os.getenv('SPICE_ARCHIVE_DATA_PATH', default='/archive/SOLAR-ORBITER/SPICE')

    rtc = write_studies_files_list_for_stp(stp_number, data_path, notebooks_root)
    if rtc:
        print('no study found for this STP ', stp_number)
        return 1 
    stp_name = 'STP' + stp_number
    studies_file = Path(notebooks_root) / stp_name / 'files_list.txt'
    for study in read_studies_files_list_for_stp(studies_file).keys():
        stp_dir = Path(notebooks_root) / stp_name
        make_notebook_from_template(study, stp_number, stp_dir)
    return 0

def process_notebooks_for_stp(stp_number):
    stp_name = 'STP' + stp_number
    notebooks_root = os.getenv("NOTEBOOKS_ROOT",default='/home/nb/pre-analysis')
    path = Path(notebooks_root) / stp_name
    notebook_files = glob.glob(str(path / '*.ipynb'))
    for notebook_file in notebook_files:
        with open(notebook_file) as file:
            nb = nbformat.read(file, as_version=4)
            ep = ExecutePreprocessor(timeout=1200)
            try:
                ep.preprocess(nb, {'metadata': {'path': str(path)}})
            except CellExecutionError:
                msg = 'Error executing the notebook "%s".\n\n' % notebook_file
                msg += 'See notebook "%s" for the traceback.' % notebook_file
                print(msg)
                raise
            finally:
                file.close() # open in read mode
                file_w = open(notebook_file,"w")
                nbformat.write(nb, file_w)
                file_w.close()
                # je ne trouve pas comment faire autrement
                cmd = "jupyter nbconvert --to markdown "+ notebook_file
                os.system(cmd)
                # remove notebookfile
                os.remove(notebook_file)

def unique_windows(raster):
    """
    Returns a list of unique windows

    :param raster: either a SpectrogramCube read by sunraster or a dict of FITS HDUs
                   (temporary solution until sunraster can read in wide slit rasters).
    :return: list of unique windows
    """
    if type(raster) is dict:  # dict type is used only temporarily for slot rasters
        headers = [raster[w].header for w in raster]
    else:
        headers = [raster[w].meta.original_header for w in raster]
    uniq = unique_headers(headers, keys=['PXEND3', 'PXBEG3'])
    return [u['EXTNAME'] for u in uniq]


def make_summary(raster, filename, n_columns=3, figsize=(12, 8)):
    unq = unique_windows(raster)
    n_windows = len(unq)
    if n_windows <= n_columns:
        n_columns = n_windows
    n_rows = n_windows // n_columns + (n_windows % n_columns > 0)
    fig, axs = plt.subplots(n_rows, n_columns, sharex=True, sharey=True, squeeze=False, figsize=figsize)
    fig.suptitle(filename)
    try: 
        for ax in axs.flatten():
            ax.axis('off')
    except:
        print('filename ', filename, ' AttributeError: AxesSubplot object has no attribute flatten')
        return

    for iw, kw in enumerate(unq):
        window = raster[kw]
        ax = axs.flat[iw]

        if window.data.shape[3] == 1:
            data = np.nanmean(window.data, axis=(0, 3)).T
        else:
            if 'w-ras' in filename:  # wide raster -> put windows side by side
                splt = np.split(np.moveaxis(window.data, 2, 1), window.data.shape[3], axis=3)
                data = np.concatenate(splt, axis=2).squeeze()
            else:  # regular (narrow slits) raster
                data = np.nanmean(window.data, axis=(0, 1))

        norm = ImageNormalize(data,
                              interval=AsymmetricPercentileInterval(1, 99),
                              stretch=SqrtStretch())
        im = ax.imshow(data, origin="lower", norm=norm, aspect='auto')

        plt.colorbar(im, ax=ax)

        ax.set_title(kw)
        ax.axis('on')
    plt.tight_layout()
    plt.show()
