from spice_uncertainties import spice_error
from astropy.io import fits


def get_spice_errors(input_file, verbose=False):
    with fits.open(input_file) as hdulist:  # specify file name here
        windows = [hdu for hdu in hdulist if hdu.is_image and hdu.name != "WCSDVARR"]
        sigmas = []
        for window in windows:
            av_constant_noise_level, sigma = spice_error(window, verbose=verbose)
            sigmas.append(sigma["Total"].value)
    return sigmas
