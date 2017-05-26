
"""
Routines for determining the scale factor between single dish and
interferometric data.


"""

import image_tools
import radio_beam
from reproject import reproject_interp
from spectral_cube import SpectralCube
from spectral_cube import wcs_utils
from astropy.io import fits
from astropy import units as u
from astropy import log
from astropy.utils.console import ProgressBar
import numpy as np
import FITS_tools
#from FITS_tools.hcongrid import hcongrid_hdu
#from FITS_tools.cube_regrid import regrid_cube_hdu
from astropy import wcs, stats
from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy import stats


from .uvcombine import feather_compare


def find_effSDbeam(hires, lores,
                   LAS,
                   lowresfwhms,
                   beam_divide_lores=True,
                   highpassfilterSD=False,
                   min_beam_fraction=0.1,
                   alpha=0.85,
                   verbose=False):
    '''
    Find the optimal FWHM of the SD data by minimizing the relation between
    the ratios in the overlapping region

    See Sec. 3.2.1 in Stanimirovic (1999)
    https://ui.adsabs.harvard.edu/#abs/1999PhDT........21S/abstract

    '''

    slopes = np.empty(lowresfwhms.size)
    slopes_CI = np.empty((2, lowresfwhms.size))

    for i, lowresfwhm in enumerate(ProgressBar(lowresfwhms)):
        out = feather_compare(hires, lores,
                              SAS=lowresfwhm,
                              LAS=LAS,
                              lowresfwhm=lowresfwhm,
                              return_ratios=True,
                              doplot=False)

        radii = out[0].to(u.karcsec)
        ratios = out[1]

        fitted = stats.theilslopes(ratios, radii.value**2,
                                   alpha=alpha)

        slopes[i] = fitted[0]
        slopes_CI[0, i] = fitted[2]
        slopes_CI[1, i] = fitted[3]

    if verbose:
        import matplotlib.pyplot as plt

        plt.errorbar(lowresfwhms.to(u.arcsec).value, slopes,
                     yerr=[slopes - slopes_CI[0],
                           slopes_CI[1] - slopes])
        plt.axhline(0)
        plt.ylabel("Slope")
        plt.xlabel("Low Res. FWHM (arcsec)")

    return slopes, slopes_CI
