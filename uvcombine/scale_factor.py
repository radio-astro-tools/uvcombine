
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
from astropy import stats as astrostats
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
                              return_samples=True,
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


def find_scale_factor(lowres_pts, highres_pts, method='distrib',
                      verbose=False,
                      **method_kwargs):
    '''
    Using overlapping points in the uv-plane, find the
    scale factor. There are three methods implemented:

        * 'distrib' -- fits a Cauchy distribution to the ratio of the points.
            The center of the distribution is fit, giving the scale factor.
        * 'linfit' -- fit a robust linear model between the low- and high-res
            points. This uses `Theil-Sen regression <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.theilslopes.html>`_
            to find the slope, which is the scale factor.
        * 'clippedstats' -- Uses `~astropy.stats.sigma_clipped_stats` to
            estimate the scale factor with outlier rejection. Uses the
            `astropy implementation <http://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipped_stats.html#astropy.stats.sigma_clipped_stats>`_

    Parameters
    ----------
    lowres_pts : `~numpy.ndarray` or `~astropy.units.Quantity`
        Points from the uv-overlap region for the low-resolution data.
    highres_pts : `~numpy.ndarray` or `~astropy.units.Quantity`
        Points from the uv-overlap region for the high-resolution data.
    method : {'distrib', 'linfit', 'clippedstats'}, optional
        Specify the method to for estimating the scale factor.
    verbose : bool, optional
        Enables plotting of the data and the scale factor relation.
    method_kwargs : Passed to `~scipy.stats.theilslopes` for 'linfit' and
        `~astropy.stats.sigma_clipped_stats` for 'clippedstats'. Not used by
        'distrib'.

    Returns
    -------
    sc_factor : float
        The scale factor returned by 'distrib' and 'linfit'.
    sc_confint : list
        The confidence interval for the scale factor using 'linfit'.
    out_dict : dict
        Returned by 'clippedstats'. Contains the clipped statistic estimates
        of the mean, median, and standard deviation.
    '''

    if lowres_pts.size != highres_pts.size:
        raise ValueError("lowres_pts must be the same size as highres_pts.")

    # Drop the units if Quantities are given.
    if hasattr(lowres_pts, "value"):
        lowres_pts = lowres_pts.value
    if hasattr(highres_pts, "value"):
        highres_pts = highres_pts.value

    if method == "distrib":

        ratio = highres_pts / lowres_pts

        # Fit a Cauchy distribution to the log of the ratios=
        log_ratio = np.log(ratio)

        params = stats.cauchy.fit(ratio)

        # The median is the scale factor.
        sc_factor = np.exp(params[0])

        if verbose:
            import matplotlib.pyplot as pl
            pl.hist(log_ratio, bins=int(np.sqrt(lowres_pts.size)), alpha=0.7,
                    normed=True)
            x_vals = np.arange(-3, 3, 0.05)
            pl.plot(x_vals, stats.cauchy.pdf(x_vals, *params))
            pl.xlabel("log Ratio")

        return sc_factor

    elif method == "linfit":

        sc_factor, intercept, sc_confint = \
            stats.theilslopes(highres_pts, x=lowres_pts, **method_kwargs)

        if verbose:
            import matplotlib.pyplot as pl
            min_fit_pt = lowres_pts.min() * sc_factor + intercept
            max_fit_pt = lowres_pts.max() * sc_factor + intercept
            pl.scatter(lowres_pts, highres_pts, alpha=0.7)
            pl.plot([lowres_pts.min(), lowres_pts.max()],
                    [min_fit_pt, max_fit_pt])

        return sc_factor, sc_confint

    elif method == "clippedstats":

        ratio = highres_pts / lowres_pts

        sclip = astrostats.sigma_clipped_stats(ratio, **method_kwargs)

        out_dict = {'scale_factor_mean': sclip[0],
                    'scale_factor_median': sclip[1],
                    'scale_factor_std': sclip[2]}

        if verbose:
            import matplotlib.pyplot as pl
            min_fit_pt = lowres_pts.min() * sclip[1]
            max_fit_pt = lowres_pts.max() * sclip[1]
            pl.scatter(lowres_pts, highres_pts, alpha=0.7)
            pl.plot([lowres_pts.min(), lowres_pts.max()],
                    [min_fit_pt, max_fit_pt])

        return out_dict

    else:
        raise ValueError("method must be 'distrib', 'linfit', or "
                         "'clippedstats'.")
