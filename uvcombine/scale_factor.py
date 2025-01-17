
"""
Routines for determining the scale factor between single dish and
interferometric data.


"""

from astropy import units as u
from tqdm import tqdm
import numpy as np
from astropy import stats as astrostats
from scipy import stats
from astropy import log

from .uvcombine import feather_compare


def find_effSDbeam(hires, lores,
                   LAS,
                   lowresfwhms,
                   beam_divide_lores=True,
                   lowpassfilterSD=False,
                   min_beam_fraction=0.1,
                   alpha=0.85,
                   verbose=False):
    '''
    Find the optimal FWHM of the SD data by minimizing the relation between
    the ratios in the overlapping region

    See Sec. 3.2.1 in Stanimirovic (1999)
    https://ui.adsabs.harvard.edu/#abs/1999PhDT........21S/abstract

    Parameters
    ----------
    hires : np.ndarray
        Sample of high-resolution points.
    lores : np.ndarray
        Sample of low-resolution points. The order of `hires` and `lores`
        should match!
    lowresfwhms : `~astropy.units.Quantity`
        Values for the low-resolution FWHM to test. These should have an
        angular unit.
    beam_divide_lores: bool, optional
        See `uvcombine.feather_compare`.
    lowpassfilterSD : bool, optional
        See `uvcombine.fftmerge`.
    min_beam_fraction : float, optional
        See `uvcombine.feather_compare`.
    alpha : float between 0 and 1
        The confidence interval range for the uncertainties on the slopes.
    verbose : bool, optional
        Enables plotting.

    Returns
    -------
    slopes : np.ndarray
        Values of the slopes for the given `lowresfwhms`.
    slopes_CI : np.ndarray
        Upper and lower confidence intervals for the slopes. The confidence
        interval region is set by `alpha`.
    '''

    slopes = np.empty(lowresfwhms.size)
    slopes_CI = np.empty((2, lowresfwhms.size))

    for i, lowresfwhm in enumerate(tqdm(lowresfwhms)):
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
                      use_likelihood_fit=True,
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
    use_likelihood_fit : bool, optional
        When using `method='distrib'`, fit the distribution with a maximum likelihood method.
        This requires the statsmodels package to be installed. The main reason for this fitting
        method is to provide standard error estimates on the fit parameters.
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

        ratio = ratio[np.isfinite(ratio)]

        # Fit a Cauchy distribution to the log of the ratios
        log_ratio = np.log(ratio)

        params = stats.cauchy.fit(ratio)

        # Try to get standard errors from a Likelihood fit with statsmodels
        if use_likelihood_fit:
            try:
                import statsmodels

                mle_model = Likelihood(log_ratio)
                fitted_model = mle_model.fit(params, method='nm')
                fitted_model.df_model = len(ratio)
                fitted_model.df_resid = len(ratio) - 2

                params = fitted_model.params
                stderr = fitted_model.bse

            except ImportError:
                log.info("Unable to import statsmodels needed for the likelihood fit."
                         " Parameter error estimates cannot be calculated.")
                stderr = np.zeros_like(params)

        else:
                stderr = np.zeros_like(params)

        # The median is the scale factor.
        sc_factor = np.exp(params[0])
        sc_factor_stderr = stderr[0] * sc_factor

        if verbose:
            import matplotlib.pyplot as pl
            pl.hist(log_ratio, bins=int(np.sqrt(lowres_pts.size)), alpha=0.7,
                    density=True)
            x_vals = np.arange(-3, 3, 0.05)
            pl.plot(x_vals, stats.cauchy.pdf(x_vals, *params))
            pl.xlabel("log Ratio")

        return sc_factor, sc_factor_stderr

    elif method == "linfit":

        sc_factor, intercept, sc_lowlim, sc_highlim = \
            stats.theilslopes(highres_pts, x=lowres_pts, **method_kwargs)

        sc_confint = np.array([sc_lowlim, sc_highlim])

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


try:
    from statsmodels.base.model import GenericLikelihoodModel

    class Likelihood(GenericLikelihoodModel):

        # Get the number of parameters from shapes.
        # Add one for scales, since we're assuming loc is frozen.
        # Keeping loc=0 is appropriate for log-normal models.
        nparams = 1 if stats.cauchy.shapes is None else \
            len(stats.cauchy.shapes.split(",")) + 1

        def loglike(self, params):
            if np.isnan(params).any():
                return - np.inf

            loglikes = \
                stats.cauchy.logpdf(self.endog, *params[:-2],
                                    scale=params[-1],
                                    loc=params[-2])
            if not np.isfinite(loglikes).all():
                return - np.inf
            else:
                return loglikes.sum()

except ImportError:
    pass
