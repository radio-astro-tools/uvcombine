.. _featherimages:

Feathering two images
=====================

`~uvcombine.feather_simple` is the primary function for feathering.

Images can be passed as a FITS filename (as a string), FITS HDU, or a `~spectral_cube.Projection`.
The latter is recommended to allow close check of the beam sizes passed for the high- and
low-resolution data.

In the simplest case, a low- and high-resolution image can be feathered with::

    >>> from uvcombine import feather_simple
    >>> from astropy.io import fits
    >>> from spectral_cube import Projection
    >>> highres_image = Projection.from_hdu(fits.open("highres.fits"))  # doctest: +SKIP
    >>> lowres_image = Projection.from_hdu(fits.open("lowres.fits"))  # doctest: +SKIP
    >>> feathered_image = feather_simple(highres_image, lowres_image)  # doctest: +SKIP

The defaults settings in `~uvcombine.feather_simple` match those used by CASA's
`feather task <https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.imaging.feather.html>`_.

`~uvcombine.feather_simple` has many options to alter the handling, uv-cutoff, or weighting of the two
images when combining::

    * Flux scaling factors to multiple the data by before combining (`lowresscalefactor`; `highresscalefactor`)
    * `pbresponse` allows a numpy array of the primary beam response of the interferometer to be applied to the low resolution data.
    * `lowresfwhm` overrides the beam size in the low resolution data.
    * `lowpassfilterSD` filters high spatial frequenceis in the low resolution image by its beam. Similar to `lowpassfiltersd` in CASA.
    * `replace_hires` will replace the high spatial frequencies of the feathered image above a set threshold in the low resolution beam kernel, rather than combining by the weighting kernel.
    * `deconvSD` will deconvolve the low resolution data by its beam before combining the data.
    * `weights` allows a 2D numpy array matching the high-resolution image size to be used as custom weighting, similar to the `pbresponse`. This can be used to taper the edges of images to avoid Gibbs ringing.

The impact of these many options is explored in depth in `this tutorial <https://github.com/radio-astro-tools/uvcombine/blob/master/examples/FeatheringTests.ipynb>`_.

