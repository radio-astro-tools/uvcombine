.. _diagnostics:

Diagnostics and consistency checks
===================================

Before (or after) feathering two images, it is useful to check that the
high- and low-resolution data agree with each other over the range of
spatial scales where their uv-coverage overlaps. uvcombine provides two
functions for this.

Comparing flux scales in the uv-overlap region
------------------------------------------------

`~uvcombine.feather_compare` compares the Fourier amplitudes of the
high- and low-resolution data over a user-specified range of angular
scales (the region where both data sets should be sensitive to the same
structures), and reports the ratio between them. A ratio far from 1
indicates a flux-scale mismatch between the two data sets that should be
corrected (for example with ``lowresscalefactor``/``highresscalefactor``
in `~uvcombine.feather_simple`) before feathering::

    >>> from uvcombine import feather_compare
    >>> from astropy import units as u
    >>> stats = feather_compare(highres_image, lowres_image,  # doctest: +SKIP
    ...                          SAS=5 * u.arcsec, LAS=30 * u.arcsec,
    ...                          lowresfwhm=30 * u.arcsec)

``SAS`` and ``LAS`` set the smallest and largest angular scales to
include in the comparison; ``LAS`` should typically correspond to the
largest angular scale recoverable by the interferometric data. By
default a diagnostic plot is produced (``doplot=True``); set
``return_samples=True`` to instead get back the individual per-pixel
samples (angular scale, ratio, and high-/low-resolution amplitudes) in
the overlap region rather than the summary statistics.

`ScaleFactors.ipynb <https://github.com/radio-astro-tools/uvcombine/blob/main/examples/ScaleFactors.ipynb>`_
compares several methods for finding this scale factor from the
uv-overlap region.

Plotting the power spectra
---------------------------

`~uvcombine.feather_plot` plots the azimuthally-averaged power spectra
of the high- and low-resolution images together with the feathering
weight kernels, which is useful for visually inspecting where the two
data sets are combined and how much weight each contributes as a
function of spatial scale::

    >>> from uvcombine import feather_plot
    >>> results = feather_plot(highres_image, lowres_image)  # doctest: +SKIP

It accepts the same ``lowresfwhm``, ``lowresscalefactor``,
``highresscalefactor``, and ``lowpassfilterSD`` options as
`~uvcombine.feather_simple` (see :ref:`featherimages`), plus
``hires_threshold``/``lores_threshold`` to exclude noise-dominated
pixels below a given value before computing the power spectra.
