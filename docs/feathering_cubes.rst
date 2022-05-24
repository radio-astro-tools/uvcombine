.. _feathercubes:

Feathering two cubes
====================

Two cubes can be feathered with the `~uvcombine.feather_simple_cube`.
This function handles a variety of operations to prepare for two cubes to be feathered,
including pre-processing checks (spatial/spectral reprojection, SD beam reading, units).
The same feathering settings used for 2D images in `~uvcombine.feather_simple` can be
passed to `~uvcombine.feather_simple_cube`.

    >>> from uvcombine import feather_simple_cube
    >>> from spectral_cube import SpectralCube
    >>> highres_cube = SpectralCube.read("highres.fits")  # doctest: +SKIP
    >>> lowres_cube = SpectralCube.read("lowres.fits")  # doctest: +SKIP
    >>> feathered_cube = feather_simple_cube(highres_cube, lowres_cube)  # doctest: +SKIP

The defaults settings in `~uvcombine.feather_simple_cube` match those used by CASA's
`feather task <https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.imaging.feather.html>`_.
For more information on the feathering settings available, see :ref:`featherimages`.

For large cubes, the `use_memmap` option can be enabled to avoid keeping the feathered
cube in memory.

.. note:: When there is a large difference in the spectral channels between the two cubes,
    we recommend following the `spectral-cube smoothing guide <https://spectral-cube.readthedocs.io/en/latest/smoothing.html#spectral-smoothing>`
    to appropriately smooth the higher spectral resolution data prior to interpolation prior to using
    `~uvcombine.feather_simple_cube`, as this only applies the interpolation step.

If the cubes you are using have already been appropriately smoothed and/or reprojected, you can avoid
unnecessary pre-processing using the kwargs for `~uvcombine.feather_simple_cube`. To avoid
spectral interpolation of the low resolution data::

    >>> feathered_cube = feather_simple_cube(highres_cube, lowres_cube, allow_spectral_resample=False)  # doctest: +SKIP

To avoid reprojecting the low resolution data::

    >>> feathered_cube = feather_simple_cube(highres_cube, lowres_cube, allow_lo_reproj=False)  # doctest: +SKIP

In both cases, consistency checks are applied and a `ValueError` describing any discrepancies will be
returned.

Feathering large cubes using dask
---------------------------------

When dask is installed, we can use the dask-integration in spectral-cube to
parallelize and load the cubes in chunk to avoid high memory use.
See the `spectral-cube documentation <https://spectral-cube.readthedocs.io/en/latest/dask.html>`
on using dask for more information.

To enable using dask, load the spectral-cubes in dask mode::

    >>> highres_cube = SpectralCube.read("highres.fits", use_dask=True)  # doctest: +SKIP
    >>> lowres_cube = SpectralCube.read("lowres.fits", use_dask=True)  # doctest: +SKIP
    >>> feathered_cube = feather_simple_cube(highres_cube, lowres_cube)  # doctest: +SKIP

This will automatically enable using dask mode in `~uvcombine.feather_simple_cube`.

A required pre-processing step is to reproject `lowres_cube` to the same pixel
grid as `highres_cube` and match the brightness units. As this can be computationally
expensive, the spectral-cube dask integration exposes an option to save temporary
intermediate products as zarr files. This is especially useful when rechunking the
data to optimize different computations (e.g., spectral versus spatial regridding.)
To enable this mode, `use_save_to_tmp_dir <https://spectral-cube.readthedocs.io/en/latest/dask.html#saving-intermediate-results-to-disk>`_
can be enabled in `~uvcombine.feather_simple_cube`::

    >>> feathered_cube = feather_simple_cube(highres_cube, lowres_cube, use_save_to_tmp_dir=True)  # doctest: +SKIP

If the cubes already have matching chunk size, you can avoid rechunking the cubes with::

    >>> feathered_cube = feather_simple_cube(highres_cube, lowres_cube, force_spatial_rechunk=False)  # doctest: +SKIP


Previous functionality
----------------------

A similar function `~uvcombine.fourier_combine_cubes` was previously implemented
in uvcombine. Its use is now depcrecated as it lacks new features implemented in
`~uvcombine.feather_simple_cube`. This method is appropriate for two cubes that each
have a single beam size (i.e., the beam does not change between spectral channels).
The low resolution beam size is also set using `lowresfwhm`, and is NOT read from
the FITS header.

