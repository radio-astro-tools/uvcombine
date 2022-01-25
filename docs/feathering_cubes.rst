.. _feathercubes:

Feathering two cubes
====================

.. note:: In progress...

.. warning:: The method desribed here is currently not tested as part of the continuous integration suite. Check your results carefully when using.

Two cubes can currently be combined with the `~uvcombine.fourier_combine_cubes`.
This method is appropriate for two cubes that each have a single beam size (i.e., the beam
does not change between spectral channels). The low resolution beam size is also set
using `lowresfwhm`, and is NOT currently read from the FITS header.

We anticipate adding more features for combining spectral-cubes with an updated function
and interface in the near future.
