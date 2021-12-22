# uvcombine

Tools for combining high-resolution images with missing large angular scales (Fourier-domain short-spacings) with low-resolution images containing the short/zero spacing.

The modules can be installed like a normal python project:

    python setup.py install
    
If you have difficulty, refer to [astropy's installation documentation](docs.astropy.org/en/stable/install.html), since we use the same infrastructure.

The main code is incompletely documented in `uvcombine/uvcombine.py`.

A good starting point to understanding what the code is doing and what uncertainties are involved in combination is this notebook:
https://github.com/radio-astro-tools/uvcombine/blob/master/examples/FeatheringTests.ipynb
