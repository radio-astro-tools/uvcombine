import pytest

import astropy.units as u
from astropy.io import fits
import numpy.testing as npt
import numpy as np
from spectral_cube import Projection

from .utils import testing_data
from ..uvcombine import feather_simple


def test_feather_simple():


    orig_hdu, lowres_hdu, highres_hdu = testing_data(return_images=True)

    # HDU input
    combo = feather_simple(lowres_hdu, highres_hdu)

    # Projection input
    lowres_proj = Projection.from_hdu(lowres_hdu)
    highres_proj = Projection.from_hdu(highres_hdu)

    combo = feather_simple(lowres_proj, highres_proj)


def test_feather_simple_mismatchunit():

    orig_hdu, lowres_hdu, highres_hdu = testing_data(return_images=True)

    highres_hdu.header['BUNIT'] = "Jy/beam"
    lowres_hdu.header['BUNIT'] = "K"

    with pytest.raises(ValueError):
        combo = feather_simple(lowres_hdu, highres_hdu, match_units=False)


def test_feather_simple_cube():
    pass