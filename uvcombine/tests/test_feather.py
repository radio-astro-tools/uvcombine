import pytest

import astropy.units as u
from astropy.io import fits
import numpy.testing as npt
import numpy as np
from spectral_cube import Projection, SpectralCube

from .utils import testing_data, testing_data_cube
from ..uvcombine import feather_simple, fourier_combine_cubes


def test_feather_simple():


    orig_hdu, lowres_hdu, highres_hdu = testing_data(return_images=True)

    # HDU input
    combo = feather_simple(highres_hdu, lowres_hdu)

    # Projection input
    lowres_proj = Projection.from_hdu(lowres_hdu)
    highres_proj = Projection.from_hdu(highres_hdu)

    combo = feather_simple(highres_proj, lowres_proj)


def test_feather_simple_mismatchunit():

    orig_hdu, lowres_hdu, highres_hdu = testing_data(return_images=True)

    highres_hdu.header['BUNIT'] = "Jy/beam"
    lowres_hdu.header['BUNIT'] = "K"

    with pytest.raises(ValueError):
        combo = feather_simple(highres_hdu, lowres_hdu, match_units=False)


def test_feather_simple_cube():

    orig_cube, sd_cube, interf_cube = testing_data_cube()

    combo_cube = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)

    combo_cube_sc = SpectralCube.read(combo_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit


def test_feather_simple_cube_diffunits():

    orig_cube, sd_cube, interf_cube = testing_data_cube()

    interf_cube = interf_cube.to(u.Jy / u.beam)

    combo_cube = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)

    combo_cube_sc = SpectralCube.read(combo_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    # Output units should in the units of the interferometer cube
    print(combo_cube_sc.unit, interf_cube.unit)

    assert combo_cube_sc.unit == interf_cube.unit
