import pytest

import astropy.units as u
from astropy.io import fits
import numpy.testing as npt
import numpy as np
from spectral_cube import Projection, SpectralCube

from ..uvcombine import feather_simple, fourier_combine_cubes


def test_feather_simple(plaw_test_data):


    orig_hdu, lowres_hdu, highres_hdu = plaw_test_data

    # HDU input
    combo = feather_simple(highres_hdu, lowres_hdu)

    # Projection input
    lowres_proj = Projection.from_hdu(lowres_hdu)
    highres_proj = Projection.from_hdu(highres_hdu)

    combo = feather_simple(highres_proj, lowres_proj)


def test_feather_simple_mismatchunit(plaw_test_data):

    orig_hdu, lowres_hdu, highres_hdu = plaw_test_data

    highres_hdu.header['BUNIT'] = "Jy/beam"
    lowres_hdu.header['BUNIT'] = "K"

    with pytest.raises(ValueError):
        combo = feather_simple(highres_hdu, lowres_hdu, match_units=False)


def test_feather_simple_cube(plaw_test_cube_sc):

    orig_cube, sd_cube, interf_cube = plaw_test_cube_sc

    combo_cube = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)

    combo_cube_sc = SpectralCube.read(combo_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit


def test_feather_simple_cube_hdu(plaw_test_cube_hdu):

    orig_hdu, sd_hdu, interf_hdu = plaw_test_cube_hdu

    combo_cube = fourier_combine_cubes(interf_hdu, sd_hdu, return_hdu=True)

    assert orig_hdu.shape == combo_cube.shape

    assert combo_cube.header['BUNIT'] == interf_hdu['BUNIT']


def test_feather_simple_cube_diffunits(plaw_test_cube_sc):

    orig_cube, sd_cube, interf_cube = plaw_test_cube_sc

    interf_cube = interf_cube.to(u.Jy / u.beam)

    combo_cube = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)

    combo_cube_sc = SpectralCube.read(combo_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    # Output units should in the units of the interferometer cube
    assert combo_cube_sc.unit == interf_cube.unit
