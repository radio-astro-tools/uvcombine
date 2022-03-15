import pytest

import astropy.units as u
from astropy.io import fits
import numpy.testing as npt
import numpy as np
from spectral_cube import Projection, SpectralCube

from ..uvcombine import (feather_simple, fourier_combine_cubes,
                         feather_simple_cube)


def test_feather_simple(plaw_test_data):


    orig_hdu, lowres_hdu, highres_hdu = plaw_test_data

    # HDU input
    combo = feather_simple(highres_hdu, lowres_hdu)

    # Projection input
    lowres_proj = Projection.from_hdu(lowres_hdu)
    highres_proj = Projection.from_hdu(highres_hdu)

    combo = feather_simple(highres_proj, lowres_proj)

    # Assert the combined data is sufficiently close to the original
    orig_data = orig_hdu.data

    # This won't work on a pixel basis as we can't exclude outliers
    # npt.assert_allclose(orig_data, combo.real)

    # Test against flux recovery
    frac_diff = (orig_data - combo.real).sum() / orig_data.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)


def test_feather_simple_mismatchunit(plaw_test_data):

    orig_hdu, lowres_hdu, highres_hdu = plaw_test_data

    highres_hdu.header['BUNIT'] = "Jy/beam"
    lowres_hdu.header['BUNIT'] = "K"

    with pytest.raises(ValueError):
        combo = feather_simple(highres_hdu, lowres_hdu, match_units=False)


def test_fourier_combine_cubes(plaw_test_cube_sc):

    orig_cube, sd_cube, interf_cube = plaw_test_cube_sc

    combo_cube = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)

    combo_cube_sc = SpectralCube.read(combo_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)


def test_fourier_combine_cubes_diffunits(plaw_test_cube_sc):

    orig_cube, sd_cube, interf_cube = plaw_test_cube_sc

    interf_cube = interf_cube.to(u.Jy / u.beam)

    combo_cube = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)

    combo_cube_sc = SpectralCube.read(combo_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    # Output units should in the units of the interferometer cube
    assert combo_cube_sc.unit == interf_cube.unit

    combo_cube_sc_smunits = combo_cube_sc.to(orig_cube.unit)

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc_smunits).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)


def test_feather_simple_cube(plaw_test_cube_sc):

    orig_cube, sd_cube, interf_cube = plaw_test_cube_sc

    combo_cube_sc = feather_simple_cube(interf_cube, sd_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)


def test_feather_simple_cube_diffunits(plaw_test_cube_sc):

    orig_cube, sd_cube, interf_cube = plaw_test_cube_sc

    interf_cube = interf_cube.to(u.Jy / u.beam)

    combo_cube_sc = feather_simple_cube(interf_cube, sd_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    # Output units should in the units of the interferometer cube
    assert combo_cube_sc.unit == interf_cube.unit

    combo_cube_sc_smunits = combo_cube_sc.to(orig_cube.unit)

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc_smunits).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)
