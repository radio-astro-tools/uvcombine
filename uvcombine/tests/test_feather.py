import pytest
import os

import astropy.units as u
from astropy.io import fits
import numpy.testing as npt
import numpy as np
from spectral_cube import Projection, SpectralCube, DaskSpectralCube

# Use sklearn to bring in metrics of image similarity
from skimage.metrics import structural_similarity, normalized_root_mse

from . import path

from ..uvcombine import (feather_simple, fourier_combine_cubes,
                         feather_simple_cube)


def cube_and_raw(filename, use_dask=None):
    if use_dask is None:
        raise ValueError('use_dask should be explicitly set')
    p = path(filename)
    if os.path.splitext(p)[-1] == '.fits':
        with fits.open(p) as hdulist:
            d = hdulist[0].data
        c = SpectralCube.read(p, format='fits', mode='readonly', use_dask=use_dask)
    elif os.path.splitext(p)[-1] == '.image':
        ia.open(p)
        d = ia.getchunk()
        ia.unlock()
        ia.close()
        ia.done()
        c = SpectralCube.read(p, format='casa_image', use_dask=use_dask)
    else:
        raise ValueError("Unsupported filetype")

    return c, d


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

    # Test against normalized_root_mse using the Euclidean metric
    # Roughly a fractional difference using the feathered data as source of "error"
    nmse = normalized_root_mse(orig_data, combo.real, normalization='euclidean')
    assert nmse < 3e-2

    # Test against structural similarity metric
    ssim = structural_similarity(orig_data, combo.real)
    assert ssim > 0.99


@pytest.mark.parametrize(('lounit', 'hiunit'),
                         ((u.K, u.Jy / u.beam),
                          (u.Jy / u.beam, u.K),
                          (u.MJy / u.sr, u.Jy / u.beam),
                          (u.Jy / u.beam, u.MJy / u.sr),
                          (u.MJy / u.sr, u.K),
                          (u.K, u.MJy / u.sr),
                          (u.Jy / u.pixel, u.MJy / u.sr),
                          (u.MJy / u.sr, u.Jy / u.pixel),
                          (u.Jy / u.pixel, u.Jy / u.beam),
                          (u.Jy / u.beam, u.Jy / u.pixel),
                          (u.Jy / u.pixel, u.K),
                          (u.K, u.Jy / u.pixel)))
def test_feather_simple_varyunits(plaw_test_data, lounit, hiunit):

    orig_hdu, lowres_hdu, highres_hdu = plaw_test_data

    orig_proj = Projection.from_hdu(orig_hdu)

    # Projection input
    lowres_proj = Projection.from_hdu(lowres_hdu).to(lounit)
    highres_proj = Projection.from_hdu(highres_hdu).to(hiunit)

    combo = feather_simple(highres_proj, lowres_proj, return_hdu=True)

    combo_proj = Projection.from_hdu(combo).to(orig_proj.unit)

    # Test against flux recovery
    frac_diff = (orig_proj - combo_proj).sum() / orig_proj.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)


def test_feather_simple_mismatchunit(plaw_test_data):

    orig_hdu, lowres_hdu, highres_hdu = plaw_test_data

    highres_hdu.header['BUNIT'] = "Jy/beam"
    lowres_hdu.header['BUNIT'] = "K"

    with pytest.raises(ValueError):
        combo = feather_simple(highres_hdu, lowres_hdu, match_units=False)


def test_fourier_combine_cubes(cube_data):

    orig_fname, sd_fname, interf_fname = cube_data

    orig_cube, orig_data = cube_and_raw(orig_fname, use_dask=False)
    sd_cube, sd_data = cube_and_raw(sd_fname, use_dask=False)
    interf_cube, interf_data = cube_and_raw(interf_fname, use_dask=False)

    combo_cube = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)

    combo_cube_sc = SpectralCube.read(combo_cube)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)


def test_fourier_combine_cubes_diffunits(cube_data):

    orig_fname, sd_fname, interf_fname = cube_data

    orig_cube, orig_data = cube_and_raw(orig_fname, use_dask=False)
    sd_cube, sd_data = cube_and_raw(sd_fname, use_dask=False)
    interf_cube, interf_data = cube_and_raw(interf_fname, use_dask=False)

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


def test_feather_simple_cube(cube_data, use_dask, use_memmap):

    orig_fname, sd_fname, interf_fname = cube_data

    orig_cube, orig_data = cube_and_raw(orig_fname, use_dask=use_dask)
    sd_cube, sd_data = cube_and_raw(sd_fname, use_dask=use_dask)
    interf_cube, interf_data = cube_and_raw(interf_fname, use_dask=use_dask)

    combo_cube_sc = feather_simple_cube(interf_cube, sd_cube,
                                        use_memmap=use_memmap,
                                        use_dask=use_dask)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)

    # Test against normalized_root_mse using the Euclidean metric
    # Roughly a fractional difference using the feathered data as source of "error"
    nmse = normalized_root_mse(orig_cube.unitless_filled_data[:],
                               combo_cube_sc.unitless_filled_data[:],
                               normalization='euclidean')
    assert nmse < 3e-2

    # Test against structural similarity metric
    # Compare channel vs. channel
    for ii in range(orig_cube.shape[0]):
        ssim = structural_similarity(orig_cube.unitless_filled_data[ii],
                                    combo_cube_sc.unitless_filled_data[ii],
                                    )
        assert ssim > 0.99


def test_feather_simple_cube_dask_consistency(cube_data):

    orig_fname, sd_fname, interf_fname = cube_data

    orig_cube, orig_data = cube_and_raw(orig_fname, use_dask=False)
    sd_cube, sd_data = cube_and_raw(sd_fname, use_dask=False)
    interf_cube, interf_data = cube_and_raw(interf_fname, use_dask=False)

    orig_cube_dask = cube_and_raw(orig_fname, use_dask=True)[0]
    sd_cube_dask = cube_and_raw(sd_fname, use_dask=True)[0]
    interf_cube_dask = cube_and_raw(interf_fname, use_dask=True)[0]

    combo_cube_sc = feather_simple_cube(interf_cube, sd_cube)

    combo_cube_sc_dask = feather_simple_cube(interf_cube_dask, sd_cube_dask)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit
    assert combo_cube_sc_dask.unit == interf_cube.unit

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)

    frac_diff = (orig_cube - combo_cube_sc_dask).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)

    combo_cube_sc_data = combo_cube_sc.unitless_filled_data[:]
    combo_cube_sc_dask_data = combo_cube_sc_dask.unitless_filled_data[:]

    npt.assert_allclose(combo_cube_sc_data, combo_cube_sc_dask_data, atol=1e-5)


def test_feather_simple_cube_diffunits(cube_data, use_dask, use_memmap):

    orig_fname, sd_fname, interf_fname = cube_data

    orig_cube, orig_data = cube_and_raw(orig_fname, use_dask=use_dask)
    sd_cube, sd_data = cube_and_raw(sd_fname, use_dask=use_dask)
    interf_cube, interf_data = cube_and_raw(interf_fname, use_dask=use_dask)

    interf_cube = interf_cube.to(u.Jy / u.beam)

    combo_cube_sc = feather_simple_cube(interf_cube, sd_cube,
                                        use_memmap=use_memmap,
                                        use_dask=use_dask)

    assert orig_cube.shape == combo_cube_sc.shape

    # Output units should in the units of the interferometer cube
    assert combo_cube_sc.unit == interf_cube.unit

    combo_cube_sc_smunits = combo_cube_sc.to(orig_cube.unit)

    # Test against flux recovery
    frac_diff = (orig_cube - combo_cube_sc_smunits).sum() / orig_cube.sum()
    npt.assert_allclose(0., frac_diff, atol=5e-3)


def test_feather_cube_consistency(cube_data, use_memmap):
    '''
    Before fourier_combine_cubes is fully deprecated, check consistency
    with the output from feather_simple_cubes.
    '''

    orig_fname, sd_fname, interf_fname = cube_data

    orig_cube, orig_data = cube_and_raw(orig_fname, use_dask=False)
    sd_cube, sd_data = cube_and_raw(sd_fname, use_dask=False)
    interf_cube, interf_data = cube_and_raw(interf_fname, use_dask=False)

    combo_cube_sc = feather_simple_cube(interf_cube, sd_cube,
                                        use_memmap=use_memmap)

    combo_cube_fcc = fourier_combine_cubes(interf_cube, sd_cube, return_hdu=True)
    combo_cube_fcc_sc = SpectralCube.read(combo_cube_fcc)

    assert orig_cube.shape == combo_cube_sc.shape

    assert combo_cube_sc.unit == interf_cube.unit

    diff_cube = (combo_cube_sc - combo_cube_fcc_sc) / combo_cube_sc

    assert diff_cube.max().value < 1e-10
    assert np.abs(diff_cube.min().value) < 1e-10

