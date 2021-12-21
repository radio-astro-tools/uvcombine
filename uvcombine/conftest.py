# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.
from __future__ import print_function, absolute_import, division

import os
from distutils.version import LooseVersion

import pytest
import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy import units

from astropy.version import version as astropy_version

from .utils import (generate_testing_data,
                    generate_test_cube,
                    generate_test_fits)

if astropy_version < '3.0':
    from astropy.tests.pytest_plugins import *
    del pytest_report_header
else:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS


def pytest_configure(config):

    config.option.astropy_header = True

    PYTEST_HEADER_MODULES['Astropy'] = 'astropy'

@pytest.fixture
def fake_overlap_samples(size=1000):

    np.random.seed(67848923)

    lowres_pts = np.random.lognormal(size=size)
    highres_pts = np.abs(lowres_pts + np.random.normal(scale=0.05, size=size))

    return lowres_pts, highres_pts

@pytest.fixture
def plaw_test_data():
    out = generate_testing_data(return_images=True,
                                powerlawindex=1.5,
                                largest_scale=56. * units.arcsec,
                                smallest_scale=3. * units.arcsec,
                                lowresfwhm=30. * units.arcsec,
                                pixel_scale=1 * units.arcsec,
                                imsize=512)

    # angscales, ratios, lowres_pts, highres_pts = out
    orig_hdu, lowres_hdu, highres_hdu = out

    return orig_hdu, lowres_hdu, highres_hdu

@pytest.fixture
def plaw_test_cube_sc():
    out = generate_test_cube(return_hdu=False,
                             powerlawindex=1.5,
                             largest_scale=56. * u.arcsec,
                             smallest_scale=3. * u.arcsec,
                             lowresfwhm=30. * u.arcsec,
                             pixel_scale=1 * u.arcsec,
                             imsize=512,
                             nchan=3)

    orig_cube, sd_cube, interf_cube = out

    return orig_cube, sd_cube, interf_cube

@pytest.fixture
def plaw_test_cube_hdu():
    out = generate_test_cube(return_hdu=True,
                             powerlawindex=1.5,
                             largest_scale=56. * u.arcsec,
                             smallest_scale=3. * u.arcsec,
                             lowresfwhm=30. * u.arcsec,
                             pixel_scale=1 * u.arcsec,
                             imsize=512,
                             nchan=3)

    orig_hdu, sd_hdu, interf_hdu = out

    return orig_hdu, sd_hdu, interf_hdu
