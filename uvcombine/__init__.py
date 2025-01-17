# Licensed under a 3-clause BSD style license - see LICENSE.rst

from ._astropy_init import __version__, test

# For egg_info test builds to pass, put package imports here.
from .uvcombine import (feather_plot, feather_simple, feather_compare,
                        fourier_combine_cubes, feather_simple_cube)

__all__ = ['feather_plot', 'feather_simple', 'feather_compare',
           'fourier_combine_cubes', 'feather_simple_cube']
