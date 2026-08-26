# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .version import version as __version__

# For egg_info test builds to pass, put package imports here.
from .uvcombine import (feather_plot, feather_simple, feather_compare,
                        feather_simple_cube)

__all__ = ['feather_plot', 'feather_simple', 'feather_compare',
           'feather_simple_cube']
