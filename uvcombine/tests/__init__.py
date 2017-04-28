# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, absolute_import, division

import os


def path(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)
