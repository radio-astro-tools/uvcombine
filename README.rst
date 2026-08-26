uvcombine
=========

uvcombine provides tools for combining high-resolution images with missing large angular scales (Fourier-domain short-spacings)
with low-resolution images containing the short/zero spacing, including:

* Feathering two images
* Feathering two cubes (including for large cubes using spectral-cube's dask implementation)
* uv-overlap consistency tests and measuring the single dish flux scaling factor.

See the `documentation <https://uvcombine.readthedocs.io/en/latest/>`_ for more information.

## Development

Example notebooks under ``examples/`` are kept free of cell outputs on ``main`` so
diffs stay reviewable. This is enforced by a ``pre-commit`` hook (`nbstripout
<https://github.com/kynan/nbstripout>`_) plus a CI check that fails if a notebook
with outputs is pushed. To set up the hook locally::

    pip install -e ".[notebooks]"
    pre-commit install

Notebooks with their outputs executed are kept on the ``with-output`` branch
instead; it is refreshed manually (not by CI) whenever an example notebook changes.

Worked notebook examples of the uvcombine functionality can be found in the `examples <https://github.com/uvcombine/uvcombine/tree/master/examples>`_ directory.

radio-astro-tools
^^^^^^^^^^^^^^^^^

This package is part of the radio-astro-tools project. See
`radio-astro-tools <https://radio-astro-tools.github.io/>`_ for more information.

