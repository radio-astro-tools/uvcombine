Installing ``uvcombine``
============================

Requirements
------------

This package has the following dependencies:

* `Python <http://www.python.org>`_ 3.10 or later
* `Numpy <http://www.numpy.org>`_ 1.8 or later
* `Astropy <http://www.astropy.org>`__ 6.1 or later
*  matplotlib 3.5 or later
*  reproject 0.13 or later
*  spectral-cube 0.4.0 or later
*  radio-beam 0.3.5 or later
*  tqdm

Installation
------------

To install the latest stable release, you can type::

    pip install uvcombine

You can download the latest tar file from
`PyPI <https://pypi.python.org/pypi/uvcombine>`_ and install it using::

    pip install -e .


Developer version
-----------------

If you want to install the latest developer version of the uvcombine code, you
can do so from the git repository::

    git clone https://github.com/radio-astro-tools/uvcombine.git
    cd uvcombine
    pip install -e .

You may need to add the ``--user`` option to the last line `if you do not
have root access <https://docs.python.org/2/install/#alternate-installation-the-user-scheme>`_.
You can also install the latest developer version in a single line with pip::

    pip install git+https://github.com/radio-astro-tools/uvcombine.git

