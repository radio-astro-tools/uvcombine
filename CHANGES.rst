1.0.1 (unreleased)
------------------
- Require numpy>=2.0, spectral-cube>=0.6.7 and radio-beam>=0.3.10 (#59)
- Test on Python 3.11 (Linux) and 3.13 (macOS/Windows); test CASA on
  Python 3.12 and 3.13 (#59)

1.0.0 (2026-09-10)
------------------
First release on PyPI. Requires Python 3.10+ and astropy 6.1+.

Highlights:

- Feathering of high-resolution (interferometric) and low-resolution
  (single-dish) images with ``feather_simple``, and of spectral cubes with
  ``feather_simple_cube``, including dask-backed, memory-limited processing of
  large cubes (#25, #34)
- uv-overlap consistency checks and single-dish flux scale factors with
  ``feather_compare``, ``feather_plot`` and ``uvcombine.scale_factor`` (#16)
- Tests comparing ``feather_simple`` with CASA's ``feather`` task (#23)

Changes leading up to 1.0.0:

- Remove the remaining ``turbustat`` imports and make ``statsmodels`` an
  optional dependency, installed with ``pip install "uvcombine[stats]"`` (#58)
- Fix ``find_scale_factor(method='distrib')`` returning exp(scale factor)
  when the likelihood fit is not used, including when statsmodels is not
  installed (#58)
- Vendor TurbuStat's ``pspec`` to drop the runtime ``turbustat`` dependency (#57)
- Handle unitless axes in ``feather_plot`` (#54)
- Remove old deprecated code (#48)
- Replace astropy's defunct ``ProgressBar`` with ``tqdm`` (#44)
- Switch PyPI publishing to trusted publishing (OIDC) and add the missing
  ``LICENSE.rst`` (#52)
- Docs, CI and packaging maintenance (#29, #30, #38, #47, #51, #53, #55, #56)
