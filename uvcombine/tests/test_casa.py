import numpy as np
import pytest
import itertools
from astropy.io import fits
from spectral_cube import SpectralCube, Projection

try:
    from casatasks import feather, importfits
    from casatools import image

    ia = image()

    casa_imported = True

# except ImportError:
#     casa_imported = False

from .. import feather_simple

@pytest.mark.skipif('not casa_imported')
@pytest.mark.parametrize(('sdfactor', 'lowpassfilterSD'), itertools.product((0.5, 1, 1.5), (True,False)))
def test_casafeather(image_sz512as_pl1p5_fwhm2as_scale1as, sdfactor, lowpassfilterSD):

    tmp_path, input_fn, intf_fn, sd_fn = image_sz512as_pl1p5_fwhm2as_scale1as

    intf_hdu = fits.open(intf_fn)[0]
    sd_hdu = fits.open(sd_fn)[0]

    # Feathering with CASA

    intf_fn_image = intf_fn.parent / intf_fn.name.replace(".fits",".image")
    sd_fn_image = sd_fn.parent / sd_fn.name.replace(".fits",".image")

    # CASA needs a posix string to work
    importfits(fitsimage=intf_fn.as_posix(), imagename=intf_fn_image.as_posix())
    importfits(fitsimage=sd_fn.as_posix(), imagename=sd_fn_image.as_posix())

    output_name = tmp_path / 'casafeathered.image'

    feather(imagename=output_name.as_posix(),
            highres=intf_fn_image.as_posix(),
            lowres=sd_fn_image.as_posix(),
            sdfactor=sdfactor,
            lowpassfiltersd=lowpassfilterSD,
           )

    # Right now read as spectralcube despite being a 2D image.
    casa_feather_proj = SpectralCube.read(output_name)[0]

    # exportfits(imagename=output, fitsimage=output+".fits",
    #            overwrite=True, dropdeg=True)

    # Feathering with uvcombine
    feathered_hdu = feather_simple(hires=intf_hdu, lores=sd_hdu,
                                   lowresscalefactor=sdfactor,
                                   lowpassfilterSD=lowpassfilterSD,
                                   return_hdu=True)

    uvcomb_feather_proj = Projection.from_hdu(feathered_hdu)

    diff = (casa_feather_proj - uvcomb_feather_proj).value

    print("Proof that we have exactly reimplemented CASA's feather: ")
    print("((casa-uvcombine)**2 / casa**2).sum() = {0}"
          .format(((diff**2)/(casa_feather_proj.value**2)).sum()))
    print("Maximum of abs(diff): {0}".format(np.abs(diff).max()))

    if lowpassfilterSD:
        assert np.abs(diff).max() < 2e-4
    else:
        assert np.abs(diff).max() < 1e-7
