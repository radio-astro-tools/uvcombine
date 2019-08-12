import numpy as np
import pytest
import itertools
from astropy.io import fits
try:
    import casa
    from taskinit import ia
except ImportError:
    from casatasks import feather
except ImportError:
    casa_imported = False

from . import path
from .. import uvcombine

@pytest.mark.skipif('not casa_imported')
@pytest.mark.parametrize(('sdfactor', 'lowpassfilterSD'), itertools.product((0.5, 1, 1.5), (True,False)))
def test_casafeather(sdfactor, lowpassfilterSD):
    try:
        from casa import feather,exportfits,importfits
    except ImportError:
        from casatasks import feather,exportfits,importfits

    print(f"Testing casafeather with sdfactor={sdfactor}")

    input_hires = path('input_image_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as.fits')
    input_lores = path('input_image_sz512as_pl1.5_fwhm2as_scale1as_sd33as.fits')

    importfits(fitsimage=input_hires, imagename=input_hires.replace(".fits",".image"), overwrite=True)
    importfits(fitsimage=input_lores, imagename=input_lores.replace(".fits",".image"), overwrite=True)
    output = path(f'casafeathered_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as_w_sd33as_sdf={sdfactor}_lowpass={lowpassfilterSD}.image')

    feather(imagename=output,
            highres=input_hires.replace(".fits",".image"),
            lowres=input_lores.replace(".fits",".image"),
            sdfactor=sdfactor,
            lowpassfiltersd=lowpassfilterSD,
           )
    exportfits(imagename=output, fitsimage=output+".fits",
               overwrite=True, dropdeg=True)

    feathered_hdu = uvcombine.feather_simple(hires=input_hires,
                                             lores=input_lores,
                                             lowresscalefactor=sdfactor,
                                             lowpassfilterSD=lowpassfilterSD,
                                             return_hdu=True)
    output2 = path(f'uvcombinefeathered_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as_w_sd33as_sdf={sdfactor}_lowpass={lowpassfilterSD}.fits')
    feathered_hdu.writeto(output2, overwrite=True)

    diff = fits.getdata(output+".fits") - feathered_hdu.data
    newhdu = feathered_hdu.copy()
    newhdu.data = diff
    output_diff = path(f'casafeathered-uvcombinefeathered_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as_w_sd33as_sdf={sdfactor}_lowpass={lowpassfilterSD}.fits')
    newhdu.writeto(output_diff, overwrite=True)

    print("Proof that we have exactly reimplemented CASA's feather: ")
    print("((casa-uvcombine)**2 / casa**2).sum() = {0}"
          .format(((diff**2)/(fits.getdata(output+".fits")**2)).sum()))
    print("Maximum of abs(diff): {0}".format(np.abs(diff).max()))
    if lowpassfilterSD:
        assert np.abs(diff).max() < 2e-4
    else:
        assert np.abs(diff).max() < 1e-7
