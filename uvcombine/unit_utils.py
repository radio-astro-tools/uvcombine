from astropy import units as u
from astropy import wcs
from astropy.utils import deprecated
import radio_beam

from .uvcombine import file_in

@deprecated("2021", message="Instead use SpectralCube.to for unit conversions.")
def convert_to_casa(hdu):
    """
    Convert a FITS HDU to casa-compatible units, i.e., Jy/beam
    """
    hdu = file_in(hdu)[0].copy()

    beam = radio_beam.Beam.from_fits_header(hdu.header)

    if hdu.header['BUNIT'] == 'K':
        imwcs = wcs.WCS(hdu.header)
        cfreq = imwcs.sub([wcs.WCSSUB_SPECTRAL]).wcs_world2pix([0], 0)[0][0]
        hdu.data = u.Quantity(hdu.data,
                              unit=u.K).to(u.Jy,
                                           u.brightness_temperature(beam,
                                                                    cfreq*u.Hz)).value
    elif u.Unit(hdu.header['BUNIT']).is_equivalent(u.MJy/u.sr):
        hdu.data = u.Quantity(hdu.data,
                              u.Unit(hdu.header['BUNIT'])).to(u.Jy/beam).value
    elif hdu.header['BUNIT'] not in ('Jy/beam', 'Jy / beam', 'beam-1 Jy'): # all are OK?
        raise ValueError("Header BUNIT not recognized")

    hdu.header['BUNIT'] = 'Jy/beam'

    return hdu
