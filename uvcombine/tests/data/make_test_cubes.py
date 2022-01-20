import numpy as np
from uvcombine import utils
from uvcombine import feather_simple
from uvcombine.unit_utils import convert_to_casa
from astropy import units as u
from astropy import log
from astropy.io import fits
import radio_beam

if __name__ == "__main__":

    pixel_scale = 1*u.arcsec
    log.info("Generate input image")
    input_hdu = utils.generate_test_fits(imsize=512, powerlaw=1.5,
                                         beamfwhm=2*u.arcsec,
                                         pixel_scale=pixel_scale,
                                         seed=0)
    input_fn = "input_image_sz512as_pl1.5_fwhm2as_scale1as.fits"
    input_hdu.writeto(input_fn,
                      overwrite=True)

    log.info("make Interferometric image")
    intf_data = utils.interferometrically_observe_image(image=input_hdu.data,
                                                        pixel_scale=pixel_scale,
                                                        largest_angular_scale=40*u.arcsec,
                                                        smallest_angular_scale=2*u.arcsec)[0].real
    intf_hdu = fits.PrimaryHDU(data=intf_data,
                               header=input_hdu.header
                              )
    intf_fn = "input_image_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as.fits"
    convert_to_casa(intf_hdu).writeto(intf_fn, overwrite=True)

    log.info("make SD image")
    sd_header = input_hdu.header.copy()
    sd_header['BMAJ'] = sd_header['BMIN'] = (33*u.arcsec).to(u.deg).value
    sd_data = utils.singledish_observe_image(image=input_hdu.data,
                                             pixel_scale=pixel_scale,
                                             beam=radio_beam.Beam(33*u.arcsec))
    sd_hdu = fits.PrimaryHDU(data=sd_data, header=sd_header)
    sd_fn = "input_image_sz512as_pl1.5_fwhm2as_scale1as_sd33as.fits"
    convert_to_casa(sd_hdu).writeto(sd_fn, overwrite=True)

    log.info("Feather data")
    feathered_hdu = feather_simple(hires=intf_hdu, lores=sd_hdu, return_hdu=True)
    feathered_hdu.writeto("input_image_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as_sd33as_feathered_MJySr.fits",
                          overwrite=True)
    feathered_hdu = feather_simple(hires=intf_fn, lores=sd_fn, return_hdu=True)
    feathered_hdu.writeto("input_image_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as_sd33as_feathered_JyBeam.fits",
                          overwrite=True)
