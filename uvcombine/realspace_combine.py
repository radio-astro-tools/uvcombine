import numpy as np
import matplotlib.pyplot as pl
from astropy import units as u
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
import radio_beam
from astropy import wcs
from spectral_cube import Projection

def linear_combine(hires, lores,
                   highresextnum=0,
                   lowresextnum=0,
                   highresscalefactor=1.0,
                   lowresscalefactor=1.0,
                   lowresfwhm=None,
                   return_hdu=False,
                   return_regridded_lores=False,
                   match_units=True,
                   convolve=convolve_fft,
                  ):
    """
    Implement a simple linear combination following Faridani et al 2017
    """

    if isinstance(hires, str):
        proj_hi = Projection.from_hdu(fits.open(hires)[highresextnum])
    else:
        proj_hi = Projection.from_hdu(hires)

    if isinstance(hires, str):
        proj_lo = Projection.from_hdu(fits.open(lores)[lowresextnum])
    else:
        proj_lo = Projection.from_hdu(lores)

    if lowresfwhm is None:
        beam_low = proj_lo.beam
        lowresfwhm = beam_low.major
        log.info("Low-res FWHM: {0}".format(lowresfwhm))
    else:
        beam_low = Beam(major=lowresfwhm)

    if match_units:
        # After this step, the units of im_hi are some sort of surface brightness
        # unit equivalent to that specified in the high-resolution header's units
        # Note that this step does NOT preserve the values of im_lowraw and
        # header_lowraw from above
        # im_lowraw, header_low = match_flux_units(image=proj_lo.value,
        #                                          image_header=proj_lo.header.copy(),
        #                                          target_header=proj_hi.header)
        # proj_lo = Projection(im_lowraw, header=header_low)

        proj_lo = proj_lo.to(proj_hi.unit)

    proj_lo_regrid = proj_lo.reproject(proj_hi.header)

    missing_flux = proj_lo_regrid - proj_hi.convolve_to(beam_low)

    combo = missing_flux + proj_hi

    if return_hdu:
        combo_hdu = fits.PrimaryHDU(data=combo.real, header=proj_hi.header)
        combo = combo_hdu

    if return_regridded_lores:
        return combo, hdu_low
    else:
        return combo


def image_space_combination(lowres, highres):
    """
    This is the outline of a general implementation of image space combination;
    it works for the specified files below.

    Remaining problems:
        - deconvolution appears only to work over the inner 1/4 of the image.
        I don't know why.
        - The deconvolved image needs to be reconvolved with the dirty beam
        and subtracted from the original data to make a residual map.  It then
        needs to be convolved with a (gaussian) restoring beam and added to the
        residual map.
    """
    pass
    # Stanimirovic 2002 Section 6.1: "Linear Combination" approach
#    orig_fn = 'ForLinearCombination/combo_model.fits'
#    clean_fn = 'ForLinearCombination/combo_alma.fits'
#    dirty_fn = 'ForLinearCombination/combo_dirty.fits'
#    dirtybeam_fn = 'ForLinearCombination/combo_psf.fits'
#    tp_fn = 'ForLinearCombination/combo_sm_tp.fits'
#
#    singledishdiameter = 25*u.m
#
#    restfrq = fits.getheader(clean_fn)['RESTFRQ']*u.Hz
#
#    lowresfwhm = (1.22*(restfrq.to(u.m, u.spectral()) /
#                        (singledishdiameter))).to(u.arcsec, u.dimensionless_angles())
#    pixscale = fits.getheader(clean_fn)['CDELT2']*u.deg
#
#    int_beam = radio_beam.Beam.from_fits_header(fits.getheader(dirty_fn))
#    tp_beam = radio_beam.Beam.from_fits_header(fits.getheader(tp_fn))
#
#    kernel_sigma_arcsec = lowresfwhm/(8*np.log(2))**0.5
#    kernel_sigma = (kernel_sigma_arcsec/pixscale).decompose()
#    # jybeam_ratio is the factor by which the data must be multiplied after
#    # convolution to preserve the Jansky/beam units
#    jybeam_ratio = (2*np.pi*kernel_sigma_arcsec**2 / int_beam.sr).decompose()
#
#    kernel = Gaussian2DKernel(kernel_sigma)
#    conv = convolve_fft(fits.getdata(orig_fn).squeeze(), kernel)
#    conv_largebeam = conv*jybeam_ratio
#
#    int_kernel_sigma_arcsec = int_beam.major/(8*np.log(2))**0.5
#    int_kernel_sigma = (int_kernel_sigma_arcsec/pixscale).decompose()
#    interferometer_kernel = Gaussian2DKernel(int_kernel_sigma)
#
#    smhdu = fits.open(orig_fn)
#    smhdu[0].data = conv_largebeam.value
#    smhdu[0].header['BMAJ'] = lowresfwhm.to(u.deg).value
#    smhdu[0].header['BMIN'] = lowresfwhm.to(u.deg).value
#    smhdu[0].header['BPA'] = 0.0
#    smhdu[0].writeto(tp_fn, clobber=True)
#
#    cln = fits.getdata(clean_fn).squeeze()
#    dirty = fits.getdata(dirty_fn).squeeze()
#    tp = fits.getdata(tp_fn).squeeze()
#
#    alpha = int_beam.sr / tp_beam.sr
#
#    dirtybeamim = fits.getdata(dirtybeam_fn).squeeze()
#    tpbeamkern = tp_beam.as_kernel(pixscale,
#                                   x_size=dirtybeamim.shape[1],
#                                   y_size=dirtybeamim.shape[0])
#    tpbeamim = tpbeamkern.array / tpbeamkern.array.max()
#
#    combined = (dirty + alpha*tp)/(1+alpha)
#    beam = (dirtybeamim + alpha*tpbeamim)/(1+alpha)
#
#    hduL = fits.open(clean_fn)
#    hduL[0].data = combined.value
#    hduL.writeto('ForLinearCombination/dirty_combined_image.fits', clobber=True)
#    hduL = fits.open(clean_fn)
#    hduL[0].data = beam.value
#    hduL.writeto('ForLinearCombination/dirty_combined_psf.fits', clobber=True)
#
#    importfits('ForLinearCombination/dirty_combined_image.fits',
#               'ForLinearCombination/dirty_combined_image.image',
#               overwrite=True)
#    importfits('ForLinearCombination/dirty_combined_psf.fits',
#               'ForLinearCombination/dirty_combined_psf.image',
#               overwrite=True)
#
#    for method in ("clark", "hogbom", "multiscale", "mem"):
#        outname = 'ForLinearCombination/deconvolved_combined_image_{0}.model'.format(method)
#        os.system('rm -rf {0}'.format(outname))
#        deconvolve(imagename='ForLinearCombination/dirty_combined_image.image',
#                   model=outname,
#                   alg=method,
#                   psf='ForLinearCombination/dirty_combined_psf.image',
#                   niter=1000,
#                   threshold='50mJy',
#                  )
#        exportfits(outname,
#                   outname+".fits",
#                   dropdeg=True,
#                   overwrite=True)
#        conv = convolve_fft(fits.getdata(outname+".fits").squeeze(), interferometer_kernel)
#        hduL = fits.open(outname+".fits")
#        hduL[0].data = conv.value
#        hduL.writeto(outname.replace(".model",".image")+".fits", clobber=True)
#
