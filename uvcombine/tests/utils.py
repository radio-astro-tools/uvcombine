from astropy import convolution
from astropy.io import fits
from astropy import units as u
import numpy as np


from ..uvcombine import feather_compare
from ..utils import make_extended


def generate_test_data(imsize, powerlaw, seed=0):
    """
    Simple wrapper of `make_extended`
    """
    np.random.seed(seed)
    im = make_extended(imsize=imsize, powerlaw=powerlaw)
    return im


def generate_header(pixel_scale, beamfwhm, imsize, restfreq, with_specaxis=False):

    header = {'CDELT1': -(pixel_scale).to(u.deg).value,
              'CDELT2': (pixel_scale).to(u.deg).value,
              'BMAJ': beamfwhm.to(u.deg).value,
              'BMIN': beamfwhm.to(u.deg).value,
              'BPA': 0.0,
              'CRPIX1': imsize / 2.,
              'CRPIX2': imsize / 2.,
              'CRVAL1': 0.0,
              'CRVAL2': 0.0,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CUNIT1': 'deg',
              'CUNIT2': 'deg',
              'BUNIT': 'MJy/sr',
              }

    if with_specaxis:
        header['CRVAL3'] = restfreq.to(u.Hz).value
        header['CUNIT3'] = 'Hz'
        header['CDELT3'] = 1e6,  # 1 MHz; doesn't matter
        header['CRPIX3'] = 1
        header['CTYPE3'] = 'FREQ'
        header['RESTFRQ'] = restfreq.to(u.Hz).value

    return fits.Header(header)


def generate_test_fits(imsize, powerlaw, beamfwhm,
                       pixel_scale=1 * u.arcsec, seed=0,
                       restfreq=100 * u.GHz,
                       ):
    """
    Create a FITS image using ``image_registration``'s toolkit for producing
    power-law power-spectrum images.

    Parameters
    ----------
    imsize : int
        Image size in pixels (square by default)
    powerlaw : float
        The power law index of the resulting image.  Typical reasonable values
        are between 1 and 2.
    beamfwhm : u.arcsec equivalent
        An astropy quantity defining the beam size.  This is not used to
        produce the image (the image is *not* smoothed), but it is put in the
        header so that the file is parseable by CASA and convertible into
        various beam-based units.
    pixel_scale : u.arcsec equivalent
        The (square) pixel size in arcsec.  Needed for proper interpretation by
        various codes that parse the FITS header
    seed : int
        Random seed to use for numpy's random number generator. Defaults to 0,
        a fixed random seed, for reproducibility's sake.
    restfreq : u.Hz equivalent
        A rest frequency.  Needed for conversion to Kelvin (brightness
        temperature) units.

    Returns
    -------
    hdu : fits.PrimaryHDU
        A FITS HDU object
    """

    im = generate_test_data(imsize=imsize, powerlaw=powerlaw, seed=seed)

    header = generate_header(pixel_scale, beamfwhm, imsize, restfreq)

    hdu = fits.PrimaryHDU(data=im, header=header)

    return hdu

def interferometrically_observe_image(image, pixel_scale,
                                      largest_angular_scale,
                                      smallest_angular_scale):
    """
    Given an array image with a specified pixel scale, interferometrically
    observe that image.

    Parameters
    ----------
    image : np.array
        The image array (should be a numpy array, not a quantity array)
    pixel_scale : u.arcsec equivalent
        The (square) pixel size in arcsec.
    largest_angular_scale : u.arcsec equivalent
        The angular scale above which the data will be filtered out (a sharp
        filter is used, which is somewhat realistic)
    smallest_angular_scale : u.arcsec equivalent
        The angular scale below which the data will be filtered out (a sharp
        filter is used, though this is usually better approximated by a
        gaussian)

    Returns
    -------
    im_interferometered : np.array
        The *complex* image array resulting from the interferometric
        observations
    ring : np.array
        The boolean mask array applied to the fourier transform of the input
        image.
    """

    ygrid, xgrid = np.indices(image.shape, dtype='float')
    rr = ((xgrid-image.shape[1]/2)**2+(ygrid-image.shape[0]/2)**2)**0.5

    # Create a UV sampling mask.
    # *please sanity check this!*
    # Are the "UV" data correct, or are they off by a factor of 2?
    # rr_uv = (rr / rr.max() / 2. / pixel_scale)
    # ring = (1/rr_uv < largest_angular_scale) & (1/rr_uv > smallest_angular_scale)

    # EWK -- Something is off in the above masking.
    img_scale = image.shape[0] * pixel_scale
    ring = (rr >= (img_scale / largest_angular_scale)) & \
        (rr <= (img_scale / smallest_angular_scale))

    # create the interferometric map by removing both large and small angular
    # scales in fourier space
    imfft = np.fft.fft2(image)
    imfft_interferometered = imfft * np.fft.fftshift(ring)
    im_interferometered = np.fft.ifft2(imfft_interferometered)

    return im_interferometered, ring

def singledish_observe_image(image, pixel_scale, smallest_angular_scale):
    """
    Given an array image with a specified pixel scale, interferometrically
    observe that image.

    Parameters
    ----------
    image : np.array
        The image array (should be a numpy array, not a quantity array)
    pixel_scale : u.arcsec equivalent
        The (square) pixel size in arcsec.
    smallest_angular_scale : u.arcsec equivalent
        The beam of the image.  This is interpreted as the FWHM of a gaussian.

    Returns
    -------
    singledish_im : np.array
        The image array resulting from smoothing the input image
    """

    FWHM_CONSTANT = (8*np.log(2))**0.5
    kernel = convolution.Gaussian2DKernel(smallest_angular_scale/FWHM_CONSTANT/pixel_scale)

    # create the single-dish map by convolving the image with a FWHM=40" kernel
    # (this interpretation is much easier than the sharp-edged stuff in fourier
    # space because the kernel is created in real space)
    singledish_im = convolution.convolve_fft(image,
                                             kernel=kernel,
                                             boundary='fill', fill_value=image.mean())

    return singledish_im


def testing_data(return_images=True,
                 powerlawindex=1.5,
                 largest_scale=56. * u.arcsec,
                 smallest_scale=3. * u.arcsec,
                 lowresfwhm=30. * u.arcsec,
                 pixel_scale=1 * u.arcsec,
                 imsize=512):

    orig_img = generate_test_data(imsize, powerlawindex, seed=67848923)

    restfreq = (2 * u.mm).to(u.GHz, u.spectral())

    sd_img = singledish_observe_image(orig_img, pixel_scale, lowresfwhm)

    interf_img = \
        interferometrically_observe_image(orig_img, pixel_scale,
                                          largest_scale,
                                          smallest_scale)[0].real

    # Make these FITS HDUs
    orig_hdr = generate_header(pixel_scale, pixel_scale, imsize, restfreq)
    orig_hdu = fits.PrimaryHDU(orig_img, header=orig_hdr)

    sd_hdr = generate_header(pixel_scale, lowresfwhm, imsize, restfreq)
    sd_hdu = fits.PrimaryHDU(sd_img, header=sd_hdr)

    interf_hdr = generate_header(pixel_scale, smallest_scale, imsize, restfreq)
    interf_hdu = fits.PrimaryHDU(interf_img, header=interf_hdr)

    if return_images:
        return orig_hdu, sd_hdu, interf_hdu

    angscales, ratios, highres_pts, lowres_pts = \
        feather_compare(interf_hdu, sd_hdu, SAS=lowresfwhm, LAS=largest_scale,
                        lowresfwhm=lowresfwhm, return_samples=True,
                        doplot=False)

    # There are a bunch of tiny points that should be empty, but aren't b/c
    # of numerical rounding
    good_pts = ratios > 1e-10
    angscales = angscales[good_pts]
    ratios = ratios[good_pts]
    highres_pts = highres_pts[good_pts]
    lowres_pts = lowres_pts[good_pts]

    return angscales, ratios, lowres_pts, highres_pts
