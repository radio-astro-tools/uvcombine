import image_registration
from astropy import convolution
from astropy.io import fits
from astropy import units as u
import numpy as np

def generate_test_data(imsize, powerlaw, seed=0):
    """
    Simple wrapper of `image_registration.tests.make_extended`
    """
    np.random.seed(seed)
    im = image_registration.tests.make_extended(imsize=imsize, powerlaw=powerlaw)
    return im

def generate_test_fits(imsize, powerlaw, beamfwhm,
                       pixel_scale=1*u.arcsec, seed=0,
                       restfreq=100*u.GHz,
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

    header = {'CDELT1': -(pixel_scale).to(u.deg).value,
              'CDELT2': (pixel_scale).to(u.deg).value,
              'BMAJ': beamfwhm.to(u.deg).value,
              'BMIN': beamfwhm.to(u.deg).value,
              'BPA': 0.0,
              'CRPIX1': imsize/2.,
              'CRPIX2': imsize/2.,
              'CRVAL1': 0.0,
              'CRVAL2': 0.0,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CUNIT1': 'deg',
              'CUNIT2': 'deg',
              'CRVAL3': restfreq.to(u.Hz).value,
              'CUNIT3': 'Hz',
              'CDELT3': 1e6, # 1 MHz; doesn't matter
              'CRPIX3': 1,
              'CTYPE3': 'FREQ',
              'RESTFRQ': restfreq.to(u.Hz).value,
              'BUNIT': 'MJy/sr',
             }

    hdu = fits.PrimaryHDU(data=im, header=fits.Header(header))

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
    rr_uv = (rr / rr.max() / 2. / pixel_scale)

    # Create a UV sampling mask.
    # *please sanity check this!*
    # Are the "UV" data correct, or are they off by a factor of 2?
    ring = (1/rr_uv < largest_angular_scale) & (1/rr_uv > smallest_angular_scale)

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
