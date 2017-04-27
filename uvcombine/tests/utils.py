import image_registration
from astropy import convolution
from astropy.io import fits
from astropy import units as u
import numpy as np

def generate_test_data(imsize, powerlaw, seed=0):
    np.random.seed(seed)
    im = image_registration.tests.make_extended(imsize=imsize, powerlaw=powerlaw)
    return im

def generate_test_fits(imsize, powerlaw, beamfwhm,
                       pixel_scale=1*u.arcsec, seed=0,
                       restfreq=100*u.GHz,
                      ):

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
              'RESTFRQ': restfreq.to(u.Hz).value,
              'BUNIT': 'MJy/sr',
             }

    hdu = fits.PrimaryHDU(data=im, header=fits.Header(header))

    return hdu

def interferometrically_observe_image(image, pixel_scale,
                                      largest_angular_scale,
                                      smallest_angular_scale):

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

    return im_interferometered

def singledish_observe_image(image, pixel_scale, smallest_angular_scale):
    """
    Smallest angular scale is interpreted as the FWHM of a Gaussian
    """

    FWHM_CONSTANT = (8*np.log(2))**0.5
    kernel = convolution.Gaussian2DKernel(smallest_angular_scale/FWHM_CONSTANT)

    # create the single-dish map by convolving the image with a FWHM=40" kernel
    # (this interpretation is much easier than the sharp-edged stuff in fourier
    # space because the kernel is created in real space)
    singledish_im = convolution.convolve_fft(image,
                                             kernel=kernel,
                                             boundary='fill', fill_value=image.mean())

    return singledish_im
