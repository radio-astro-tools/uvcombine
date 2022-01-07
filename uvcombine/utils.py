
import numpy as np
from astropy.utils import NumpyRNGContext
from astropy import convolution
from astropy.io import fits
from astropy import units as u
from spectral_cube import SpectralCube
from radio_beam import Beam


from .uvcombine import feather_compare


def make_extended(imsize, powerlaw=2.0,
                  return_fft=False, full_fft=True, seed=32788324):
    '''

    Generate a 2D power-law image with a specified index and random phases.

    Adapted from https://github.com/keflavich/image_registration and
    https://github.com/Astroua/TurbuStat/blob/master/turbustat/simulator/gen_field.py.

    Parameters
    ----------
    imsize : int
        Array size.
    powerlaw : float, optional
        Powerlaw index.
    return_fft : bool, optional
        Return the FFT instead of the image. The full FFT is
        returned, including the redundant negative phase phases for the RFFT.
    full_fft : bool, optional
        When `return_fft=True`, the full FFT, with negative frequencies, will
        be returned. If `full_fft=False`, the RFFT is returned.
    randomseed: int, optional
        Seed for random number generator.

    Returns
    -------
    newmap : np.ndarray
        Two-dimensional array with the given power-law properties.
    full_powermap : np.ndarray
        The 2D array in Fourier space. The zero-frequency is shifted to
        the centre.
    '''
    imsize = int(imsize)

    yy, xx = np.meshgrid(np.fft.fftfreq(imsize),
                         np.fft.rfftfreq(imsize), indexing="ij")

    # Circular whenever ellip == 1
    rr = (xx**2 + yy**2)**0.5

    # flag out the bad point to avoid warnings
    rr[rr == 0] = np.nan

    with NumpyRNGContext(seed):

        Np1 = (imsize - 1) // 2 if imsize % 2 != 0 else imsize // 2

        angles = np.random.uniform(0, 2 * np.pi,
                                   size=(imsize, Np1 + 1))

    phases = np.cos(angles) + 1j * np.sin(angles)

    # Rescale phases to an amplitude of unity
    phases /= np.sqrt(np.sum(phases**2) / float(phases.size))

    output = (rr**(-powerlaw / 2.)).astype('complex') * phases

    output[np.isnan(output)] = 0.

    # Impose symmetry
    # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
    if imsize % 2 == 0:
        output[1:Np1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1:Np1, -1] = np.conj(output[imsize:Np1:-1, -1])
        output[Np1, 0] = output[Np1, 0].real + 1j * 0.0
        output[Np1, -1] = output[Np1, -1].real + 1j * 0.0

    else:
        output[1:Np1 + 1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1:Np1 + 1, -1] = np.conj(output[imsize:Np1:-1, -1])

    # Zero freq components must have no imaginary part to be own conjugate
    output[0, -1] = output[0, -1].real + 1j * 0.0
    output[0, 0] = output[0, 0].real + 1j * 0.0

    if return_fft:

        if not full_fft:
            return output

        # Create the full power map, with the symmetric conjugate component
        if imsize % 2 == 0:
            power_map_symm = np.conj(output[:, -2:0:-1])
        else:
            power_map_symm = np.conj(output[:, -1:0:-1])

        power_map_symm[1::, :] = power_map_symm[:0:-1, :]

        full_powermap = np.concatenate((output, power_map_symm), axis=1)

        if not full_powermap.shape[1] == imsize:
            raise ValueError("The full output should have a square shape."
                             " Instead has {}".format(full_powermap.shape))

        return np.fft.fftshift(full_powermap)

    newmap = np.fft.irfft2(output)

    return newmap


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

def singledish_observe_image(image, pixel_scale, beam):
    """
    Given an array image with a specified pixel scale, interferometrically
    observe that image.

    Parameters
    ----------
    image : np.array
        The image array (should be a numpy array, not a quantity array)
    pixel_scale : u.arcsec equivalent
        The (square) pixel size in arcsec.
    beam : `~radio_beam.Beam`
        The beam of the image.

    Returns
    -------
    singledish_im : np.array
        The image array resulting from smoothing the input image
    """

    kernel = beam.as_kernel(pixel_scale)

    # create the single-dish map by convolving the image with a FWHM=40" kernel
    # (this interpretation is much easier than the sharp-edged stuff in fourier
    # space because the kernel is created in real space)
    singledish_im = convolution.convolve_fft(image,
                                             kernel=kernel,
                                             boundary='fill', fill_value=image.mean())

    return singledish_im


def generate_test_fits(imsize, powerlaw, beamfwhm,
                       pixel_scale=1 * u.arcsec,
                       restfreq=100 * u.GHz,
                       seed=32788324):
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

    im = make_extended(imsize=imsize, powerlaw=powerlaw, seed=seed)

    header = generate_header(pixel_scale, beamfwhm, imsize, restfreq)

    hdu = fits.PrimaryHDU(data=im, header=header)

    return hdu


def generate_header(pixel_scale, beamfwhm, imsize, restfreq, with_specaxis=False,
                    bunit=u.K):

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
              'BUNIT': bunit.to_string(),
              }

    if with_specaxis:
        header['CRVAL3'] = restfreq.to(u.Hz).value
        header['CUNIT3'] = 'Hz'
        header['CDELT3'] = 1e6  # 1 MHz; doesn't matter
        header['CRPIX3'] = 1
        header['CTYPE3'] = 'FREQ'
        header['RESTFRQ'] = restfreq.to(u.Hz).value

    return fits.Header(header)


def generate_testing_data(return_images=True,
                          powerlawindex=1.5,
                          largest_scale=56. * u.arcsec,
                          smallest_scale=3. * u.arcsec,
                          lowresfwhm=30. * u.arcsec,
                          pixel_scale=1 * u.arcsec,
                          imsize=512,
                          seed=32788324):

    orig_img = make_extended(imsize=imsize, powerlaw=powerlawindex, seed=seed)

    restfreq = (2 * u.mm).to(u.GHz, u.spectral())

    sd_img = singledish_observe_image(orig_img, pixel_scale, Beam(lowresfwhm))

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


def generate_test_cube(return_hdu=False,
                       powerlawindex=1.5,
                       largest_scale=56. * u.arcsec,
                       smallest_scale=3. * u.arcsec,
                       lowresfwhm=30. * u.arcsec,
                       pixel_scale=1 * u.arcsec,
                       imsize=512,
                       nchan=3,
                       seed=32788324):
    '''
    '''

    restfreq = (2 * u.mm).to(u.GHz, u.spectral())

    channels = []
    channels_sd = []
    channels_interf = []
    for i in range(nchan):
        chan = make_extended(imsize, powerlawindex, seed=seed)

        sd_img = singledish_observe_image(chan, pixel_scale, Beam(lowresfwhm))

        interf_img = \
            interferometrically_observe_image(chan, pixel_scale,
                                            largest_scale,
                                            smallest_scale)[0].real

        channels.append(chan)
        channels_sd.append(sd_img)
        channels_interf.append(interf_img)

    orig_cube = np.array(channels)
    sd_cube = np.array(channels_sd)
    interf_cube = np.array(channels_interf)

    # Make these FITS HDUs
    orig_hdr = generate_header(pixel_scale, pixel_scale, imsize,
                               restfreq, with_specaxis=True)
    orig_hdu = fits.PrimaryHDU(orig_cube, header=orig_hdr)

    sd_hdr = generate_header(pixel_scale, lowresfwhm, imsize,
                             restfreq, with_specaxis=True)
    sd_hdu = fits.PrimaryHDU(sd_cube, header=sd_hdr)

    interf_hdr = generate_header(pixel_scale, smallest_scale, imsize,
                                 restfreq, with_specaxis=True)
    interf_hdu = fits.PrimaryHDU(interf_cube, header=interf_hdr)

    if return_hdu:
        return orig_hdu, sd_hdu, interf_hdu
    else:
        # Return as spectral-cubes

        orig_sc = SpectralCube.read(orig_hdu)
        sd_sc = SpectralCube.read(sd_hdu)
        interf_sc = SpectralCube.read(interf_hdu)

        return orig_sc, sd_sc, interf_sc
