"""
Code for fourier-space combination of single dish and interferometer data
(or zero-spacing and non-zero-spacing data).  See:

    Stanimirovic et al 1999 (http://esoads.eso.org/abs/1999MNRAS.302..417S)
    (compares single-dish and interferometer data over commonly sampled
    UV space, but then does image combination + image-space deconvolution,
    which is not performed here)
"""
import radio_beam
from reproject import reproject_interp
from spectral_cube import SpectralCube
from spectral_cube import wcs_utils
from astropy.io import fits
from astropy import units as u
from astropy import log
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.utils.console import ProgressBar
import numpy as np
import FITS_tools
#from FITS_tools.hcongrid import hcongrid_hdu
#from FITS_tools.cube_regrid import regrid_cube_hdu
from astropy import wcs,stats
from astropy.convolution import convolve_fft, Gaussian2DKernel

def file_in(filename, extnum=0):
    """
    Take the input files. If input is already HDU, then return it.
    If input is a .fits filename, then read the .fits file.

    Return
    ----------
    hdu :  obj
       An object containing both the image and the header
    im  :  (float point?) array
       The image array
    header : header object
       The header of the input .fits file

    Parameters
    ----------
    filename : str
         The input .fits filename or a HDU variable name
    extnum   : int
         The extension number to use from the input .fits file
    """
    if isinstance(filename, (fits.ImageHDU, fits.PrimaryHDU)):
        hdu = filename
    else:
        hdu = fits.open(filename)[extnum]

    im = hdu.data.squeeze()
    #header = FITS_tools.strip_headers.flatten_header(hdu.header)
    header = wcs_utils.strip_wcs_from_header(hdu.header)
    mywcs = wcs.WCS(hdu.header).celestial
    header.update(mywcs.to_header())
    header['NAXIS'] = im.ndim
    for ii,dim in enumerate(im.shape):
        header['NAXIS{0}'.format(im.ndim-ii)] = dim


    # Attempt to determine spectral information and store that data for use
    # later
    imspwcs = wcs.WCS(hdu.header).sub([wcs.WCSSUB_SPECTRAL])
    if imspwcs.wcs.naxis == 1:
        reffreq, = u.Quantity(imspwcs.wcs_pix2world([0], 0),
                              unit=imspwcs.wcs.cunit[0])[0]
        reffreq = reffreq.to(u.Hz)
        header['REFFREQ'] = reffreq.value

    return hdu, im, header


def match_flux_units(image, image_header, target_header):
    """
    Match the flux units of the input image to the target header.  There are
    significant restrictions on the allowed units in the image and target
    headers.

    Parameters
    ----------
    image : array
        An array of flux density or surface brightness units
    image_header : fits.Header
        The header associated with the above array
    target_header : fits.Header
        The header whose units should be matched

    Returns
    -------
    new_image : array
        The array in the new units
    new_header : fits.Header
        The corresponding FITS header that specifies the appropriate units
        (including the beam area if the appropriate unit is Jy/beam)
    """

    # copy because we're modifying this and we don't want to inplace modify the
    # object ever
    image_header = image_header.copy()

    if 'BUNIT' not in image_header:
        raise ValueError("Image header must have a BUNIT specified")
    if 'BUNIT' not in target_header:
        raise ValueError("Target header must have a BUNIT specified")

    im_wcs = wcs.WCS(image_header).celestial
    target_wcs = wcs.WCS(target_header).celestial

    target_unit = u.Unit(target_header['BUNIT'])
    image_unit = u.Unit(image_header['BUNIT'])

    equivalency = None

    if u.beam in image_unit.bases or u.beam in target_unit.bases:
        image_beam = radio_beam.Beam.from_fits_header(image_header)

    if target_unit.is_equivalent(u.Jy/u.beam):
        target_beam = radio_beam.Beam.from_fits_header(target_header)
        bases = target_unit.bases
        bases.remove(u.beam)

        target_unit = bases[0] / target_beam.sr
        log.debug("Target_unit = {0}".format(target_unit))
        target_header_unit = 'Jy/beam'

        # Updated header: converting the input image to target image units
        image_header.update(target_beam.to_header_keywords())

    elif target_unit.is_equivalent(u.K) and not image_unit.is_equivalent(u.K):
        cfreq_in = image_header['REFFREQ']*u.Hz
        cfreq_target = target_header['REFFREQ']*u.Hz
        if cfreq_in != cfreq_target:
            raise ValueError("To combine images with brightness"
                             "temperature units, the observed frequency"
                             " must be specified in the header using "
                             "CRVAL3, CRPIX3, CDELT3, and CUNIT3, and "
                             "they must be the same.")
        if image_unit.is_equivalent(u.Jy/u.beam):
            image_unit = image_unit.bases[0]
            equivalency = u.brightness_temperature(beam_area=image_beam,
                                                   frequency=cfreq_in,)
        elif image_unit.is_equivalent(u.Jy/u.pixel):
            pixel_area = wcs.utils.proj_plane_pixel_area(im_wcs)*u.deg**2
            image_unit = image_unit.bases[0]
            equivalency = u.brightness_temperature(beam_area=pixel_area,
                                                   frequency=cfreq_in,)
        target_header_unit = 'K'
    elif target_unit.is_equivalent(u.Jy/u.pixel):
        raise ValueError("Jy/pixel is not an accepted unit to convert into "
                         "since it is dependent on the pixel size.  Try Jy/"
                         "sr instead?")
    else:
        target_header_unit = target_unit.to_string('FITS')

    # do this as a separate if statement because we may have converted
    # Jy/beam->Jy/sr above
    if (((hasattr(target_unit, 'unit') and
          target_unit.unit.is_equivalent(u.Jy/u.sr)) or
         target_unit.is_equivalent(u.Jy/u.sr))):
        if image_unit.is_equivalent(u.K):
            cfreq_in = image_header['REFFREQ']*u.Hz
            equivalency = u.brightness_temperature(frequency=cfreq_in,)
        elif image_unit.is_equivalent(u.Jy/u.beam):
            image_unit = image_unit.bases[0] / image_beam.sr
        elif image_unit.is_equivalent(u.Jy/u.pixel):
            pixel_area = wcs.utils.proj_plane_pixel_area(im_wcs)*u.deg**2
            image_unit = image_unit.bases[0] / pixel_area

    log.info("Converting data from {0} to {1}".format(image_unit, target_unit))
    image = u.Quantity(image, image_unit).to(target_unit, equivalency)
    image_header['BUNIT'] = target_header_unit

    return image, image_header



def flux_unit(image, header):
    """
    Convert all possible units to un-ambiguous unit like Jy/pixel or Jy/arcsec^2.

    Parameter/Return
    ----------------
    image : (float point?) array
       The input image with arbitrary flux unit (e.g. Jy/beam).
       Get converted to Jy/arcsec^2 units in output.
    header : header object
       Header of the input/output image
    """
    raise NotImplementedError

    return image, header



def regrid(hd1, im1, im2raw, hd2):
    """
    Regrid the low resolution image to have the same dimension and pixel size with the
    high resolution image.

    Parameters
    ----------
    hd1 : header object
       The header of the high resolution image
    im1 : (float point?) array
       The high resolution image
    im2raw : (float point?) array
       The pre-regridded low resolution image
    hd2 : header object
       header of the low resolution image

    Returns
    -------
    hdu2 : An object containing both the image and the header
       This will containt the regridded low resolution image, and the image header taken
       from the high resolution observation.
    im2  : (float point?) array
       The image array which stores the regridded low resolution image.
    nax1, nax2 : int(s)
       Number of pixels in each of the spatial axes.
    pixscale : float (?)
       pixel size in the input high resolution image.

    """

    # Sanity Checks:
    assert hd2['NAXIS'] == im2raw.ndim == 2, 'Error: Input lores image dimension != 2.'
    assert hd1['NAXIS'] == im1.ndim == 2, 'Error: Input hires image dimension != to 2.'

    # read pixel scale from the header of high resolution image
    #pixscale = FITS_tools.header_tools.header_to_platescale(hd1)
    #pixscale = wcs.utils.proj_plane_pixel_scales(wcs.WCS(hd1))[0]
    pixscale = wcs.utils.proj_plane_pixel_area(wcs.WCS(hd1))**0.5
    log.debug('pixscale = {0} deg'.format(pixscale))

    # read the image array size from the high resolution image
    nax1,nax2 = (hd1['NAXIS1'],
                 hd1['NAXIS2'],
                )

    # create a new HDU object to store the regridded image
    hdu2 = fits.PrimaryHDU(data=im2raw, header=hd2)

    # regrid the image
    #hdu2 = hcongrid_hdu(hdu2, hd1)
    #im2 = hdu2.data.squeeze()
    im2, coverage = reproject_interp(hdu2, hd1)

    # return hdu contains the im2 data, reprojected, with the im1 header
    rhdu = fits.PrimaryHDU(data=im2, header=hd1)

    # return variables
    return rhdu, im2, nax1, nax2, pixscale



def pbcorr(fft2, hd1, hd2):
    """
    Divide the fourier transformed low resolution image with its fourier
    transformed primary beam, and then times the fourier transformed primary
    beam of the high resolution image.

    Parameters
    ----------
    fft2 : float array
       Fourier transformed low resolution image
    hd1 : header object
       Header of the high resolution image
    hd2 : header object
       Header of the low resolution image

    Returns
    -------
    fft2 : float array
       Fourier transformed low resolution image, after corrected for the primary beam effect
    """

    return fft2



def flux_match(fft1, fft2):
    """
    Scale the flux level of the high resolution image, based on the flux level of the low
    resolution image. This is because we probably trust the flux scale from the space better,
    given that is it not affected by the atmospheric effects and the related calibrations.
    This also maintain a consistency if we want to incorporate more bands from the space
    observatory for science analysis.

    Parameters
    ----------
    fft1 : float array
       Fourier transformed high resolution image
    fft2 : float array
       Fourier transformed low resolution image

    Return
    -----------
    fft1 : float array
       Fourier transformed low resolution image after flux rescaling.
    """

    return fft1


def color_correction_factors(n_center_hi, n_center_lo, pb_hi, pb_lo, alpha):
    """
    Calculate the color correction factors for the input images before combination.
    This is to account for different nominal center frequency for different instruments, as 
    well as different flux calibration process.
    
    Parameters
    ----------
    n_center_hi:
       The nominal frequency center of high resolution image (GHz). 
       This is the default frequency center of the output correction factors.
    n_center_lo:
       The nominal frequency center of high resolution image (GHz).
    pb_hi:
       The passband curve of high resolution observation (in ../filter/**).
    pb_lo:
       The passband curve of low resolution observation.
    alpha:
       The assumed spetra index of source emission. 
     
    Return
    ------------
    cc_hi:
       Color correction factor for the high resolution image.
    cc_lo:
       Color correction factor for the low resolution image.
    """
    wv_hi, response_hi = pb_hi
    wv_lo, response_lo = pb_lo
    freq_hi = (wv_hi*u.um).to(u.GHz,equivalencies=u.spectral()).value
    freq_lo = (wv_lo*u.um).to(u.GHz,equivalencies=u.spectral()).value
    
    #calculate the color corrections according to assumed source index
    #here the default calibration for space telescope is Inum*num=const, for ground-based observations is Inum \propto num**2.
    cc_hi = np.trapz(response_hi*(freq_hi/n_center_hi)**2, freq_hi)/np.trapz(response_hi*(freq_hi/n_center_hi)**alpha, freq_hi)
    cc_lo = np.trapz(response_lo*(freq_lo/n_center_lo)**(-1), freq_lo)/np.trapz(response_lo*(freq_lo/n_center_lo)**alpha, freq_lo)
    
    #calculate the color correction for low resolution image to nominal frequency of high resolution image
    cc_lo = cc_lo*(n_center_hi/n_center_lo)**alpha
    
    
    return cc_hi, cc_lo

    
    


def feather_kernel(nax2, nax1, lowresfwhm, pixscale):
    """
    Construct the weight kernels (image arrays) for the fourier transformed low
    resolution and high resolution images.  The kernels are the fourier transforms
    of the low-resolution beam and (1-[that kernel])

    Parameters
    ----------
    nax2, nax1 : int
       Number of pixels in each axes.
    lowresfwhm : float
       Angular resolution of the low resolution image (FWHM)
    pixscale : quantity (arcsec equivalent)
       pixel size in the input high resolution image.

    Return
    ----------
    kfft : float array
       An image array containing the weighting for the low resolution image
    ikfft : float array
       An image array containing the weighting for the high resolution image
       (simply 1-kfft)
    """
    # note for Baobab:
    # pixel indices 1 and -1 correspond to a sine with period equal to the full
    # image size
    # pixel index 0 corresponds to the zero-spacing, i.e. the mean of the image


    # Construct arrays which hold the x and y coordinates (in unit of pixels)
    # of the image
    ygrid,xgrid = (np.indices([nax2,nax1]) -
                   np.array([(nax2-1.)/2,(nax1-1.)/2.])[:,None,None])

    # constant converting "resolution" in fwhm to sigma
    fwhm = np.sqrt(8*np.log(2))

    if not hasattr(pixscale, 'unit'):
        pixscale = u.Quantity(pixscale, u.deg)

    # sigma in pixels
    sigma = ((lowresfwhm/fwhm/(pixscale)).decompose().value)
    # not used, just noted that these are the theoretical values (...maybe...)
    #sigma_fftspace = (1/(4*np.pi**2*sigma**2))**0.5
    #sigma_fftspace = (2*np.pi*sigma)**-1
    #log.debug('sigma = {0}, sigma_fftspace={1}'.format(sigma, sigma_fftspace))

    # technically, the fftshift here does nothing since we're using the
    # absolute value of the kernel below, so the phase is irrelevant
    kernel = np.fft.fftshift(np.exp(-(xgrid**2+ygrid**2)/(2*sigma**2)))
    # convert the kernel, which is just a gaussian in image space,
    # to its corresponding kernel in fourier space
    kfft = np.abs(np.fft.fft2(kernel)) # should be mostly real

    if np.any(np.isnan(kfft)):
        raise ValueError("NaN value encountered in kernel")

    # normalize the kernel
    kfft/=kfft.max()
    ikfft = 1-kfft

    return kfft, ikfft



def fftmerge(kfft, ikfft, im_hi, im_lo,  lowpassfilterSD=False,
             replace_hires=False, deconvSD=False, min_beam_fraction=0.1):
    """
    Combine images in the fourier domain, and then output the combined image
    both in fourier domain and the image domain.

    Parameters
    ----------
    kernel1,2 : float array
       Weighting images.
    im1,im2: float array
       Input images.
    lowpassfilterSD: bool or str
        Re-convolve the SD image with the beam?  If ``True``, the SD image will
        be weighted by the deconvolved beam.
    replace_hires: Quantity or False
        If set, will simply replace the fourier transform of the single-dish
        data with the fourier transform of the interferometric data above the
        specified kernel level.  Can be used in conjunction with either
        ``lowpassfilterSD`` or ``deconvSD``.  Must be set to a floating-point
        threshold value; this threshold will be applied to the single-dish
        kernel.
    deconvSD: bool
        Deconvolve the single-dish data before adding in fourier space?
        This "deconvolution" is a simple division of the fourier transform
        of the single dish image by its fourier transformed beam
    min_beam_fraction : float
        The minimum fraction of the beam to include; values below this fraction
        will be discarded when deconvolving

    Returns
    -------
    fftsum : float array
       Combined image in fourier domain.
    combo  : float array
       Combined image in image domain.
    """

    fft_hi = np.fft.fft2(np.nan_to_num(im_hi))
    fft_lo = np.fft.fft2(np.nan_to_num(im_lo))

    # Combine and inverse fourier transform the images
    if lowpassfilterSD:
        lo_conv = kfft*fft_lo
    elif deconvSD:
        lo_conv = fft_lo / kfft
        lo_conv[kfft < min_beam_fraction] = 0
    else:
        lo_conv = fft_lo


    if replace_hires:
        if replace_hires is True:
            raise ValueError("If you are specifying replace_hires, "
                             "you must give a floating point value "
                             "corresponding to the beam-fraction of the "
                             "single-dish image below which the "
                             "high-resolution data will be used.")
        fftsum = lo_conv.copy()

        # mask where the hires data is above a threshold
        mask = ikfft > replace_hires

        fftsum[mask] = fft_hi[mask]
    else:
        fftsum = lo_conv + ikfft*fft_hi

    combo = np.fft.ifft2(fftsum)

    return fftsum, combo



def smoothing(combo, targres, origfwhm, pixscale):
    """
    Smooth the image to the targeted final angular resolution.

    Parameters
    ----------
    combo : float array
       Combined image
    targres : float
       The HPBW of the smoothed image (in units of arcsecond)
    origfwhm : float
       The original HPBW of input image (in units of arcsecond)
    pixscale : float
       The pixscale of the input image (in units of arcsecond)

    Returns
    -------
    combo : float array
       Smoothed image
    """
    fwhm = np.sqrt(8*np.log(2))
    kernel_size = ((targres/fwhm)**2-(hiresfwhm/fwhm)**2)**0.5
    pixel_n = kernel_size/pixscale
    
    #smooth the image using gaussian 2d kernel
    gauss_kernel = Gaussian2DKernel(pixel_n)
    combo = convolve(combo, gauss_kernel,normalize_kernel=True)
    
    return combo



def akb_plot(fft1, fft2, fftsum, outname="akb_combine.pdf"):
    """
    Generate plots for examining the combined results in fourier domain.

    Parameters
    ----------
    fft1 : float array
       Fourier transformed high resolution image
    fft2 : float array
       Fourier transformed low resolution image
    fftsum : float array
       Fourier transformed combined image
    """
    return


def casaheader(header):
    """
    Generate the header which is compatible with CASA.

    Parameters
    ----------
    header : header object
       The header of the high resolution image.

    Return
    combo_header : header object
       The generated CASA compatible header
    """
    combo_header = header
    return combo_header



def outfits(image, header, outname="output.fits"):
    """
    Output .fits format image.

    Parameters
    ----------
    image : (float point?) array
       The combined image
    header : header object
       Header of the combined image
    outname : str
       Filename of the .fits output of the combined image
    """
    hdu = fits.PrimaryHDU(data=np.abs(image), header=header)
    hdu.writeto(outname)



def freq_filling(im1, im2, hd1, hd2, hd3):
    """
    Derive spectral index from image array, and make interpolation.

    Parameters
    ----------
    im1,im2  : float array
       The input images to be interpolated
    hd1, hd2 : header object
       Headers of the input images
    hd3      : header object
       Header for extracting the targeted frequency for interpolation
    """
    interpol = im1
    interpol_header = hd1
    interpol_hdu = fits.PrimaryHDU(data=np.abs(im1), header=hd1)

    return interpol, interpol_header, interpol_hdu



#################################################################

def AKB_interpol(lores1, lores2, hires,
                 extnum1=0,
                 extnum2=0,
                 hiresextnum=0,
                 scalefactor1=1.0,
                 scalefactor2=1.0,
                 output_fits=True,
                 outfitsname='interpolate.fits'):
    """
    This procedure is provided for the case that we need to interpolate
    two space observatory image, to make the image at the observing
    frequency of the ground based one.

    Parameter
    ---------
    lores1, lores2 : str
       Filaname of the input images, either variable name of HDUs, or
       can be the .fits format files. lores2 should be at the lower observing
       frequency.
    hires : str
       Filaname of the groundbased observing image. This is to supply header
       for obtaining the targeted frequency for interpolation.
    extnum1,2 : int
       The extension number to use from the low-res FITS file
    hiresextnum : int
       The extension number to use from the hi-res FITS file
    scalefactor1,2 : float
       scaling factors of the input images.
    fitsoutput     : bool
       Option to set whether we have .fits output
    outfitsname    : str
       The filename of .fits output.

    Return
    ---------
    lores : HDU object
       The interpolated image.
    """

    # Read images
    hdu1, im1, hd1 = file_in(lores1, extnum1)
    hdu2, im2, hd2 = file_in(lores2, extnum2)
    hdu3, im3, hd3 = file_in(hires, hiresextnum)

    # Match flux unit
    im1, hd1 = flux_unit(im1, hd1)
    im2, hd2 = flux_unit(im2, hd2)

    # Smooth the high resolution image to the low resolution one
    # Here need to reead the header of the low resolution image,
    # to know what is the targeted resolution
    targres = 0.0
    origfwhm =  0.0
    pixscale = FITS_tools.header_tools.header_to_platescale(hd1)

    im1 = smoothing(im1, targres, origfwhm, pixscale)

    #* Image Registration (Match astrometry)
    #  [Should be an optional step]
    #  The initial offsets between images should not be too big. Otherwise
    #  the correlation might be trapped to a local maximum.
    # Package exist, but not sure how to use it.

    # Derive Spectral index and Make interpolation
    interpol, interpol_header, interpol_hdu = freq_filling(im1, im2, hd1, hd2, hd3)

    # output .fits file
    if output_fits:
        outfits(interpol, interpol_header, outname=outfitsname)

    # return hdu
    return interpol_hdu

#################################################################

def AKB_combine(hires, lores,
                highresextnum=0,
                lowresextnum=0,
                highresscalefactor=1.0,
                lowresscalefactor=1.0,
                lowresfwhm=1*u.arcmin,
                targres=-1.0,
                return_hdu=False,
                return_regridded_lores=False, output_fits=True):
    """
    Fourier combine two data cubes

    Parameters
    ----------
    highresfitsfile : str
        The high-resolution FITS file
    lowresfitsfile : str
        The low-resolution (single-dish) FITS file
    highresscalefactor : float
    lowresscalefactor : float
        A factor to multiply the high- or low-resolution data by to match the
        low- or high-resolution data
    lowresfwhm : `astropy.units.Quantity`
        The full-width-half-max of the single-dish (low-resolution) beam;
        or the scale at which you want to try to match the low/high resolution
        data
    return_hdu : bool
        Return an HDU instead of just an image.  It will contain two image
        planes, one for the real and one for the imaginary data.
    return_regridded_cube2 : bool
        Return the 2nd cube regridded into the pixel space of the first?
    """

    #* Input data
    hdu1, im1,    hd1 = file_in(hires, highresextnum)
    hdu2, im2raw, hd2 = file_in(lores, lowresextnum)

    # load default parameters (primary beam, the simultaneous FOV of the ground
    #                          based observations)
    # Ke Wang part. Need to think about which is the best way of doing this.
    # Here better to get the resolution information into the header (bmaj, bmin),
    # if it isn't there.

    #* Match flux unit (convert all possible units to un-ambiguous unit like Jy/pixel or Jy/arcsec^2)
    im1,    hd1 = flux_unit(im1, hd1)
    im2raw, hd2 = flux_unit(im2raw, hd2)

    # Regrid the low resolution image to the same pixel scale and
    # field of view of the high resolution image
    hdu2, im2, nax1, nax2, pixscale = regrid(hd1, im1, im2raw, hd2)

    #* Image Registration (Match astrometry)
    #  [Should be an optional step]
    #  The initial offsets between images should not be too big. Otherwise
    #  the correlation might be trapped to a local maximum.
    # Package exist, but not sure how to use it.

    # Fourier transform the images
    fft1 = np.fft.fft2(np.nan_to_num(im1*highresscalefactor))
    fft2 = np.fft.fft2(np.nan_to_num(im2*lowresscalefactor))

    #* Correct for the primary beam attenuation in fourier domain
    fft2 = pbcorr(fft2, hd1, hd2)

    #* flux matching [Use space observatory image to determine absolute flux]
    #  [should be an optional step]
    fft1 = flux_match(fft1, fft2)

    # Constructing weight kernal (normalized to max=1)
    kernel2, kernel1 = feather_kernel(nax2, nax1, lowresfwhm, pixscale)

    #* Combine images in the fourier domain
    fftsum, combo = fftmerge(kernel1, kernel2, fft1, fft2)

    #* Final Smoothing
    # [should be an optional step]
    if (targres > 0.0):
        combo = smoothing(combo, targres, origfwhm, pixscale)

    #* generate amplitude plot and PDF output
    akb_plot(fft1, fft2, fftsum)

    #* Generate the CASA 4.3 compatible header
    combo_header = casaheader(hdu1.header)

    # fits output
    if output_fits:
        outfits(combo, combo_header)

    # Return combined image array(s)
    if return_regridded_lores:
        return combo, hdu2
    else:
        return combo

#################################################################



# example
# os.system("rm -rf output.fits")
# f = AKB_combine("faint_final.shift.fix.fits","Dragon.im350.crop.fits", lowresscalefactor=0.0015,return_hdu=True)

#os.system("rm -rf output.fits")
#os.system("rm -rf interpolate.fits")
#interpol_hdu = AKB_interpol("Dragon.im350.crop.fits", "Dragon.im350.crop.fits", "faint_final.shift.fix.fits")
#f = AKB_combine("faint_final.shift.fix.fits",interpol_hdu, lowresscalefactor=0.0015,return_hdu=True)

def simple_deconvolve_sdim(hdu, lowresfwhm, minval=1e-1):
    """
    Perform a very simple fourier-space deconvolution of single-dish data.
    i.e., divide the fourier transform of the single-dish data by the fourier
    transform of its beam, and fourier transform back.

    This is not a generally useful method!
    """
    hdu_low, im_lowraw, header_low = file_in(hdu)
    nax2,nax1 = im_lowraw.shape
    pixscale = wcs.utils.proj_plane_pixel_area(wcs.WCS(header_low))**0.5
    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)

    fft_lo = (np.fft.fft2(np.nan_to_num(im_lowraw)))
    decfft_lo = fft_lo.copy()
    decfft_lo[kfft > minval] = (fft_lo / kfft)[kfft > minval]
    dec_lo = np.fft.ifft2(decfft_lo)
    return dec_lo

def simple_fourier_unsharpmask(hdu, lowresfwhm, minval=1e-1):
    """
    Like simple_deconvolve_sdim, try unsharp masking by convolving
    with (1-kfft) in the fourier domain
    """
    hdu_high, im_highraw, header_high = file_in(hdu)
    nax2,nax1 = im_highraw.shape
    pixscale = wcs.utils.proj_plane_pixel_area(wcs.WCS(header_high))**0.5
    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)

    fft_hi = (np.fft.fft2(np.nan_to_num(im_highraw)))
    #umaskfft_hi = fft_hi.copy()
    #umaskfft_hi[ikfft < minval] = (fft_hi * ikfft)[ikfft < minval]
    umaskfft_hi = fft_hi * ikfft
    umask_hi = np.fft.ifft2(umaskfft_hi)
    return umask_hi



def feather_simple(hires, lores,
                   highresextnum=0,
                   lowresextnum=0,
                   highresscalefactor=1.0,
                   lowresscalefactor=1.0,
                   pbresponse=None,
                   lowresfwhm=None,
                   lowpassfilterSD=False,
                   replace_hires=False,
                   deconvSD=False,
                   return_hdu=False,
                   return_regridded_lores=False,
                   match_units=True,
                   weights=None,
                   ):
    """
    Fourier combine two single-plane images.  This follows the CASA approach,
    as far as it is discernable.  Both images should be actual images of the
    sky in the same units, not fourier models or deconvolved images.

    The default parameters follow Equation 11 in section 5.2 of
    http://esoads.eso.org/abs/2002ASPC..278..375S
    very closely, except the interferometric data are not re-convolved with the
    interferometric beam.  It seems that S5.2 of Stanimirovic et al 2002
    actually wants the deconvolved *model* data, not the deconvolved *clean*
    data, as input, which would explain their equation.  This may be a
    different use of the same words.  ``lowresscalefactor`` corresponds to
    their ``f`` parameter.

    There is a remaining question: does CASA-feather re-weight the SD image by
    its beam, as suggested in
    https://science.nrao.edu/science/meetings/2016/vla-data-reduction/DR2016imaging_jott.pdf
    page 24, or does it leave the single-dish image unweighted (assuming its
    beam size is the same as the effective beam size of the image) as inferred
    from the Feather source code?

    Parameters
    ----------
    highresfitsfile : str
        The high-resolution FITS file
    lowresfitsfile : str
        The low-resolution (single-dish) FITS file
    highresextnum : int
        The extension number to use from the high-res FITS file
    lowresextnum : int
        The extension number to use from the low-res FITS file
    highresscalefactor : float
        A factor to multiply the high-resolution data by to match the
        low- or high-resolution data
    lowresscalefactor : float
        A factor to multiply the low-resolution data by to match the
        low- or high-resolution data
    pbresponse : `~numpy.ndarray`
        The primary beam response of the high-resolution data. When given,
        `highresfitsfile` should **not** be primary-beam corrected.
        `pbresponse` will be multiplied with `lowresfitsfile`, and the
        feathered image will be divided by `pbresponse` to create the final
        image.
    lowresfwhm : `astropy.units.Quantity`
        The full-width-half-max of the single-dish (low-resolution) beam;
        or the scale at which you want to try to match the low/high resolution
        data
    lowpassfilterSD: bool or str
        Re-convolve the SD image with the beam?  If ``True``, the SD image will
        be weighted by the beam, which effectively means it will be convolved
        with the beam before merging with the interferometer data.  This isn't
        really the right behavior; it should be filtered with the *deconvolved*
        beam, but in the current framework that effectively means "don't weight
        the single dish data".  See
        http://keflavich.github.io/blog/what-does-feather-do.html
        for further details about what feather does and how this relates.
    replace_hires: Quantity or False
        If set, will simply replace the fourier transform of the single-dish
        data with the fourier transform of the interferometric data above the
        specified kernel level.  Can be used in conjunction with either
        ``lowpassfilterSD`` or ``deconvSD``.  Must be set to a floating-point
        threshold value; this threshold will be applied to the single-dish
        kernel.
    deconvSD: bool
        Deconvolve the single-dish data before adding in fourier space?
        This "deconvolution" is a simple division of the fourier transform
        of the single dish image by its fourier transformed beam
    return_hdu : bool
        Return an HDU instead of just an image.  It will contain two image
        planes, one for the real and one for the imaginary data.
    return_regridded_lores : bool
        Return the 2nd image regridded into the pixel space of the first?
    match_units : bool
        Attempt to match the flux units between the files before combining?
        See `match_flux_units`.
    weights : `~numpy.ndarray`, optional
        Provide an array of weights with the spatial shape of the high-res
        data. This is useful when either of the data have emission at the map
        edge, which will lead to ringing in the Fourier transform. A weights
        array can be provided to smoothly taper the edges of each map to avoid
        this issue. **This will be applied to both the low and high resolution
        images!**

    Returns
    -------
    combo : image
        The image of the combined low and high resolution data sets
    combo_hdu : fits.PrimaryHDU
        (optional) the image encased in a FITS HDU with the relevant header
    """
    hdu_hi, im_hi, header_hi = file_in(hires, highresextnum)
    hdu_low, im_lowraw, header_low = file_in(lores, lowresextnum)

    if lowresfwhm is None:
        beam_low = radio_beam.Beam.from_fits_header(header_low)
        lowresfwhm = beam_low.major
        log.info("Low-res FWHM: {0}".format(lowresfwhm))

    # If weights are given, they must match the shape of the hires data
    if weights is not None:
        if not weights.shape == im_hi.shape:
            raise ValueError("weights must be an array with the same shape as"
                             " the high-res data.")
    else:
        weights = 1.

    if pbresponse is not None:
        if not pbresponse.shape == im_hi.shape:
            raise ValueError("pbresponse must be an array with the same"
                             " shape as the high-res data.")

    if match_units:
        # After this step, the units of im_hi are some sort of surface brightness
        # unit equivalent to that specified in the high-resolution header's units
        # Note that this step does NOT preserve the values of im_lowraw and
        # header_lowraw from above
        im_lowraw, header_low = match_flux_units(image=im_lowraw,
                                                 image_header=header_low.copy(),
                                                 target_header=header_hi)

    hdu_low, im_low, nax1, nax2, pixscale = regrid(hd1=header_hi,
                                                   im1=im_hi,
                                                   im2raw=im_lowraw,
                                                   hd2=header_low)

    # Apply the pbresponse to the regridded low-resolution data
    if pbresponse is not None:
        im_low *= pbresponse

    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale,)

    fftsum, combo = fftmerge(kfft, ikfft,
                             im_hi * highresscalefactor * weights,
                             im_low * lowresscalefactor * weights,
                             replace_hires=replace_hires,
                             lowpassfilterSD=lowpassfilterSD,
                             deconvSD=deconvSD,
                             )

    # Divide by the
    if pbresponse is not None:
        combo /= pbresponse

    if return_hdu:
        combo_hdu = fits.PrimaryHDU(data=combo.real, header=hdu_hi.header)
        combo = combo_hdu

    if return_regridded_lores:
        return combo, hdu_low
    else:
        return combo

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

    hdu_hi, im_hi, header_hi = file_in(hires)
    hdu_low, im_lowraw, header_low = file_in(lores)

    if lowresfwhm is None:
        beam_low = radio_beam.Beam.from_fits_header(header_low)
        lowresfwhm = beam_low.major
        log.info("Low-res FWHM: {0}".format(lowresfwhm))

    if match_units:
        # After this step, the units of im_hi are some sort of surface brightness
        # unit equivalent to that specified in the high-resolution header's units
        # Note that this step does NOT preserve the values of im_lowraw and
        # header_lowraw from above
        im_lowraw, header_low = match_flux_units(image=im_lowraw,
                                                 image_header=header_low.copy(),
                                                 target_header=header_hi)

    hdu_low, im_low, nax1, nax2, pixscale = regrid(hd1=header_hi,
                                                   im1=im_hi,
                                                   im2raw=im_lowraw,
                                                   hd2=header_low)

    missing_flux = im_low - convolve(im_hi, beam_low.as_kernel(pixscale))

    combo = missing_flux + im_hi

    if return_hdu:
        combo_hdu = fits.PrimaryHDU(data=combo.real, header=hdu_hi.header)
        combo = combo_hdu

    if return_regridded_lores:
        return combo, hdu_low
    else:
        return combo

def feather_plot(hires, lores,
                 highresextnum=0,
                 lowresextnum=0,
                 highresscalefactor=1.0,
                 lowresscalefactor=1.0,
                 lowresfwhm=None,
                 lowpassfilterSD=False,
                 xaxisunit='arcsec',
                 hires_threshold=None,
                 lores_threshold=None,
                 match_units=True,
                ):
    """
    Plot the power spectra of two images that would be combined
    along with their weights.

    High-res will be shown in red, low-res in blue.

    Parameters
    ----------
    highresfitsfile : str
        The high-resolution FITS file
    lowresfitsfile : str
        The low-resolution (single-dish) FITS file
    highresextnum : int
        The extension number to use from the high-res FITS file
    highresscalefactor : float
    lowresscalefactor : float
        A factor to multiply the high- or low-resolution data by to match the
        low- or high-resolution data
    lowresfwhm : `astropy.units.Quantity`
        The full-width-half-max of the single-dish (low-resolution) beam;
        or the scale at which you want to try to match the low/high resolution
        data
    xaxisunit : 'arcsec' or 'lambda'
        The X-axis units.  Either arcseconds (angular scale on the sky)
        or baseline length (lambda)
    hires_threshold : float or None
    lores_threshold : float or None
        Threshold to cut off before computing power spectrum to remove
        the noise contribution.  Threshold will be applied *after* scalefactor.
    match_units : bool
        Attempt to match the flux units between the files before combining?
        See `match_flux_units`.

    Returns
    -------
    combo : image
        The image of the combined low and high resolution data sets
    combo_hdu : fits.PrimaryHDU
        (optional) the image encased in a FITS HDU with the relevant header
    """
    import image_tools
    hdu_hi, im_hi, header_hi = file_in(hires)
    hdu_low, im_lowraw, header_low = file_in(lores)

    print("featherplot")
    pb = ProgressBar(13)

    if match_units:
        # After this step, the units of im_hi are some sort of surface brightness
        # unit equivalent to that specified in the high-resolution header's units
        # Note that this step does NOT preserve the values of im_lowraw and
        # header_lowraw from above
        im_lowraw, header_low = match_flux_units(image=im_lowraw,
                                                 image_header=header_low.copy(),
                                                 target_header=header_hi)

    hdu_low, im_low, nax1, nax2, pixscale = regrid(header_hi, im_hi,
                                                   im_lowraw, header_low)
    pb.update()

    if lowresfwhm is None:
        beam_low = radio_beam.Beam.from_fits_header(header_low)
        lowresfwhm = beam_low.major
        log.info("Low-res FWHM: {0}".format(lowresfwhm))

    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)
    log.debug("bottom-left pixel before shifting: kfft={0}, ikfft={1}".format(kfft[0,0], ikfft[0,0]))
    print("bottom-left pixel before shifting: kfft={0}, ikfft={1}".format(kfft[0,0], ikfft[0,0]))
    pb.update()
    kfft = np.fft.fftshift(kfft)
    pb.update()
    ikfft = np.fft.fftshift(ikfft)
    pb.update()

    if hires_threshold is None:
        fft_hi = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_hi*highresscalefactor)))
    else:
        hires_tofft = np.nan_to_num(im_hi*highresscalefactor)
        hires_tofft[hires_tofft < hires_threshold] = 0
        fft_hi = np.fft.fftshift(np.fft.fft2(hires_tofft))
    pb.update()
    if lores_threshold is None:
        fft_lo = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_low*lowresscalefactor)))
    else:
        lores_tofft = np.nan_to_num(im_low*lowresscalefactor)
        lores_tofft[lores_tofft < lores_threshold] = 0
        fft_lo = np.fft.fftshift(np.fft.fft2(lores_tofft))
    pb.update()

    rad,azavg_kernel = image_tools.radialprofile.azimuthalAverage(np.abs(kfft), returnradii=True)
    pb.update()
    rad,azavg_ikernel = image_tools.radialprofile.azimuthalAverage(np.abs(ikfft), returnradii=True)
    pb.update()
    rad,azavg_hi = image_tools.radialprofile.azimuthalAverage(np.abs(fft_hi), returnradii=True)
    pb.update()
    rad,azavg_lo = image_tools.radialprofile.azimuthalAverage(np.abs(fft_lo), returnradii=True)
    pb.update()
    rad,azavg_hi_scaled = image_tools.radialprofile.azimuthalAverage(np.abs(fft_hi*ikfft), returnradii=True)
    pb.update()
    rad,azavg_lo_scaled = image_tools.radialprofile.azimuthalAverage(np.abs(fft_lo*kfft), returnradii=True)
    pb.update()
    rad,azavg_lo_deconv = image_tools.radialprofile.azimuthalAverage(np.abs(fft_lo/kfft), returnradii=True)
    pb.update()

    # use the same "OK" mask for everything because it should just be an artifact
    # of the averaging
    OK = np.isfinite(azavg_kernel)

    # 1/min(rad) ~ number of pixels from center to corner of image
    # pixscale in degrees.  Convert # pixels to arcseconds
    # 2 pixels = where rad is 1/2 the image (square) dimensions
    # (nax1) pixels = where rad is 1
    # *** ASSUMES SQUARE ***
    rad_pix = nax1/rad
    rad_as = pixscale * 3600 * rad_pix
    log.debug("pixscale={0} nax1={1}".format(pixscale, nax1))
    if xaxisunit == 'lambda':
        #restfrq = (wcs.WCS(hd1).wcs.restfrq*u.Hz)
        lam = 1./(rad_as*u.arcsec).to(u.rad).value
        xaxis = lam
    elif xaxisunit == 'arcsec':
        xaxis = rad_as
    else:
        raise ValueError("xaxisunit must be in (arcsec, lambda)")


    import pylab as pl

    pl.clf()
    ax1 = pl.subplot(2,1,1)
    ax1.loglog(xaxis[OK], azavg_kernel[OK], color='b', linewidth=2, alpha=0.8,
               label="Low-res Kernel")
    ax1.loglog(xaxis[OK], azavg_ikernel[OK], color='r', linewidth=2, alpha=0.8,
               label="High-res Kernel")
    ax1.vlines(lowresfwhm.to(u.arcsec).value, 1e-5, 1.1, linestyle='--', color='k')
    ax1.set_ylim(1e-5, 1.1)
    arg_xmin = np.nanargmin(np.abs((azavg_ikernel)-(1-1e-5)))
    xlim = xaxis[arg_xmin]/1.1, xaxis[1]*1.1
    log.debug("Xlim: {0}".format(xlim))
    assert all(np.isfinite(xlim))
    ax1.set_xlim(*xlim)

    ax1.set_ylabel("Kernel Weight")


    ax2 = pl.subplot(2,1,2)
    ax2.loglog(xaxis[OK], azavg_lo[OK], color='b', linewidth=2, alpha=0.8,
               label="Low-res image")
    ax2.loglog(xaxis[OK], azavg_hi[OK], color='r', linewidth=2, alpha=0.8,
               label="High-res image")
    ax2.set_xlim(*xlim)
    ax2.set_ylim(min([azavg_lo[arg_xmin], azavg_lo[2], azavg_hi[arg_xmin], azavg_hi[2]]),
                 1.1*max([np.nanmax(azavg_lo), np.nanmax(azavg_hi)]),
                )
    ax2.set_ylabel("Power spectrum $|FT|$")

    ax3 = pl.subplot(2,1,2)
    ax3.loglog(xaxis[OK], azavg_lo_scaled[OK], color='b', linewidth=2, alpha=0.5,
               linestyle='--',
               label="Low-res scaled image")
    ax3.loglog(xaxis[OK], azavg_lo_deconv[OK], color='b', linewidth=2, alpha=0.5,
               linestyle=':',
               label="Low-res deconvolved image")
    ax3.loglog(xaxis[OK], azavg_hi_scaled[OK], color='r', linewidth=2, alpha=0.5,
               linestyle='--',
               label="High-res scaled image")
    ax3.set_xlim(*xlim)
    if xaxisunit == 'arcsec':
        ax3.set_xlabel("Angular Scale (arcsec)")
    elif xaxisunit == 'lambda':
        ax3.set_xscale('linear')
        ax3.set_xlim(0, xlim[0])
        ax3.set_xlabel("Baseline Length (lambda)")
    ax3.set_ylim(min([azavg_lo_scaled[arg_xmin], azavg_lo_scaled[2], azavg_hi_scaled[arg_xmin], azavg_hi_scaled[2]]),
                 1.1*max([np.nanmax(azavg_lo_scaled), np.nanmax(azavg_hi_scaled)]),
                )

    return {'radius':rad,
            'radius_as': rad_as,
            'azimuthally_averaged_kernel': azavg_kernel,
            'azimuthally_averaged_inverse_kernel': azavg_ikernel,
            'azimuthally_averaged_low_resolution': azavg_lo,
            'azimuthally_averaged_high_resolution': azavg_hi,
            'azimuthally_averaged_low_res_filtered': azavg_lo_scaled,
            'azimuthally_averaged_high_res_filtered': azavg_hi_scaled,
           }

def spectral_regrid(cube, outgrid):
    """
    Spectrally regrid a cube onto a new spectral output grid

    (this is redundant with regrid_cube_hdu, but will work independently if you
    already have spatially matched frames)

    Parameters
    ----------
    cube : SpectralCube
        A SpectralCube object to regrid
    outgrid : array
        An array of the spectral positions to regrid onto

    Returns
    -------
    cube : fits.PrimaryHDU
        An HDU containing the output cube in FITS HDU form
    """

    assert isinstance(cube, SpectralCube)

    inaxis = cube.spectral_axis.to(outgrid.unit)

    indiff = np.mean(np.diff(inaxis))
    outdiff = np.mean(np.diff(outgrid))
    if outdiff < 0:
        outgrid=outgrid[::-1]
        outdiff = np.mean(np.diff(outgrid))
    if indiff < 0:
        cubedata = cube.filled_data[::-1]
        inaxis = cube.spectral_axis.to(outgrid.unit)[::-1]
        indiff = np.mean(np.diff(inaxis))
    else:
        cubedata = cube.filled_data[:]
    if indiff < 0 or outdiff < 0:
        raise ValueError("impossible.")

    assert np.all(np.diff(outgrid) > 0)
    assert np.all(np.diff(inaxis) > 0)

    np.testing.assert_allclose(np.diff(outgrid), outdiff,
                               err_msg="Output grid must be linear")

    if outdiff > 2 * indiff:
        raise ValueError("Input grid has too small a spacing.  It needs to be "
                         "smoothed prior to resampling.")

    newcube = np.empty([outgrid.size, cube.shape[1], cube.shape[2]])

    yy,xx = np.indices(cube.shape[1:])

    log.info("Regridding images.")
    pb = ProgressBar(xx.size)
    for ix, iy in (zip(xx.flat, yy.flat)):
        newcube[:,iy,ix] = np.interp(outgrid.value, inaxis.value,
                                     cubedata[:,iy,ix].value)
        pb.update()

    newheader = cube.header
    newheader['CRPIX3'] = 1
    newheader['CRVAL3'] = outgrid[0].value
    newheader['CDELT3'] = outdiff.value
    newheader['CUNIT3'] = outgrid.unit.to_string('FITS')

    return fits.PrimaryHDU(data=newcube, header=newheader)


def spectral_smooth_and_downsample(cube, kernelfwhm):
    """
    Smooth the cube along the spectral axis by a specific Gaussian kernel, then
    downsample by an integer factor that still nyquist samples the smoothed
    data.

    Parameters
    ----------
    cube : SpectralCube
        A SpectralCube object to regrid
    kernelfwhm : float
        the full-width-half-max of the spectral kernel in pixels

    Returns
    -------
    cube_ds_hdu : fits.PrimaryHDU
        An HDU containing the output cube in FITS HDU form
    """

    kernelwidth = kernelfwhm / np.sqrt(8*np.log(2))

    # there may not be an alternative to this anywhere in the astropy ecosystem
    cube_smooth = FITS_tools.cube_regrid.spectral_smooth_cube(cube,
                                                              kernelwidth)
    log.debug("completed cube smooth")

    integer_dsfactor = int(np.floor(kernelfwhm))

    cube_ds = cube_smooth[::integer_dsfactor,:,:]
    log.debug("downsampled")
    (cube.wcs.wcs)
    (cube.wcs)
    log.debug("wcs'd")
    cube.filled_data[:]
    log.debug("filled_data")
    (cube.hdu) # this is a hack to prevent abort traps (never figured out why these happened)
    log.debug("hdu'd")
    cube.hdu # this is a hack to prevent abort traps (never figured out why these happened)
    log.debug("hdu'd again")
    cube_ds_hdu = cube.hdu
    log.debug("made hdu")
    cube_ds_hdu.data = cube_ds
    log.debug("put data in hdu")
    cube_ds_hdu.header['CRPIX3'] = 1
    # why min? because we're forcing CDELT3 to be positive, therefore the 0'th channel
    # must be the reference value.  Since we're using a symmetric kernel to downsample,
    # the reference channel - wherever we pick it - must stay fixed.
    cube_ds_hdu.header['CRVAL3'] = cube.spectral_axis[0].to(u.Hz).value
    cube_ds_hdu.header['CUNIT3'] = cube.spectral_axis[0].to(u.Hz).unit.to_string('FITS')
    cube_ds_hdu.header['CDELT3'] = cube.wcs.wcs.cdelt[2] * integer_dsfactor
    log.debug("completed header making")

    return cube_ds_hdu

def fourier_combine_cubes(cube_hi, cube_lo, highresextnum=0,
                          highresscalefactor=1.0,
                          lowresscalefactor=1.0, lowresfwhm=1*u.arcmin,
                          return_regridded_cube_lo=False,
                          return_hdu=False,
                          maximum_cube_size=1e8,
                         ):
    """
    Fourier combine two data cubes

    Parameters
    ----------
    cube_hi : SpectralCube
    highresfitsfile : str
        The high-resolution FITS file
    cube_lo : SpectralCube
    lowresfitsfile : str
        The low-resolution (single-dish) FITS file
    highresextnum : int
        The extension number to use from the high-res FITS file
    highresscalefactor : float
    lowresscalefactor : float
        A factor to multiply the high- or low-resolution data by to match the
        low- or high-resolution data
    lowresfwhm : `astropy.units.Quantity`
        The full-width-half-max of the single-dish (low-resolution) beam;
        or the scale at which you want to try to match the low/high resolution
        data
    return_hdu : bool
        Return an HDU instead of just a cube.  It will contain two image
        planes, one for the real and one for the imaginary data.
    return_regridded_cube_lo : bool
        Return the 2nd cube regridded into the pixel space of the first?
    """
    if isinstance(cube_hi, str):
        cube_hi = SpectralCube.read(cube_hi)
    if isinstance(cube_lo, str):
        cube_lo = SpectralCube.read(cube_lo)

    if cube_hi.size > maximum_cube_size:
        raise ValueError("Cube is probably too large "
                         "for this operation to work in memory")

    im_hi = cube_hi._data # want the raw data for this
    hd_hi = cube_hi.header
    assert hd_hi['NAXIS'] == im_hi.ndim == 3
    wcs_hi = cube_hi.wcs
    pixscale = FITS_tools.header_tools.header_to_platescale(hd_hi)

    cube_lo = cube_lo.to(cube_hi.unit)

    assert cube_hi.unit == cube_lo.unit, 'Cubes must have same or equivalent unit'
    assert cube_hi.unit.is_equivalent(u.Jy/u.beam) or cube_hi.unit.is_equivalent(u.K), "Cubes must have brightness units."

    #fitshdu_low = regrid_fits_cube(lowresfitsfile, hd_hi)
    log.info("Regridding cube (this step may take a while)")
    # old version, using FITS_tools
    #fitshdu_low = regrid_cube_hdu(cube_lo.hdu, hd_hi)
    # new version, using reproject & spectral-cube
    cube_lo_rg = cube_lo.reproject(hd_hi)
    fitshdu_low = cube_lo_rg.hdu
    #w2 = wcs.WCS(fitshdu_low.header)

    nax1,nax2 = (hd_hi['NAXIS1'],
                 hd_hi['NAXIS2'],
                 )

    dcube_hi = im_hi
    dcube_lo = fitshdu_low.data
    outcube = np.empty_like(dcube_hi)

    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)

    log.info("Fourier combining each of {0} slices".format(dcube_hi.shape[0]))
    pb = ProgressBar(dcube_hi.shape[0])

    for ii,(slc_hi,slc_lo) in enumerate(zip(dcube_hi, dcube_lo)):

        fftsum, combo = fftmerge(kfft, ikfft, slc_hi*highresscalefactor,
                                 slc_lo*lowresscalefactor)

        outcube[ii,:,:] = combo.real

        pb.update(ii+1)

    if return_regridded_cube_lo:
        return outcube, fitshdu_low
    elif return_hdu:
        return fits.PrimaryHDU(data=outcube, header=wcs_hi.to_header())
    else:
        return outcube

def feather_compare(hires, lores,
                    SAS,
                    LAS,
                    lowresfwhm,
                    beam_divide_lores=True,
                    lowpassfilterSD=False,
                    min_beam_fraction=0.1,
                    plot_min_beam_fraction=1e-3,
                    doplot=True,
                    return_samples=False,
                    weights=None,
                   ):
    """
    Compare the single-dish and interferometer data over the region where they
    should agree

    Parameters
    ----------
    highresfitsfile : str
        The high-resolution FITS file
    lowresfitsfile : str
        The low-resolution (single-dish) FITS file
    SAS : `astropy.units.Quantity`
        The smallest angular scale to plot
    LAS : `astropy.units.Quantity`
        The largest angular scale to plot (probably the LAS of the high
        resolution data)
    lowresfwhm : `astropy.units.Quantity`
        The full-width-half-max of the single-dish (low-resolution) beam;
        or the scale at which you want to try to match the low/high resolution
        data
    beam_divide_lores: bool
        Divide the low-resolution data by the beam weight before plotting?
        (should do this: otherwise, you are plotting beam-downweighted data)
    min_beam_fraction : float
        The minimum fraction of the beam to include; values below this fraction
        will be discarded when deconvolving
    plot_min_beam_fraction : float
        Like min_beam_fraction, but used only for plotting
    doplot : bool
        If true, make plots.  Otherwise will just return the results.
    return_samples : bool, optional
        Return the samples in the overlap region. This includes: the angular
        scale at each point, the ratio, the high-res values, and the low-res
        values.
    weights : `~numpy.ndarray`, optional
        Provide an array of weights with the spatial shape of the high-res
        data. This is useful when either of the data have emission at the map
        edge, which will lead to ringing in the Fourier transform. A weights
        array can be provided to smoothly taper the edges of each map to avoid
        this issue.

    Returns
    -------
    stats : dict
        Statistics on the ratio of the high-resolution FFT data to the
        low-resolution FFT data over the range SAS < x < LAS.  Sigma-clipped
        stats are included.

    """
    assert LAS > SAS

    hdu_hi, im_hi, header_hi = file_in(hires)
    hdu_low, im_lowraw, header_low = file_in(lores)

    # If weights are given, they must match the shape of the hires data
    if weights is not None:
        if not weights.shape == im_hi.shape:
            raise ValueError("weights must be an array with the same shape as"
                             " the high-res data.")
    else:
        weights = 1.

    hdu_low, im_low, nax1, nax2, pixscale = regrid(header_hi, im_hi,
                                                   im_lowraw, header_low)

    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)
    kfft = np.fft.fftshift(kfft)
    ikfft = np.fft.fftshift(ikfft)

    yy,xx = np.indices([nax2, nax1])
    rr = ((xx-(nax1-1)/2.)**2 + (yy-(nax2-1)/2.)**2)**0.5
    angscales = nax1/rr * pixscale*u.deg

    fft_hi = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_hi * weights)))
    fft_lo = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_low * weights)))
    if beam_divide_lores:
        fft_lo_deconvolved = fft_lo / kfft
    else:
        fft_lo_deconvolved = fft_lo

    below_beamscale = kfft < min_beam_fraction
    below_beamscale_plotting = kfft < plot_min_beam_fraction
    fft_lo_deconvolved[below_beamscale_plotting] = np.nan

    mask = (angscales > SAS) & (angscales < LAS) & (~below_beamscale)
    assert mask.sum() > 0

    ratio = np.abs(fft_hi)[mask] / np.abs(fft_lo_deconvolved)[mask]
    sclip = stats.sigma_clipped_stats(ratio, sigma=3, maxiters=5)

    if doplot:
        import pylab as pl
        pl.clf()
        pl.suptitle("{0} - {1}".format(SAS,LAS))
        pl.subplot(2,2,1)
        pl.plot(np.abs(fft_hi)[mask], np.abs(fft_lo_deconvolved)[mask], '.')
        mm = np.array([np.abs(fft_hi)[mask].min(), np.abs(fft_hi)[mask].max()])
        pl.plot(mm, mm, 'k--')
        pl.plot(mm, mm/sclip[1], 'k:')
        pl.xlabel("High-resolution")
        pl.ylabel("Low-resolution")
        pl.subplot(2,2,2)

        pl.hist(ratio[np.isfinite(ratio)], bins=30)
        pl.xlabel("High-resolution / Low-resolution")
        pl.subplot(2,1,2)
        srt = np.argsort(angscales.to(u.arcsec).value[~below_beamscale_plotting])
        pl.plot(angscales.to(u.arcsec).value[~below_beamscale_plotting][srt],
                np.nanmax(np.abs(fft_lo_deconvolved))*kfft.real[~below_beamscale_plotting][srt],
                'k-', zorder=-5)
        pl.loglog(angscales.to(u.arcsec).value, np.abs(fft_hi), 'r,', alpha=0.5, label='High-res')
        pl.loglog(angscales.to(u.arcsec).value, np.abs(fft_lo_deconvolved), 'b,', alpha=0.5, label='Lo-res')
        ylim = pl.gca().get_ylim()
        pl.vlines([SAS.to(u.arcsec).value,LAS.to(u.arcsec).value],
                  ylim[0], ylim[1], linestyle='-', color='k')
        #pl.legend(loc='best')

    if return_samples:
        return (angscales.to(u.arcsec)[mask], ratio, np.abs(fft_hi)[mask],
                np.abs(fft_lo_deconvolved)[mask])

    return {'median': np.nanmedian(ratio),
            'mean': np.nanmean(ratio),
            'std': np.nanstd(ratio),
            'mean_sc': sclip[0],
            'median_sc': sclip[1],
            'std_sc': sclip[2],
           }

def angular_range_image_comparison(hires, lores, SAS, LAS, lowresfwhm,
                                   beam_divide_lores=True,
                                   lowpassfilterSD=False,
                                   min_beam_fraction=0.1,
                                   plot_min_beam_fraction=1e-3, doplot=True,):
    """
    Compare the single-dish and interferometer data over the region where they
    should agree, but do the comparison in image space!

    Parameters
    ----------
    highresfitsfile : str
        The high-resolution FITS file
    lowresfitsfile : str
        The low-resolution (single-dish) FITS file
    SAS : `astropy.units.Quantity`
        The smallest angular scale to plot
    LAS : `astropy.units.Quantity`
        The largest angular scale to plot (probably the LAS of the high
        resolution data)
    lowresfwhm : `astropy.units.Quantity`
        The full-width-half-max of the single-dish (low-resolution) beam;
        or the scale at which you want to try to match the low/high resolution
        data
    beam_divide_lores: bool
        Divide the low-resolution data by the beam weight before plotting?
        (should do this: otherwise, you are plotting beam-downweighted data)
    min_beam_fraction : float
        The minimum fraction of the beam to include; values below this fraction
        will be discarded when deconvolving
    plot_min_beam_fraction : float
        Like min_beam_fraction, but used only for plotting
    doplot : bool
        If true, make plots.  Otherwise will just return the results.

    Returns
    -------
    scalefactor : float
        The mean ratio of the high- to the low-resolution image over the range
        of shared angular sensitivity weighted by the low-resolution image
        intensity over that range.  The weighting is necessary to avoid errors
        introduced by the fact that these images are forced to have zero means.
    """
    assert LAS > SAS
    hdu_hi, im_hi, header_hi = file_in(hires)
    hdu_low, im_lowraw, header_low = file_in(lores)

    hdu_low, im_low, nax1, nax2, pixscale = regrid(header_hi, im_hi,
                                                   im_lowraw, header_low)

    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)
    kfft = np.fft.fftshift(kfft)
    ikfft = np.fft.fftshift(ikfft)

    yy,xx = np.indices([nax2, nax1])
    rr = ((xx-(nax1-1)/2.)**2 + (yy-(nax2-1)/2.)**2)**0.5
    angscales = nax1/rr * pixscale*u.deg

    fft_hi = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_hi)))
    fft_lo = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_low)))
    if beam_divide_lores:
        fft_lo_deconvolved = fft_lo / kfft
    else:
        fft_lo_deconvolved = fft_lo

    below_beamscale = kfft < min_beam_fraction
    below_beamscale_plotting = kfft < plot_min_beam_fraction
    fft_lo_deconvolved[below_beamscale_plotting] = np.nan

    mask = (angscales > SAS) & (angscales < LAS) & (~below_beamscale)
    assert mask.sum() > 0

    hi_img_ring = (np.fft.ifft2(np.fft.fftshift(fft_hi*mask)))
    lo_img_ring = (np.fft.ifft2(np.fft.fftshift(fft_lo*mask)))
    lo_img_ring_deconv = (np.fft.ifft2(np.fft.fftshift(np.nan_to_num(fft_lo_deconvolved*mask))))

    lo_img = lo_img_ring_deconv if beam_divide_lores else lo_img_ring

    ratio = (hi_img_ring.real).ravel() / (lo_img.real).ravel()
    sd_weighted_mean_ratio = (((lo_img.real.ravel())**2 * ratio).sum() /
                              ((lo_img.real.ravel())**2).sum())

    return sd_weighted_mean_ratio


def scale_comparison(original_image, test_image, scales, sm_orig=True):
    """
    Compare the 'test_image' to the original image as a function of scale (in
    pixel units)

    (warning: kinda slow)
    """

    chi2s = []

    for scale in scales:

        kernel = Gaussian2DKernel(scale)

        sm_img = convolve_fft(test_image, kernel)
        if sm_orig:
            sm_orig_img = convolve_fft(original_image, kernel)
        else:
            sm_orig_img = original_image

        chi2 = (((sm_img-sm_orig_img)/sm_orig_img)**2).sum()

        chi2s.append(chi2)

    return np.array(chi2s)
