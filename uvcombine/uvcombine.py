import image_tools
from reproject import reproject_interp
from spectral_cube import SpectralCube
from spectral_cube import wcs_utils
from astropy.io import fits
from astropy import units as u
from astropy import log
from astropy.utils.console import ProgressBar
import numpy as np
#import FITS_tools
#from FITS_tools.hcongrid import hcongrid_hdu
#from FITS_tools.cube_regrid import regrid_cube_hdu
from astropy import wcs

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
   
    return hdu, im, header



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
    assert hd2['NAXIS'] == im2raw.ndim == 2, 'Error: Input lores image dimension non-equal to 2.'
    assert hd1['NAXIS'] == im1.ndim == 2, 'Error: Input hires image dimension non-equal to 2.'

    # read pixel scale from the header of high resolution image
    pixscale = FITS_tools.header_tools.header_to_platescale(hd1)
    log.debug('pixscale = {0}'.format(pixscale))

    # read the image array size from the high resolution image
    nax1,nax2 = (hd1['NAXIS1'],
                 hd1['NAXIS2'],
                )

    # create a new HDU object to store the regridded image
    hdu2 = fits.PrimaryHDU(data=im2raw, header=hd2)

    # regrid the image
    #hdu2 = hcongrid_hdu(hdu2, hd1)
    #im2 = hdu2.data.squeeze()
    im2 = reproject_interp(hdu2, hd1)

    # return variables
    return hdu2, im2, nax1, nax2, pixscale



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
    pixscale : float (?)
       pixel size in the input high resolution image.

    Return
    ----------
    kfft : float array
       An image array containing the weighting for the low resolution image
    ikfft : float array
       An image array containing the weighting for the high resolution image
       (simply 1-kfft)
    """
    # Construct arrays which hold the x and y coordinates (in unit of pixels)
    # of the image
    ygrid,xgrid = (np.indices([nax2,nax1]) -
                   np.array([(nax2-1.)/2,(nax1-1.)/2.])[:,None,None])

    # constant converting "resolution" in fwhm to sigma
    fwhm = np.sqrt(8*np.log(2))

    # sigma in pixels
    sigma = ((lowresfwhm/fwhm/(pixscale*u.deg)).decompose().value)
    # not used, just noted that these are the theoretical values (...maybe...)
    #sigma_fftspace = (1/(4*np.pi**2*sigma**2))**0.5
    #sigma_fftspace = (2*np.pi*sigma)**-1
    #log.debug('sigma = {0}, sigma_fftspace={1}'.format(sigma, sigma_fftspace))

    kernel = np.fft.fftshift(np.exp(-(xgrid**2+ygrid**2)/(2*sigma**2)))
    # convert the kernel, which is just a gaussian in image space,
    # to its corresponding kernel in fourier space
    kfft = np.abs(np.fft.fft2(kernel)) # should be mostly real
    # normalize the kernel
    kfft/=kfft.max()
    ikfft = 1-kfft

    return kfft, ikfft



def fftmerge(kfft,ikfft,im_hi,im_lo):
    """
    Combine images in the fourier domain, and then output the combined image
    both in fourier domain and the image domain.

    Parameters
    ----------
    kernel1,2 : float array
       Weighting images.
    im1,im2: float array
       Input images.

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
    fftsum = kfft*fft_lo + ikfft*fft_hi

    combo = np.fft.ifft2(fftsum)

    return fftsum, combo



def smoothing(combo, targres):
    """
    Smooth the image to the targeted final angular resolution.

    Parameters
    ----------
    combo : float array
       Combined image
    targres : float
       The HPBW of the smoothed image (in units of arcsecond)
    """

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
    im1 = smoothing(im1, targres)

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
        combo = smoothing(combo, targres)

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

def feather_simple(hires, lores,
                   highresextnum=0,
                   lowresextnum=0,
                   highresscalefactor=1.0,
                   lowresscalefactor=1.0, lowresfwhm=1*u.arcmin,
                   return_hdu=False,
                   return_regridded_lores=False):
    """
    Fourier combine two single-plane images.

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
    return_hdu : bool
        Return an HDU instead of just an image.  It will contain two image
        planes, one for the real and one for the imaginary data.
    return_regridded_lores : bool
        Return the 2nd image regridded into the pixel space of the first?

    Returns
    -------
    combo : image
        The image of the combined low and high resolution data sets
    combo_hdu : fits.PrimaryHDU
        (optional) the image encased in a FITS HDU with the relevant header
    """
    hdu_hi, im_hi, header_hi = file_in(hires)
    hdu_low, im_lowraw, header_low = file_in(lores)

    hdu_low, im_low, nax1, nax2, pixscale = regrid(header_hi, im_hi,
                                                   im_lowraw, header_low)

    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)

    fftsum, combo = fftmerge(kfft, ikfft, im_hi*highresscalefactor,
                             im_low*lowresscalefactor)

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
                 lowresscalefactor=1.0, lowresfwhm=1*u.arcmin
                ):
    """
    Plot the power spectra of two images that would be combined
    along with their weights.

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

    Returns
    -------
    combo : image
        The image of the combined low and high resolution data sets
    combo_hdu : fits.PrimaryHDU
        (optional) the image encased in a FITS HDU with the relevant header
    """
    hdu_hi, im_hi, header_hi = file_in(hires)
    hdu_low, im_lowraw, header_low = file_in(lores)

    print("featherplot")
    pb = ProgressBar(12)

    hdu_low, im_low, nax1, nax2, pixscale = regrid(header_hi, im_hi,
                                                   im_lowraw, header_low)
    pb.update()

    kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale)
    log.debug("bottom-left pixel before shifting: kfft={0}, ikfft={1}".format(kfft[0,0], ikfft[0,0]))
    print("bottom-left pixel before shifting: kfft={0}, ikfft={1}".format(kfft[0,0], ikfft[0,0]))
    pb.update()
    kfft = np.fft.fftshift(kfft)
    pb.update()
    ikfft = np.fft.fftshift(ikfft)
    pb.update()

    fft_hi = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_hi*highresscalefactor)))
    pb.update()
    fft_lo = np.fft.fftshift(np.fft.fft2(np.nan_to_num(im_low*lowresscalefactor)))
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

    import pylab as pl

    pl.clf()
    ax1 = pl.subplot(2,1,1)
    ax1.loglog(rad_as[OK], azavg_kernel[OK], color='b', linewidth=2, alpha=0.8,
               label="Low-res Kernel")
    ax1.loglog(rad_as[OK], azavg_ikernel[OK], color='r', linewidth=2, alpha=0.8,
               label="High-res Kernel")
    ax1.vlines(lowresfwhm.to(u.arcsec).value, 1e-5, 1.1, linestyle='--', color='k')
    ax1.set_ylim(1e-5, 1.1)
    arg_xmin = np.nanargmin(np.abs((azavg_kernel)-1e-5))
    xlim = rad_as[arg_xmin]/1.1, rad_as[1]*1.1
    log.debug("Xlim: {0}".format(xlim))
    assert all(np.isfinite(xlim))
    ax1.set_xlim(*xlim)

    ax1.set_ylabel("Kernel Weight")


    ax2 = pl.subplot(2,1,2)
    ax2.loglog(rad_as[OK], azavg_lo[OK], color='b', linewidth=2, alpha=0.8,
               label="Low-res image")
    ax2.loglog(rad_as[OK], azavg_hi[OK], color='r', linewidth=2, alpha=0.8,
               label="High-res image")
    ax2.set_xlim(*xlim)
    ax2.set_ylim(min([azavg_lo[arg_xmin], azavg_lo[2], azavg_hi[arg_xmin], azavg_hi[2]]),
                 1.1*max([np.nanmax(azavg_lo), np.nanmax(azavg_hi)]),
                )
    ax2.set_ylabel("Power spectrum $|FT|$")

    ax3 = pl.subplot(2,1,2)
    ax3.loglog(rad_as[OK], azavg_lo_scaled[OK], color='b', linewidth=2, alpha=0.5,
               linestyle='--',
               label="Low-res scaled image")
    ax3.loglog(rad_as[OK], azavg_hi_scaled[OK], color='r', linewidth=2, alpha=0.5,
               linestyle='--',
               label="High-res scaled image")
    ax3.set_xlim(*xlim)
    ax3.set_xlabel("Size Scale (arcsec)")
    ax3.set_ylim(min([azavg_lo_scaled[arg_xmin], azavg_lo_scaled[2], azavg_hi_scaled[arg_xmin], azavg_hi_scaled[2]]),
                 1.1*max([np.nanmax(azavg_lo_scaled), np.nanmax(azavg_hi_scaled)]),
                )

    return rad, rad_as, azavg_kernel, azavg_ikernel, azavg_lo, azavg_hi, azavg_lo_scaled, azavg_hi_scaled

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

    if cube_hi.size > 1e8:
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
