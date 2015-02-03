import os
import image_registration
from image_registration.fft_tools.zoom import zoom_on_pixel
from FITS_tools.cube_regrid import regrid_fits_cube,regrid_cube_hdu
from FITS_tools.hcongrid import hcongrid,hcongrid_hdu
import FITS_tools
import fft_psd_tools
import spectral_cube.io.fits
from astropy import wcs
from astropy.io import fits
from astropy import coordinates
from astropy import units as u
from astropy import log
from astropy.utils.console import ProgressBar
from itertools import izip
import numpy as np

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
   if isinstance(filename, (fits.ImageHDU, fits.PrimaryHDU) ):
      hdu = filename
   else:
      hdu = fits.open(filename)[extnum]
   
   im     = hdu.data.squeeze()
   header = FITS_tools.strip_headers.flatten_header(hdu.header)

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

    return image, header



def regrid(hd1,im1,im2raw,hd2):
    """
    Regrid the low resolution image to have the same dimension and pixel size with the
    high resolution image.

    Return
    ----------
    hdu2 : An object containing both the image and the header
       This will containt the regridded low resolution image, and the image header taken
       from the high resolution observation.
    im2  : (float point?) array
       The image array which stores the regridded low resolution image.
    nax1, nax2 : int(s)
       Number of pixels in each of the spatial axes.
    pixscale : float (?)
       pixel size in the input high resolution image.

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
    hdu2 = hcongrid_hdu(hdu2, hd1)
    im2 = hdu2.data.squeeze()

    # return variables
    return hdu2, im2, nax1, nax2, pixscale



def pbcorr(fft2, hd1, hd2):
    """
    Divide the fourier transformed low resolution image with its fourier transformed primary
    beam, and then times the fourier transformed primary beam of the high resolution image.

    Parameters
    ----------
    fft2 : float array 
       Fourier transformed low resolution image
    hd1 : header object
       Header of the high resolution image
    hd2 : header object
       Header of the low resolution image  
 
    Return
    ---------- 
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
    Construct the weight kernels (image arrays) for the fourier transformed low resolution and
    high resolution images. This routine follow the "feather algorithm", e.g.:
  
    (***To be detailed.)  
  
    Return
    ----------
    kernel  : float array
       An image array containing the weighting for the low resolution image
    kernel1 : float array
       An image array containing the weighting for the high resolution image

    Parameters
    ----------
    nax2, nax1 : int
       Number of pixels in each axes. 
    lowresfwhm : float
       Angular resolution of the low resolution image (FWHM)  
    pixscale : float (?)
       pixel size in the input high resolution image.
    """
    # Construct arrays which hold the x and y coordinates (in unit of pixels)
    # of the image
    ygrid,xgrid = (np.indices([nax2,nax1])-np.array([(nax2-1.)/2,(nax1-1.)/2.])[:,None,None])

    fwhm = np.sqrt(8*np.log(2))
    # sigma in pixels
    sigma = ((lowresfwhm/fwhm/(pixscale*u.deg)).decompose().value)
    log.debug('sigma = {0}'.format(sigma))

    kernel = np.fft.fftshift( np.exp(-(xgrid**2+ygrid**2)/(2*sigma**2)) )
    kernel/=kernel.max()
    kernel1 = 1 - kernel

    return kernel,kernel1



def fftmerge(kernel1,kernel2,fft1,fft2):
    """
    Combine images in the fourier domain, and then output the combined image
    both in fourier domain and the image domain.

    Parameters
    ----------
    kernel1,2 : float array
       Weighting images.
    fft1,fft2: float array
       Fourier transformed input images.

    Return
    ----------
    fftsum : float array
       Combined image in fourier domain.
    combo  : float array
       Combined image in image domain.
    """

    # Sanity check in case that the two input images does not overlap well
    # in the uv-distance range.

    # Combine and inverse fourier transform the images
    fftsum = kernel2*fft2 + kernel1*fft1
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
    interpol        = im1
    interpol_header = hd1
    interpol_hdu    = fits.PrimaryHDU(data=np.abs(im1), header=hd1)

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
                targres = -1.0,
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
    hdu1, im1,    hd1    = file_in(hires, highresextnum)
    hdu2, im2raw, hd2    = file_in(lores, lowresextnum)

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

os.system("rm -rf output.fits")
os.system("rm -rf interpolate.fits")
interpol_hdu = AKB_interpol("Dragon.im350.crop.fits", "Dragon.im350.crop.fits", "faint_final.shift.fix.fits")
f = AKB_combine("faint_final.shift.fix.fits",interpol_hdu, lowresscalefactor=0.0015,return_hdu=True)
