import image_registration
from astropy import convolution
import numpy as np
import pylab as pl

# deconvolve using skimage techniques
from skimage.restoration import unsupervised_wiener, richardson_lucy

np.random.seed(0)
im = image_registration.tests.make_extended(imsize=256., powerlaw=1.5)

singledish_kernel = convolution.Gaussian2DKernel(40/2.35, x_size=256, y_size=256)
singledish_kernel_fft = np.fft.fft2(singledish_kernel)

singledish_im = convolution.convolve_fft(im,
                                         convolution.Gaussian2DKernel(40/2.35),
                                         boundary='fill', fill_value=im.mean())

# deconvolve in fourier space
# This "works", but requires that you set a limit on where you perform the
# fourier-space division that is pretty arbitrary / empirical, otherwise you
# just get junk out.
deconv_kernel = singledish_kernel_fft.copy()
badpix = np.abs(deconv_kernel) < 1e-1
imfft = np.fft.fft2(im)
naive_deconvolution_fft = (imfft / deconv_kernel)
naive_deconvolution_fft[badpix] = imfft[badpix]
naive_deconvolution = np.fft.ifft2(naive_deconvolution_fft)

pl.clf()
pl.imshow(np.fft.fftshift(naive_deconvolution.real), cmap='viridis')
pl.colorbar()
pl.title("Deconvolved (Fourier division) pl=1.5 image")
pl.savefig("fourierdivisiondeconvolve_singledish_image_pl1.5.png")


sd_min = singledish_im.min()
sd_range = (singledish_im.max() - sd_min)
singledish_scaled = (singledish_im - sd_min) / sd_range
norm_kernel = singledish_kernel.array/singledish_kernel.array.max()

# note that for skimage, it is critical that clip=False be specified!

# Wiener deconvolution is some sort of strange Bayesian approach.  It works
# great on "real" photos of people, but doesn't work at all on our data
wienered_singledish,chain = unsupervised_wiener(image=singledish_scaled,
                                                psf=norm_kernel,
                                                clip=False,
                                                user_params=dict(max_iter=1000,
                                                                 burnin=100,
                                                                 min_iter=100,
                                                                 threshold=1e-5),)

pl.clf()
pl.imshow(wienered_singledish*sd_range+sd_min, cmap='viridis')
pl.colorbar()
pl.title("Deconvolved Single Dish (Wiener) pl=1.5 image")
pl.savefig("wienerdeconvolve_singledish_image_pl1.5.png")


# Lucy-Richardson/Richardson-Lucy is an iterative, fourier-based approach.  I
# don't understand why, but it is *very* slow.  It looks like it works fine,
# except that it saturates the brightest pixels.  This is probably an issue of
# skimage assuming everything is normalized in the range [0,1]
lucyrichardson_singledish = richardson_lucy(singledish_scaled, norm_kernel,
                                            iterations=10, clip=False)

pl.clf()
pl.imshow(lucyrichardson_singledish*sd_range+sd_min, cmap='viridis')
pl.colorbar()
pl.title("Deconvolved Single Dish (LucyRichardson) pl=1.5 image")
pl.savefig("lucyrichardsondeconvolve_singledish_image_pl1.5.png")
