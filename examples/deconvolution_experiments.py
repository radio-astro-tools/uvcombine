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

sd_min = singledish_im.min()
sd_range = (singledish_im.max() - sd_min)
singledish_scaled = (singledish_im - sd_min) / sd_range
norm_kernel = singledish_kernel.array/singledish_kernel.array.max()
wienered_singledish,chain = unsupervised_wiener(image=singledish_scaled,
                                                psf=norm_kernel,
                                                user_params=dict(max_iter=10000,
                                                                 burnin=100,
                                                                 min_iter=100,
                                                                 threshold=1e-6),)

pl.clf()
pl.imshow(wienered_singledish*sd_range+sd_min, cmap='viridis')
pl.colorbar()
pl.title("Deconvolved Single Dish (Wiener) pl=1.5 image")
pl.savefig("wienerdeconvolve_singledish_image_pl1.5.png")


lucyrichardson_singledish = richardson_lucy(singledish_scaled, norm_kernel, iterations=50)

pl.clf()
pl.imshow(lucyrichardson_singledish*sd_range+sd_min, cmap='viridis')
pl.colorbar()
pl.title("Deconvolved Single Dish (LucyRichardson) pl=1.5 image")
pl.savefig("lucyrichardsondeconvolve_singledish_image_pl1.5.png")
