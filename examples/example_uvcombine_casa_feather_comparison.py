
'''
Example script for generating a power-law imaging, mocking a
single dish and interferometer response, and comparing feathered
images created by uvcombine.feather_simple and the feather task
in CASA.

This example originally tested using the pip-installed CASA v6.4.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from uvcombine.utils import (generate_test_fits,
                             interferometrically_observe_image,
                             singledish_observe_image)
from uvcombine import feather_simple

from spectral_cube import SpectralCube, Projection
from radio_beam import Beam

from turbustat.statistics import PowerSpectrum

# Consistency with CASA
import astropy.units as units
from pathlib import Path

tmp_path = Path(".")

pixel_scale = 1 * units.arcsec
restfreq = 100 * units.GHz

highres_major = 2 * units.arcsec

# Generate input image
input_hdu = generate_test_fits(imsize=512, powerlaw=1.5,
                               beamfwhm=highres_major,
                               pixel_scale=pixel_scale,
                               restfreq=restfreq,
                               brightness_unit=units.Jy / units.sr)

input_fn = tmp_path / "input_image_sz512as_pl1.5_fwhm2as_scale1as.fits"
input_hdu.writeto(input_fn, overwrite=True)

input_proj = Projection.from_hdu(input_hdu).to(units.Jy / units.beam)

# Make Interferometric image
intf_data = interferometrically_observe_image(image=input_hdu.data,
                                              pixel_scale=pixel_scale,
                                              largest_angular_scale=40*units.arcsec,
                                              smallest_angular_scale=highres_major)[0].real
intf_hdu = fits.PrimaryHDU(data=intf_data.value if hasattr(intf_data, "value") else intf_data,
                            header=input_hdu.header)
intf_proj = Projection.from_hdu(intf_hdu).to(units.Jy / units.beam)
intf_fn = tmp_path / "input_image_sz512as_pl1.5_fwhm2as_scale1as_intf2to40as.fits"
intf_proj.write(intf_fn, overwrite=True)

# Make SD image
sd_header = input_hdu.header.copy()

major = 15*units.arcsec
# Eff SD diam (to compare with CASA in troubleshooting)
D_sd = 1.22 * restfreq.to(units.m, units.spectral()) / major.to(units.rad).value

sd_beam = Beam(major=major)
sd_header.update(sd_beam.to_header_keywords())

sd_fn = tmp_path / "input_image_sz512as_pl1.5_fwhm2as_scale1as_sd15as.fits"
sd_data = singledish_observe_image(input_hdu.data,
                                   pixel_scale=pixel_scale,
                                   beam=sd_beam,
                                   boundary='wrap')

sd_hdu = fits.PrimaryHDU(data=sd_data.value if hasattr(sd_data, "value") else sd_data,
                        header=sd_header)
sd_hdu.header.update(sd_beam.to_header_keywords())
sd_proj = Projection.from_hdu(sd_hdu).to(units.Jy / units.beam)
sd_proj.write(sd_fn, overwrite=True)


# Feathering with CASA
from casatasks import importfits, feather

intf_fn_image = intf_fn.parent / intf_fn.name.replace(".fits",".image")
sd_fn_image = sd_fn.parent / sd_fn.name.replace(".fits",".image")

# CASA needs a posix string to work
importfits(fitsimage=intf_fn.as_posix(), imagename=intf_fn_image.as_posix(),
           defaultaxes=True, defaultaxesvalues=['', '', '100GHz', 'I'],
           overwrite=True)
importfits(fitsimage=sd_fn.as_posix(), imagename=sd_fn_image.as_posix(),
           defaultaxes=True, defaultaxesvalues=['', '', '100GHz', 'I'],
           overwrite=True)

output_name = tmp_path / 'casafeathered.image'

os.system(f"rm -r {output_name}")

sdfactor = 1.0
lowpassfilterSD = False

# The CASA feather is currently off by a ratio VERY close to the beam area
# ratio. I don't really understand why...
# beamarea_factor = (major / highres_major).decompose().value**2
beamarea_factor = (sd_hdu.header['BMAJ'] / intf_hdu.header['BMIN'])**2

feather(imagename=output_name.as_posix(),
        highres=intf_fn_image.as_posix(),
        lowres=sd_fn_image.as_posix(),
        sdfactor=sdfactor,
        lowpassfiltersd=lowpassfilterSD,
        )

# Right now read as spectralcube despite being a 2D image.
casa_feather_proj = SpectralCube.read(output_name)[0]
# casa_sd_proj = SpectralCube.read(sd_fn_image)[0]

# exportfits(imagename=output, fitsimage=output+".fits",
#            overwrite=True, dropdeg=True)

# Feathering with uvcombine
feathered_hdu = feather_simple(hires=intf_proj, lores=sd_proj,
                                lowresscalefactor=sdfactor,
                                lowpassfilterSD=lowpassfilterSD,
                                deconvSD=False,
                                return_hdu=True)

uvcomb_feather_proj = Projection.from_hdu(feathered_hdu)

diff = (uvcomb_feather_proj - casa_feather_proj).value

diff_orig_uvc = (input_proj.value - uvcomb_feather_proj.value)
diff_orig_casa = (input_proj.value - casa_feather_proj.value)

diffsq = (diff**2) / (casa_feather_proj.value**2)

diff_frac_orig = diff / input_proj.value

print("Proof that we have exactly reimplemented CASA's feather: ")
print("((casa-uvcombine)**2 / casa**2).sum() = {0}"
        .format(diffsq.sum()))
print("Maximum of abs(diff): {0}".format(np.abs(diff).max()))

# There can be individual outlier pixels.
print("Median of diff^2/casa^2: {0}".format(np.median(diffsq)))

plt.subplot(221)
plt.imshow(input_proj.value, origin='lower')
plt.colorbar()
plt.title("Original")

plt.subplot(222)
plt.imshow(casa_feather_proj.value, origin='lower')
plt.colorbar()
plt.title("CASA feather")

plt.subplot(223)
plt.imshow(uvcomb_feather_proj.value, origin='lower')
plt.colorbar()
plt.title("uvcombine feather")

plt.subplot(224)
plt.imshow(diff_frac_orig, origin='lower', vmax=-0.1, vmin=0.1)
plt.colorbar()
plt.title("(uvcombine - CASA) / Original")

plt.tight_layout()
plt.savefig("plaw1p5_feather_comparison.png")
plt.savefig("plaw1p5_feather_comparison.pdf")

plt.figure()

ax1 = plt.subplot(321)
_ = plt.hist(input_proj.value[np.isfinite(input_proj.value)])
plt.title("Original")

plt.subplot(322, sharex=ax1)
_ = plt.hist(casa_feather_proj.value[np.isfinite(casa_feather_proj)])
plt.title("CASA feather")

plt.subplot(323, sharex=ax1)
_ = plt.hist(uvcomb_feather_proj.value[np.isfinite(uvcomb_feather_proj)])
plt.title("uvcombine feather")

plt.subplot(324, sharex=ax1)
_ = plt.hist(diff[np.isfinite(diff)])
plt.title("Diff in feathers")

plt.subplot(325, sharex=ax1)
_ = plt.hist(diff_orig_uvc[np.isfinite(diff_orig_uvc)])
plt.title("Orig - uvcombine feather")

plt.subplot(326, sharex=ax1)
_ = plt.hist(diff_orig_casa[np.isfinite(diff_orig_casa)])
plt.title("Orig - CASA feather")


plt.figure()

orig_pspec = PowerSpectrum(input_proj).run(verbose=False, fit_2D=False)
casafeather_pspec = PowerSpectrum(casa_feather_proj).run(verbose=False, fit_2D=False)
uvcombfeather_pspec = PowerSpectrum(uvcomb_feather_proj).run(verbose=False, fit_2D=False)

sd_pspec = PowerSpectrum(sd_proj).run(verbose=False, fit_2D=False)
sd_pspec_beamcorr = PowerSpectrum(sd_proj).run(verbose=False, fit_2D=False, beam_correct=True)
intf_pspec = PowerSpectrum(intf_proj).run(verbose=False, fit_2D=False)

plt.loglog(orig_pspec.freqs.value, orig_pspec.ps1D, label='Original')
plt.loglog(casafeather_pspec.freqs.value, casafeather_pspec.ps1D, label='CASA feath')
plt.loglog(uvcombfeather_pspec.freqs.value, uvcombfeather_pspec.ps1D, label='uvcomb feath')

plt.loglog(sd_pspec.freqs.value, sd_pspec.ps1D, ":", label='SD')
plt.loglog(sd_pspec_beamcorr.freqs.value, sd_pspec_beamcorr.ps1D, ":",
           label='SD Beam corr', linewidth=2)
plt.loglog(intf_pspec.freqs.value, intf_pspec.ps1D, "--", label='Intf')

plt.legend()
