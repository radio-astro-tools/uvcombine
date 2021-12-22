import numpy as np
from .uvcombine import fftmerge, feather_kernel

def compare_parameters_feather_simple(im, im_hi, im_low, lowresfwhm, pixscale,
                                      suffix="", replacement_threshold=0.5,
                                      psd_axlims=(1e-3,1,10,5e3),
                                     ):
    """
    Create diagnostic plots for different simulated feathers
    """

    from turbustat.statistics import psds
    import pylab as pl

    feathers = {}
    fig1 = pl.figure(1, figsize=(16,16))
    fig2 = pl.figure(2, figsize=(16,16))
    fig3 = pl.figure(3, figsize=(16,16))
    fig1.clf()
    fig2.clf()
    fig3.clf()


    plotnum = 1
    for replace_hires,ls in ((replacement_threshold, '--'),(False,':')):
        for lowpassfilterSD,lw in ((True,2),(False,1)):
            for deconvSD,color in ((True,'r'), (False, 'k')):
                #im_hi = im_interferometered
                #im_low = singledish_im
                lowresscalefactor=1
                highresscalefactor=1

                nax1,nax2 = im.shape
                kfft, ikfft = feather_kernel(nax2, nax1, lowresfwhm, pixscale,)

                fftsum, combo = fftmerge(kfft*1,
                                         ikfft*1,
                                         im_hi*highresscalefactor,
                                         im_low*lowresscalefactor,
                                         replace_hires=replace_hires,
                                         lowpassfilterSD=lowpassfilterSD,
                                         deconvSD=deconvSD,
                                        )
                combo = combo.real
                feathers[replace_hires, lowpassfilterSD, deconvSD] = combo
                resid = im-combo


                pfreq, ppow = psds.pspec(np.fft.fftshift(np.abs(fftsum)))
                name = (("Replace < {}; ".format(replace_hires) if replace_hires else "") +
                        ("filterSD;" if lowpassfilterSD else "")+
                        ("deconvSD" if deconvSD else ""))
                if name == "":
                    name = "CASA defaults"
                pfreq = pfreq[np.isfinite(ppow)]
                ppow = ppow[np.isfinite(ppow)]

                pfreq_resid, ppow_resid = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(resid))))
                pfreq_resid = pfreq_resid[np.isfinite(ppow_resid)]
                ppow_resid = ppow_resid[np.isfinite(ppow_resid)]

                ax1 = fig1.add_subplot(3, 3, plotnum)
                ax1.loglog(pfreq, ppow, label=name, linestyle=ls, linewidth=lw, color=color, alpha=0.75)
                ax1.loglog(pfreq_resid, ppow_resid, linestyle=ls, linewidth=lw, color='b', alpha=0.75)
                ax1.axis(psd_axlims)
                ax1.set_title(name)

                ax2 = fig2.add_subplot(3, 3, plotnum)
                ax2.imshow(combo, interpolation='none', origin='lower')
                ax2.set_title(name)
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])

                ax3 = fig3.add_subplot(3, 3, plotnum)
                ax3.imshow(resid, interpolation='none', origin='lower')
                ax3.set_title(name)
                ax3.set_xticklabels([])
                ax3.set_yticklabels([])


                plotnum += 1


    ax1 = fig1.add_subplot(3, 3, plotnum)
    pfreq, ppow = psds.pspec(np.fft.fftshift(np.abs(np.fft.fft2(im))))
    pfreq = pfreq[np.isfinite(ppow)]
    ppow = ppow[np.isfinite(ppow)]
    ax1.loglog(pfreq, ppow, linestyle='-', linewidth=4, color='g', alpha=1)
    ax1.axis(psd_axlims)
    ax1.set_title("Original Image")

    ax2 = fig2.add_subplot(3, 3, plotnum)
    ax2.imshow(im, interpolation='none', origin='lower')
    ax2.set_title("Original Image")
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    ax3 = fig3.add_subplot(3, 3, plotnum)
    ax3.imshow(im, interpolation='none', origin='lower')
    ax3.set_title("Original Image")
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    fig1.subplots_adjust(hspace=0.1, wspace=0.1)
    fig2.subplots_adjust(hspace=0.01, wspace=0.01)
    fig3.subplots_adjust(hspace=0.01, wspace=0.01)
    for ii in range(5):
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()


    fig1.savefig("parameter_comparison_powerspectra{}.png".format(suffix), bbox_inches='tight')
    fig2.savefig("parameter_comparison_images{}.png".format(suffix), bbox_inches='tight')
    fig3.savefig("parameter_comparison_residuals{}.png".format(suffix), bbox_inches='tight')
    #fig1.legend(loc='best')

    #ax1.set_ylim(1e1,1e5)
