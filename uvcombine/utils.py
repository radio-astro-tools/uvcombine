
import numpy as np
from astropy.utils import NumpyRNGContext


def make_extended(imsize, powerlaw=2.0,
                  return_fft=False, full_fft=True, randomseed=32788324):
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

    with NumpyRNGContext(randomseed):

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
