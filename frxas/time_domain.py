import numpy as np
from numpy import fft


def gauss_win(signal, frequency, time, window_param):
    """Applies a gaussian windowing function to a time-domain signal

    Parameters
    ----------
    signal : np.ndarray
        Time domain array of signal to be windowed

    frequency : float
        Fundamental frequency of the signal

    time : np.ndarray
        Array of timestamps corresponding to signal samples

    window_param : float
        Defines decay length of window function. Corresponds to number of
        waveforms at the signals frequency.

    Returns
    -------
    win_sig : np.ndarray
        Time domain array of signal with window function applied.
    """
    f = frequency
    t = time
    b = window_param

    window = np.exp((-f**2 * t**2) / b**2)
    win_sig = window * signal

    return win_sig


def freq_bin(freq_in, freq_list, harmonic):
    """Finds the index of the FFT bin for an input frequency.

    Parameters
    ----------
    freq_in: float
        Fundamental frequency of input signal
    freq_list: np.ndarray
        Array of frequencies in FFT ordered from negative to positive, as
        output by numpy.fft.fftshift
    harmonic: integer
        Number of harmonics to return indices for where the fundamental
        frequency is harmonic 1

    Returns
    -------
    bins: np.ndarray
        Array of shape(harmonic, 2) containing indices for the fft bins of the
        desired harmonic frequencies.
    """
    bins = np.ones((harmonic, 2), dtype=int)
    for i in range(0, harmonic):
        try:
            bins[i, 0] = np.isclose(freq_list, -freq_in * (i+1)).nonzero()[0]
            bins[i, 1] = np.isclose(freq_list, freq_in * (i+1)).nonzero()[0]
        except ValueError:
            print("{} harmonic not found in frequency list".format(i+1))
            return bins[:i]

    return bins


def phase_adjust(fft, bins, phase=0):
    """Adjusts components of an fft to represent the desired phase
    Parameters
    ----------
    fft: np.ndarray
        Array containing the FFT of the signal to be phase adjusted
    bins: np.ndarray
        Array of shape(harmonic, 2) containing indices for the fft bins of the
        desired harmonic frequencies.
    phase: float
        Desired phase angle, in degrees, for resulting FFT

    Returns
    -------
    fft_adj: np.ndarray
        Array containing the FFT of a signal adjusted to the desired phase
        angle
    """
    # Pull out indices of negative and positive fundamental frequency
    idx_n = bins[0, 0]
    idx_p = bins[0, 1]

    phs = phase * np.pi / 180

    fft_adj = fft.copy()
    mag = np.abs(fft_adj[idx_p])

    fft_adj[idx_p] = (np.cos(phs) + 1j * np.sin(phs)) * mag
    fft_adj[idx_n] = (np.cos(phs) - 1j * np.sin(phs)) * mag

    return fft_adj


def phase_align(time, reference, signal, freq_in, phase=0, harmonics=1):
    """Adjusts a time-domain reference to a desired phase angle and aligns
    a time-domain signal to the reference while maintaining phase coherence.

    Parameters
    ----------
    time: np.ndarray
        Array of the time sample points corresponding to reference and signal
    reference: np.ndarray
        Array containing values of the reference signal
    signal: np.ndarray
        Array containing values of the time-domain signal to be aligned
    freq_in: float
        Expected frequency to be found in reference and signal
    phase: float
        Desired phase angle, in degrees, for adjusted reference
    harmonics: integer
        Number of harmonics to analyze for phase adjustments
    """
    Ns = np.size(time)

    freqs = fft.fftshift(fft.fftfreq(Ns, time[1]))
    ref_fft = fft.fftshift(fft.fft(reference)/(Ns/2))
    sig_fft = fft.fftshift(fft.fft(signal)/(Ns/2))

    bins = freq_bin(freq_in, freqs, harmonics)
    # Pull out indices of negative and positive fundamental frequency
    idx_p = bins[0, 1]

    phs = phase * np.pi/180

    ref_ang = np.angle(ref_fft[idx_p])
    sig_ang = np.angle(sig_fft[idx_p])

    ang_adj = phs - ref_ang

    print("Before Angle1: ", ref_ang*180/np.pi, " Angle2: ",
     sig_ang*180/np.pi)
    print("Angle adj: ", ang_adj*180/np.pi)

    ref_fft_adj = phase_adjust(ref_fft, bins, phase)
    sig_fft_adj = phase_adjust(sig_fft, bins, (sig_ang + ang_adj)*180/np.pi)

    print("After Angle1: ", np.angle(ref_fft_adj[idx_p], deg=True),
           " Angle2: ", np.angle(sig_fft_adj[idx_p], deg=True))

    ref = fft.ifft(fft.ifftshift(ref_fft_adj)*(Ns/2))
    sig = fft.ifft(fft.ifftshift(sig_fft_adj)*(Ns/2))

    return ref, sig
