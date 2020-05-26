import os

import numpy as np
import glob
import lmfit

from pandas import read_csv
from numpy import fft
from scipy.special import dawsn


def sort_files(all_files):
    return int(all_files.split('\\R')[-1][2:-4])


def extract_data(file_direc, match_str, start=0, end=-1, irr=100,
                 xray_disp=True, xray_raw=False, skip_head=1,
                 sort_func=sort_files):
    """
    """

    all_files = glob.glob(os.path.join(file_direc + match_str))
    all_files.sort(key=sort_func)

    if len(all_files[start:end]) == 0:
        print(f'No files in list with start={start} and end={end}, please',
              ' extend range')
        return

    for i, file in enumerate(all_files[start:end]):
        if i == start:
            data = np.array(read_csv(file, delimiter='\t', header=None,
                                     skiprows=skip_head))
            t = data[:, 0]
        else:
            dum = np.array(read_csv(file, delimiter='\t', header=None,
                                    skiprows=skip_head))
            data = np.append(data, dum, axis=0)

            # Avoid repeating time point where t=0 at the beginning of each
            # file
            t = np.append(t, t[-1] + t[1])
            t = np.append(t, t[-1] + dum[1:, 0])

    # Subtract average to remove DC component from signals. The X-ray signal
    # may still have a DC component from beam intensity drifts during
    # measurement time.
    V = sub_mean(data[:, 4])

    # Use value of potentiostat internal resistor to convert measured current
    # from units of volts to amperes
    J = sub_mean(data[:, 3]) / irr

    Ir = data[:, 2] / data[:, 1]
    Iravg = np.mean(Ir)
    # Divide by average value of X-ray signal to make it a displacement from
    # the mean, which is more physically significant for spatially resolved
    # measurements. Have option of leaving average for incident energy
    # resolved measurements.
    Ir = sub_mean(Ir)
    if xray_disp:
        Ir = Ir / Iravg

    if xray_raw:
        Io = data[:, 1]
        If = data[:, 2]
        return t, V, J, Ir, Io, If
    else:
        return t, V, J, Ir


def freq_bin(freq_in, frequencies, harmonic):
    """Finds the index of the FFT bin for an input frequency.

    Parameters
    ----------
    freq_in: float
        Fundamental frequency of input signal
    frequencies: np.ndarray
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
            bins[i, 0] = np.isclose(frequencies, -freq_in * (i+1)).nonzero()[0]
            bins[i, 1] = np.isclose(frequencies, freq_in * (i+1)).nonzero()[0]
        except ValueError:
            print("{} harmonic not found in frequency list".format(i+1))
            return bins[:i]

    return bins


def get_windowed_fft(time, signal, freq_in, window_param):
    """Applies gaussian window to signal and calculates the complex FFT.

    Parameters
    ----------
    time : np.ndarray
        Array of the time sample points corresponding to reference and signal
    signal : np.ndarray
        Array containing values of the time-domain signal to be aligned
    freq_in : float
        Expected frequency to be found in reference and signal
    window_param : float
        Defines decay length of window function. Corresponds to number of
        waveforms at the signals frequency.
    Returns
    -------
    sig_win_fft: np.ndarray
        Complex FFT of signal with window function applied.

    """
    Ns = np.size(time)
    sig_win = gauss_window(signal, freq_in, time, window_param)
    sig_win_fft = fft.fftshift(fft.fft(sig_win)/(Ns * np.pi))
    return sig_win_fft


def fit_windowed_fft(frequencies, signal_fft, freq_in, window_param,
                     harmonics=1, fit_kws=None, weights=None):
    """Fits the windowed fft.

    Parameters
    ----------
    frequencies: np.ndarray
        Array of frequencies in FFT ordered from negative to positive, as
        output by numpy.fft.fftshift
    signal_fft: np.ndarray
        Complex FFT of signal with window function applied.
    freq_in : float
        Expected frequency to be found in reference and signal.
    window_param : float
        Defines decay length of window function. Corresponds to number of
        waveforms at the signals frequency.
    harmonics : integer
        Number of harmonics to analyze for phase adjustments.
    fit_kws: dict, optional
        Optional parameters to pass to the minimizer.
    Returns
    -------
    model_fit: lmfit.ModelResult
        ModelResult class object with best fit parameters and various goodness-
        of-fit statistics

    """
    # Set up lmfit parameters and model classes
    params = lmfit.Parameters()
    model = None
    # Change range to start from 0 to capture DC behavior, which is nominally
    # subtracted, but may be unavoidable in X-ray data from changing incident
    # beam flux. Voltage and current shouldn't have significant DC components.
    for i in range(1, harmonics+1):
        if model:
            model += lmfit.Model(fft_shape, prefix=f'h{i}_', independent_vars=[
                                 'frequencies', 'freq_in', 'window_param'],
                                 fit_kws=fit_kws)
        else:
            model = lmfit.Model(fft_shape, prefix=f'h{i}_', independent_vars=[
                                'frequencies', 'freq_in', 'window_param'],
                                fit_kws=fit_kws)
        params.add('h%i_harmonic' % i, value=i, vary=False)
        params.add('h%i_re_comp' % i, value=0)
        params.add('h%i_im_comp' % i, value=-1)

    model_fit = model.fit(signal_fft, params=params, frequencies=frequencies,
                          freq_in=freq_in, window_param=window_param,
                          weights=weights)
    return model_fit


def phase_align(time, reference, signal, freq_in, window_param, phase=0,
                harmonics=1, return_params=True, fit_kws=None, weights=None):
    """Adjusts a time-domain reference to a desired phase angle and aligns
    a time-domain signal to the reference while maintaining phase coherence.

    Parameters
    ----------
    time : np.ndarray
        Array of the time sample points corresponding to reference and signal.
    reference : np.ndarray
        Array containing values of the reference data.
    signal : np.ndarray
        Array containing values of the time-domain signal to be aligned.
    freq_in : float
        Expected frequency to be found in reference and signal.
    window_param : float
        Defines decay length of window function. Corresponds to number of
        waveforms at the signals frequency.
    phase : float
        Desired phase angle, in degrees, for adjusted reference.
    harmonics : integer
        Number of harmonics to analyze for phase adjustments.
    fit_kws: dict, optional
        Optional parameters to pass to the minimizer.

    Returns
    -------
    sig : np.ndarray
        Array of the time-domain signal after phase adjustment
    sig_fit: lmfit.Model, optional
        Model class object

    """
    Ns = np.size(time)
    freqs = fft.fftshift(fft.fftfreq(Ns, time[1]))

    # Get FFT of gaussian windowed (apodized) signals
    ref_fft = get_windowed_fft(time, reference, freq_in, window_param)
    sig_fft = get_windowed_fft(time, signal, freq_in, window_param)
    del reference, signal

    if fit_kws is None:
        fit_kws = {}

    # Now fit both models. This can take a long time with large data files.
    ref_fit = fit_windowed_fft(freqs, ref_fft, freq_in, window_param,
                               harmonics=harmonics, fit_kws=fit_kws,
                               weights=weights)
    sig_fit = fit_windowed_fft(freqs, sig_fft, freq_in, window_param,
                               harmonics=harmonics, fit_kws=fit_kws,
                               weights=weights)
    del ref_fft, sig_fft

    # Pull out magnitudes and phase angles of harmonics into a dict so
    # original info isn't overwritten

    ref_comps = {}
    sig_comps = {}

    for i in range(1, harmonics+1):
        ref_comps[f'ang_{i}'] = \
            np.angle(ref_fit.params[f'h{i}_re_comp'].value + 1j *
                     ref_fit.params[f'h{i}_im_comp'].value)
        ref_comps[f'mag_{i}'] = \
            np.abs(ref_fit.params[f'h{i}_re_comp'].value + 1j *
                   ref_fit.params[f'h{i}_im_comp'].value)
        sig_comps[f'ang_{i}'] = \
            np.angle(sig_fit.params[f'h{i}_re_comp'].value + 1j *
                     sig_fit.params[f'h{i}_im_comp'].value)
        sig_comps[f'mag_{i}'] = \
            np.abs(sig_fit.params[f'h{i}_re_comp'].value + 1j *
                   sig_fit.params[f'h{i}_im_comp'].value)

    # Desired phase angle to radians and find adjustment factor
    phs = phase * np.pi/180
    ang_adj = phs - ref_comps['ang_1']

    # Adjust real an imag component of each harmonic by adjustment factor
    # times harmonic index

    for i in range(1, harmonics+1):
        ref_fit.params[f'h{i}_re_comp'].value = \
            np.cos(ref_comps[f'ang_{i}'] + ang_adj * i) * ref_comps[f'mag_{i}']
        ref_fit.params[f'h{i}_im_comp'].value = \
            np.sin(ref_comps[f'ang_{i}'] + ang_adj * i) * ref_comps[f'mag_{i}']
        sig_fit.params[f'h{i}_re_comp'].value = \
            np.cos(sig_comps[f'ang_{i}'] + ang_adj * i) * sig_comps[f'mag_{i}']
        sig_fit.params[f'h{i}_im_comp'].value = \
            np.sin(sig_comps[f'ang_{i}'] + ang_adj * i) * sig_comps[f'mag_{i}']

    sig = fft.ifft(fft.ifftshift(sig_fit.eval()*(Ns*np.pi)))
    # Update data in lmfit.ModelResult so the data and best_fit results don't
    # look wildly different.
    sig_fit.data = get_windowed_fft(time, sig, freq_in, window_param)
    if return_params:
        return sig, sig_fit
    else:
        return sig


def get_freq(direc, point, amplitude, file):
    """
    """
    all_files = glob.glob(os.path.join(direc + point + amplitude + file +
                                       ' [0-9][0-9][0-9].txt'))
    head = np.genfromtxt(all_files[0], delimiter='\t', max_rows=1)
    return head[2]


def sub_mean(data):
    """
    """
    return data - data.mean()


def gauss_window(signal, freq_in, time, window_param):
    """Applies a gaussian windowing function to a time-domain signal.

    Parameters
    ----------
    signal : np.ndarray
        Time domain array of signal to be windowed
    freq_in: float
        Fundamental frequency of input signal
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
    f = freq_in
    t = time
    b = window_param

    window = np.exp((-f**2 * t**2) / b**2)
    win_sig = window * signal

    return win_sig


def gauss_fft(frequencies, freq_in, window_param, harmonic=1):
    """ Calculates the gaussian lineshape for FFT fitting

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies in FFT ordered from negative to positive, as
        output by numpy.fft.fftshift
    freq_in : float
        Fundamental frequency of input signal
    window_param : float
        Defines decay length of window function. Corresponds to number of
        waveforms at the signals frequency.
    harmonic: integer
        Number of harmonics to return indices for where the fundamental
        frequency is harmonic 1
    Returns
    -------
    g_k : np.ndarray
        Array for the gaussian lineshape in an FFT

    """

    w = 2 * np.pi * frequencies
    w_tilde = 2 * np.pi * freq_in
    b = window_param
    k = harmonic
    Ns = frequencies.size
    # Using frequency list to find dt based on Nyquist sampling theorem
    dt = -1 / (2 * frequencies[0])

    x = ((w - k * w_tilde) * np.pi * b) / w_tilde

    g_k = (b * np.sqrt(np.pi) / (dt * Ns * w_tilde)) * np.exp(-x**2)

    return g_k


def dawson_fft(frequencies, freq_in, window_param, harmonic=1):
    """ Calculates the dawson function lineshape for FFT fitting.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies in FFT ordered from negative to positive, as
        output by numpy.fft.fftshift
    freq_in : float
        Fundamental frequency of input signal
    window_param : float
        Defines decay length of window function. Corresponds to number of
        waveforms at the signals frequency.
    harmonic: integer
        Number of harmonics to return indices for where the fundamental
        frequency is harmonic 1.

    Returns
    -------
    d_k : np.ndarray
        Array for the dawson function lineshape in an FFT

    """

    w = 2 * np.pi * frequencies
    w_tilde = 2 * np.pi * freq_in
    b = window_param
    k = harmonic
    Ns = frequencies.size
    # Using frequency list to find dt based on Nyquist sampling theorem
    dt = -1 / (2 * frequencies[0])

    x = ((w - k * w_tilde) * np.pi * b) / w_tilde

    d_k = dawsn(x) * (2 * b / (dt * Ns * w_tilde))

    return d_k


def fft_shape(frequencies, freq_in, window_param, harmonic=1, re_comp=1,
              im_comp=1):
    r""" Calculates the lineshape of a gaussian windowed signal for fitting.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies in FFT ordered from negative to positive, as
        output by numpy.fft.fftshift
    freq_in : float
        Fundamental frequency of input signal
    window_param : float
        Defines decay length of window function. Corresponds to number of
        waveforms at the signals frequency.
    harmonic: integer
        Number of harmonics to return indices for where the fundamental
        frequency is harmonic 1
    re_comp: float
        Coefficient of signals real component
    im_comp: float
        Coefficient of signals imaginary component

    Returns
    -------
    d_k : np.ndarray
        Array for the dawson function lineshape in an FFT

    Notes
    -----
    This FFT fitting procedure follows the process in Wilson, Schwartz and
    Adler[1]
    The fitting lineshape assumes the input signal is of the following form:

    .. math::

        V(t) = \frac{1}{2} \sum_{k=0}^{\infty} \hat{V_k} exp(j k
        \tilde{\omega}t) + \hat{V_{-k}} exp(-j k \tilde{\omega}t)

    Where :math:`\tilde{\omega}` is the radial frequency of the signal, k is
    the harmonic index and the complex Fourier coefficients of each harmonic
    are defined as :math:`\hat{V_{\pm k}} = \hat{V^{'}_k}\pm j \hat{V^{''}_k}`

    A gaussian apodization window is applied as:

    .. math::

        W(t) = exp(-(\frac{\tilde{\omega}t}{2\pi b})^2)

    Where b is a windowing parameter tied to the number of signal cycles.

    The Fourier Transform is defined as:

    .. math::

        f(\omega)=\frac{1}{\pi}\int_0^{\infty}f(t)e^{-j\omega t}

    Using that definition, the fourier transform of the apodized signal is:

    .. math::

        \hat{V}(\omega)=\frac{1}{\pi}\int_0^{\infty}exp(\frac{-\tilde{\omega}^2
        t^2}{4\pi^2b^2})V(t)e^{-j\omega t}

    Solving this and rearranging we get:

    .. math::

        \hat{V}(\omega)=\frac{1}{2} \sum_{k=0}^{\infty} \hat{V^{'}_k} (G_k
        (\omega) + G_{-k}(\omega)) + \hat{V^{''}_k}(D_k(\omega)-D_{-k}(\omega))
        \\ +\frac{j}{2}\sum_{k=0}^{\infty}\hat{V^{''}_k}(G_k(\omega)-
        G_{-k}(\omega))+\hat{V^{'}_k}(-D_k(\omega)-D_{-k}(\omega))

    Where we also define:

    .. math::

        G_k(\omega)=\frac{b\sqrt{\pi}}{\tilde{\omega}}exp(\frac{-(\omega-k
        \tilde{\omega})^2\pi^2b^2}{\tilde{\omega}^2}) \\
        G_{-k}(\omega)=\frac{b\sqrt{\pi}}{\tilde{\omega}}exp(\frac{-(\omega+k
        \tilde{\omega})^2\pi^2b^2}{\tilde{\omega}^2}) \\
        D_{k}(\omega)=\frac{b\sqrt{\pi}}{\tilde{\omega}}exp(\frac{-(\omega-k
        \tilde{\omega})^2\pi^2b^2}{\tilde{\omega}^2})erfi(\frac{(\omega-k
        \tilde{\omega})\pi b}{\tilde{\omega}}) \\
        D_{-k}(\omega)=\frac{b\sqrt{\pi}}{\tilde{\omega}}exp(\frac{-(\omega+k
        \tilde{\omega})^2\pi^2b^2}{\tilde{\omega}^2})erfi(\frac{(\omega+k
        \tilde{\omega})\pi b}{\tilde{\omega}})

    To account for finite measurement time, each shape function is also scaled
    by :math:`\frac{1}{dt Ns}` where dt is the time between data points and Ns
    is the total number of data points.

    For using this fitting shape, data should be FFT'd as:
    ``np.fft.fftshift(np.fft.fft(<data>)/(Ns * np.pi))``

    [1] J.R. Wilson, D.T. Schwartz, and S.B. Adler,
    Electrochimia Acta, 51, 1389-1402 (2006)
    `doi:10.1016/j.electacta.2005.02.109
    <https://doi.org/10.1016/j.electacta.2005.02.109>`_.

    """

    f = frequencies
    f_tilde = freq_in
    b = window_param
    k = harmonic

    vpk = re_comp
    vppk = im_comp

    g_pk = gauss_fft(f, f_tilde, b, k)
    g_nk = gauss_fft(f, f_tilde, b, -k)
    d_pk = dawson_fft(f, f_tilde, b, k)
    d_nk = dawson_fft(f, f_tilde, b, -k)

    fft_shape = 0.5 * (vpk * (g_pk + g_nk) + vppk * (d_pk - d_nk)) + \
        1j / 2 * (vppk * (g_pk - g_nk) + vpk * (-d_pk - d_nk))

    return fft_shape
