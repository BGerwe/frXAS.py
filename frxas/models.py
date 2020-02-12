import numpy as np
from scipy.special import lambertw


def dataset_fun(params, i, x, fun):
    r"""Calculate a function's lineshape from parameters for a single data set.

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Contains the Parameters for the model.
    i : int
        Index of the data set being evauluated.
    x : np.ndarray
        Array of independent variable values.
    fun : callable
        Function to be evaluated following the form fun(x, \*args)

    Returns
    -------
    np.ndarray
        Values from evaluating the callable function with the given data set
        parameters and independent variable array.
    """
    args = []
    for pname in params:
        # find all parameters with suffix for current index
        if pname.endswith(str(i+1)):
            args.append(params[pname])

    return fun(x, *args)


def objective_fun(params, x, data, fun):
    r"""Calculate a residuals array for a given model function

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Contains the Parameters for the model.
    x : list or np.ndarray
        Array or list of arrays of independent variable values.
    data : list or np.ndarray
        Array or list of arrays of data sets to be fit.
    fun : callable
        Function to be evaluated following the form fun(x, \*args).

    Returns
    -------
    np.ndarray
        1-D array of residuals for the given model function.
    """
    ndata = np.shape(data)[0]
    resid = 0.0*data[:]

    resid_c = []
    # make residual per data set
    for i in range(ndata):
        resid[i] = data[i] - dataset_fun(params, i, x[i], fun)
        if resid[i].dtype == np.complex:
            # convert to floats as required by minimize()
            resid_c.append(resid[i].ravel().view(np.float))
    # change to array so residuals can be flattened as needed by minimize
    resid_c = np.array(resid_c)
    return np.concatenate(resid_c).ravel()


def calc_Ao(aoo, po2, po2_ref):
    r"""Calculates an adjusted thermodynamic factor `ao`.

    Parameters
    ----------
    aoo : float
        Thermodynamic factor for a reference pO2, `po2_ref`
    po2 : float
        Experimental :math:`pO_2` condition added in units % :math:`O_2`.
    po2_ref : float
        Reference :math:`pO_2` condition added in units % :math:`O_2`.

    Returns
    -------
    ao : float
        Thermodynamic factor adjusted to `po2`.

    Notes
    -----

    This relies on knowing a reference therm. factor, `aoo`, at a reference
    pO2, `po2_ref`, and adjusts it for a given experimental pO2.

    The adjustment is made as:

    .. math::

        A_o = 1 + W( \frac{A_{oo} - 1 * e^{A_{oo} - 1}}\
        {\sqrt{\frac{pO_2}{pO_{2,ref}}}})

    Where W is the lambert W function [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Lambert_W_function

    """
    ao = 1 + lambertw((aoo - 1) * np.exp(aoo - 1) / np.sqrt(po2 / po2_ref))

    # lambertw() returns a complex number, but it should be purely real for any
    # reasonable scenarios
    return ao.real


def chi_ideal(x, ld, tg, Ao, f):
    r"""Summarize the function in one line.

    Function for dimensionless vacancy concentrations assuming ideal behavior
    and overpotential control.

    Parameters
    ----------
    x : list or np.ndarray
        Array or list of arrays of length values.
    ld : float
        :math:`l_{\delta}` : Characteristic length scale of vacancy profile,
        often called the "utilization length".
    tg : float
        :math:`t_G`  : Characteristic time scale of vacancy profile. Reflects
        time scale of switching from kinetic limitations (low frequency) to
        co-limitation by kinetics and diffusion (moderate to high frequency).
    Ao : float
        :math:`A_o` : Thermodynamic factor.
    f : float
        Applied linear frequency in units of Hz.

    Returns
    -------
    np.ndarray
        Evaluated function for given length array and parameters.

    Notes
    -----
    .. math::

        \chi = \frac{-e^{-\frac{x}{l_{\delta}}\sqrt{1 + j 2 \pi f t_G}}}{A_o}

    Based on [1]_ this solution assumes :
        1. Only bulk diffusion is considered. :math:`O_2` reduction occurs at
           electrode/gas interface.
        2. Thermodynamic factor is constant: A :math:`\approx A_o`
        3. Experiment is conducted with a controlled overpotential

    References
    ----------

    .. [1] Y. Lu, C. Kreller, and S.B. Adler,
        Journal of The Electrochemical Society, 156, B513-B525 (2009)
        `doi:10.1149/1.3079337 <https://doi.org/10.1149/1.3079337>`_.
    """
    # After closing class docstring, there should be one blank line to
    # separate following codes (according to PEP257).
    # But for function, method and module, there should be no blank lines
    # after closing the docstring.

    return -1 / Ao * np.exp(-x / ld * np.sqrt(1 + 1j * tg * 2 * np.pi * f))
