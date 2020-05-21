import re
import numpy as np
from scipy.special import lambertw
from lmfit import Parameters, fit_report


def dataset_fun(params, i, x, fun):
    """Calculate a function's lineshape from parameters for a single data set.

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
        if pname.split('_')[-1] == str(i+1):
            args.append(params[pname])

    # print(args)
    return fun(x, *args)


def calc_resid(data, model):
    resid_c = []
    resid = data - model

    if resid.dtype == np.complex:
        resid_c = resid.ravel().view(np.float)
    else:
        resid_c = resid
    resid = None
    data = None
    return resid_c


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
    resid = []

    if np.shape(data[0]) == ():
        # If only one data set passed in
        resid.append(calc_resid(data, dataset_fun(params, 0, x, fun)))
    else:
        # If multiple data sets passed
        ndata = np.shape(data)[0]

        # make residual per data set
        for i in range(ndata):

            if data[i].dtype == np.complex:
                # convert to floats as required by minimize()
                resid.append(calc_resid(data[i],
                             dataset_fun(params, i, x[i], fun)))
            else:
                resid.append(calc_resid(data[i],
                             dataset_fun(params, i, x[i], fun)))
    # change to array so residuals can be flattened as needed by minimize
    data = None
    ndata = None
    resid = np.array(resid)
    return np.concatenate(resid).ravel()


def calc_ao(aoo, po2, po2_ref):
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


def chi_ideal(x, ao, ld, tg, f):
    """Function for dimensionless vacancy concentrations assuming ideal
    behavior and overpotential control.

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
    ao : float
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

    return -1 / ao * np.exp(-x / ld * np.sqrt(1 + 1j * tg * 2 * np.pi * f))


def chi_amp(x, amp, ld, tg, f):
    """Function for dimensionless vacancy concentrations assuming ideal
    behavior and overpotential control.

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
    amp : float
        Physically non-specific fitting parameter to capture amplitude
        variation.
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

    return amp * np.exp(-x / ld * np.sqrt(1 + 1j * tg * 2 * np.pi * f))


def save_fit_report(filename, fit, start_inds=None):
    """Function to save lmfit minimize results from `fit_report`.

    Parameters
    ----------
    filename: str
        Directory and file name to save fit report
    fit: lmfit.minimizer.MinimizerResult
        Output from lmfit minimizer
    start_inds: array like
        Indices of starting positions for each fr-XAS profile.

    """
    f = open(filename, "w+")
    report = fit_report(fit)
    f.write(f"Starting indices: {start_inds} \n")
    f.write(report)
    f.close()


def load_fit_report(filename):
    """Extracts information from saved fit report into a `Parameters` object.

    Parameters
    ----------
    filename: str
        Directory and file name of saved fit report
    Returns
    -------
    params: lmfit.parameter.Parameters
        Object containing all parameter information from saved fit. Min and Max
        values are 20% lower and higher, respectively, than the estimated
        standard error from the fit, if available.

    """
    f = open(filename, mode='r')
    lines = f.readlines()
    f.close()

    for i, line in enumerate(lines):
        if line.startswith("[[Variables]]"):
            start_line = i + 1

        elif line.startswith("[[Correlations]]"):
            end_line = i
        else:
            end_line = i
    raw = lines[start_line:end_line]

    # Pulling out key info from each line
    params = Parameters()
    # lmfit gets mad when you try to add an expression for a variable that
    # that doesn't exist yet, so we have to do it at the end by saving all
    # parameter names and expr variables
    names, exprs = [], []
    for line in raw:
        name_str = re.search(r'[a-zA-Z]+_[0-9]+', line)
        val_str = re.search(r' (\d+(\.\d+)?)', line)
        bound_str = re.search(r'\+/- (\d+(\.\d+)?)', line)
        fix_str = re.search(r'fixed', line)
        expr_str = re.search(r'== .*', line)

        if name_str:
            name = name_str.group()
            names.append(name)
        if val_str:
            val = float(val_str.group())
        if bound_str and val:
            lb = val - float(bound_str.group()[4:]) * 1.2
            ub = val + float(bound_str.group()[4:]) * 1.2
        else:
            lb = -np.inf
            ub = np.inf
        vary = True
        if fix_str:
            vary = False
        expr = None
        if expr_str:
            expr = expr_str.group()[4:-1]
        exprs.append(expr)

        params.add(name, value=val, min=lb, max=ub, vary=vary)

    # Now we set expressions
    for name, expr in zip(names, exprs):
        params[name].expr = expr
    return params
