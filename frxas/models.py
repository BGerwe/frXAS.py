import re
import numpy as np
from scipy.special import lambertw
from lmfit import Parameters, fit_report, minimizer


def dataset_fun(params, i, x, fun):
    """Calculate a model function profile from fit parameters for one data set.

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Contains the Parameters for the model.
    i : int
        Index of the data set being evauluated.
    x : np.ndarray
        Array of independent variable values.
    fun : callable
        Function to be evaluated following the form fun(x, \\*args)

    Returns
    -------
    np.ndarray
        Values from evaluating the callable function with the given data set
        parameters and independent variable array.

    """
    args = {}
    for pname in params:
        # find all parameters with suffix for current index
        if pname.split("_")[-1] == str(i + 1):
            args[pname.split("_")[0]] = params[pname]

    return fun(x, **args)


def calc_resid(data, model):
    """Calculates residuals of fit to one data set.

    Parameters
    ----------
    data : np.ndarray
        Values of FR-XAS profile data.
    model : np.ndarray
        Values of fit profile from model function.

    Returns
    -------
    resid_c : np.ndarry
        Residuals.

    """
    resid_c = []
    resid = data - model

    if resid.dtype == complex:
        resid_c = resid.ravel().view(float)
    else:
        resid_c = resid
    resid = None
    data = None
    return resid_c


def objective_fun(params, x, data, fun):
    """Calculate a residuals array of all data sets fit to model function.

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Contains the Parameters for the model.
    x : list or np.ndarray
        Array or list of arrays of independent variable values.
    data : list or np.ndarray
        Array or list of arrays of data sets to be fit.
    fun : callable
        Function to be evaluated following the form fun(x, \\*args).

    Returns
    -------
    np.ndarray
        1-D array of residuals for the given model function.

    """
    resid = []

    if np.shape(data[0]) == ():
        # If only one data set passed in
        resid.append(calc_resid(data, dataset_fun(params, 0, x, fun)))
        resid = np.array(resid)

    else:
        # If multiple data sets passed
        ndata = np.shape(data)[0]

        # make residual per data set
        for i in range(ndata):
            resid.append(calc_resid(data[i], dataset_fun(params, i, x[i], fun)))

        # change to array so residuals can be flattened as needed by minimize
        resid = np.array(resid, dtype=object)
    del data

    return np.concatenate(resid)


def calc_ao(aoo, po2, po2_ref):
    """Calculates an adjusted thermodynamic factor `ao`.

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

        A_o = 1 + W( \\frac{A_{oo} - 1 * e^{A_{oo} - 1}}\
        {\\sqrt{\\frac{pO_2}{pO_{2,ref}}}})

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
    """Model of chi profile w/ overpotential control and ideal thermo (Ao = 1).

    Parameters
    ----------
    x : list or np.ndarray
        Array or list of arrays of length values.
    ld : float
        :math:`l_{\\delta}` : Characteristic length scale of vacancy profile,
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

        \\chi = \\frac{-e^{-\\frac{x}{l_{\\delta}} |br|
        \\sqrt{1 + j 2 \\pi f t_G}}}{A_o}

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
    return -1 / ao * np.exp(-x / ld * np.sqrt(1 + 1j * tg * 2 * np.pi * f))


def chi_amp(x, amp, ld, tg, f):
    """Ideal chi profile with adjustable amplitude.

    Parameters
    ----------
    x : list or np.ndarray
        Array or list of arrays of length values.
    ld : float
        :math:`l_{\\delta}` : Characteristic length scale of vacancy profile,
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

        \\chi = \\frac{-e^{-\\frac{x}{l_{\\delta}} |br|
        \\sqrt{1 + j 2 \\pi f t_G}}}{A_o}

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
    return amp * np.exp(-x / ld * np.sqrt(1 + 1j * tg * 2 * np.pi * f))


def chi_patterned(x, amp=1, gammap=1e-3, ld=15, tg=1, f=1, L=0.6):
    """Model of chi profile with voltage control and nonideal thermo.

    Parameters
    ----------
    x : list or np.ndarray
        Array or list of arrays of length values.
    amp : float
        Arbitrary scaling factor relating absorbance to concentration
        displacements
    gammap : float
        Ratio of electrolyte to electrode resistances
    ld : float
        :math:`l_{\\delta}` : Characteristic length scale of vacancy profile,
        often called the "utilization length".
    tg : float
        :math:`t_G`  : Characteristic time scale of vacancy profile. Reflects
        time scale of switching from kinetic limitations (low frequency) to
        co-limitation by kinetics and diffusion (moderate to high frequency).
    f : float
        Applied linear frequency in units of Hz.
    L : float
        Film thickness in microns.

    Returns
    -------
    np.ndarray
        Evaluated function for given length array and parameters.

    """
    g_p = gammap
    w = 2 * np.pi * f
    chi = -amp / (1 + g_p * np.sqrt(1 + 1j * tg * w)) * np.exp(-x / ld * np.sqrt(1 + 1j * tg * w))
    # Note gamma_p = gamma * L / ld

    return chi


def save_fit_report(filename, fit, start_inds=None):
    """Function to save lmfit minimize results from `lmfit.fit_report`.

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
    params: lmfit.MinimizerResult
        Object containing parameter information and fit statsitics from report.
        Information not included in the report is not carried over from the
        original MinimizerResult object.

    """
    f = open(filename, mode="r")
    lines = f.readlines()
    f.close()

    # Start with empty MinimizerResult class and build up based on
    # saved report txt
    mini = minimizer.MinimizerResult()
    mini.ndata = 1
    mini.nfree = 1
    mini.errorbars = False
    mini.params = Parameters()

    # Pull out fit statistics values and find positions of "Variables" and
    # "Correlations" sections
    for i, line in enumerate(lines):
        if "# fitting method" in line:
            mini.method = line.split("=")[-1][1:].split("\n")[0]
        if "# function evals" in line:
            mini.nfev = int(line.split("=")[-1][1:])
        if "# data points" in line:
            mini.ndata = int(line.split("=")[-1][1:])
        if "# variables" in line:
            mini.nvarys = int(line.split("=")[-1][1:])
        if "chi-square" in line and "chisqr" not in dir(mini):
            mini.chisqr = float(line.split("=")[-1][1:])
        if "reduced chi-square" in line:
            mini.redchi = float(line.split("=")[-1][1:])
        if "Akaike info" in line:
            mini.aic = float(line.split("=")[-1][1:])
        if "Bayesian info" in line:
            mini.bic = float(line.split("=")[-1][1:])

        if line.startswith("[[Variables]]"):
            start_varys = i + 1
        elif line.startswith("[[Correlations]]"):
            end_varys = i
            start_correls = i + 1
        elif "end_varys" in locals():
            end_correls = i
        else:
            end_varys = i

    if mini.method in ("leastsq", "least_squares"):
        mini.errorbars = True

    varys = lines[start_varys:end_varys]
    if "start_correls" in locals():
        correls = lines[start_correls : end_correls + 1]

    # Walk through "Variables" section text and reconstruct Parameters
    for line in varys:
        name_str = re.search(r"[a-zA-Z]+_[0-9]+", line)
        val_str = re.search(r" -?\d+(\.\d+)?", line)

        if name_str and val_str:
            name = name_str.group()
            val = float(val_str.group())
            mini.params.add(name, value=val)
        else:
            continue

        if "(fixed)" in line:
            mini.params[name].vary = False
            mini.params[name].stderr = 0
        else:
            mini.params[name].vary = True

            bound_str = re.search(r"\+/- (\d+(\.\d+)?(e-?\d+)?)", line)
            expr_str = expr_str = re.search(r"== .*", line)
            init_str = re.search(r"init = -?\d+(\.\d+)?", line)

            if bound_str:
                stderr = float(bound_str.group()[4:])
                mini.params[name].stderr = stderr
            if init_str:
                init = float(init_str.group()[6:])
                mini.params[name].init_value = init
            elif expr_str:
                expr = expr_str.group()[4:-1]
                mini.params[name].expr = expr

    # Walk through "Correlations" text and extract that information.
    # Unfortunately, they are output with 3 sig figs, which creates ordering
    # issues in the report since they are sorted by correlation magnitude.
    for line in correls:
        vary1_str = re.search(r"C[(][a-zA-Z]+_[0-9]+", line).group()[2:]
        vary2_str = re.search(r"[a-zA-Z]+_[0-9]+[)]", line).group()[:-1]
        correl = float(line.split("=")[-1][1:])

        if mini.params[vary1_str].correl:
            mini.params[vary1_str].correl[vary2_str] = correl
        else:
            mini.params[vary1_str].correl = {vary2_str: correl}

    return mini
