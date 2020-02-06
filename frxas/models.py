import numpy as np


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
    fun: callable
        Function to be evaluated following the form fun(x, *args)

    Returns
    -------
    np.ndarray
        Values from evaluating the callable function with the given data set
        parameters and independent variable array/
"""

    args = []
    for pname in params:
        # find all parameters with suffix for current index
        if pname.endswith(str(i+1)):
            args.append(params[pname])

    return fun(x, *args)


def objective_fun(params, x, data, fun):
    """Calculate a residuals array for a given model function

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Contains the Parameters for the model.
    x : list or np.ndarray
        Array or list of arrays of independent variable values.
    data : list or np.ndarray
        Array or list of arrays of data sets to be fit.
    fun: callable
        Function to be evaluated following the form fun(x, *args).

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


def chi_ideal(x, ld, tg, Ao, f):
    """Function for dimensionless vacancy concentrations assuming ideal
    behavior and overpotential control.

    Parameters
    ----------
    x : list or np.ndarray
        Array or list of arrays of length values.
    ld : float
        Characteristic length scale of vacancy profile, often called the
        "utilization length".
    tg : float
        Characteristic time scale of vacancy profile. Reflects time scale of
        switch from kinetic limitations (low frequency) to co-limitation by
        kinetics and diffusion (moderate to high frequency).
    Ao : float
        Thermodynamic factor.
    f : float
        Applied linear frequency in units of Hz.

    Returns
    -------
    np.ndarray
        Evaluated function for given length array and parameters.

    Notes
    -----
    This solution assumes:
        1. Thermodynamic factor is constant: :math:`A \approx A_o`
        2. Experiment is conducted with a controlled overpotential
        3.
  
    Reference
    ---------
    Lu et al....
"""
    return -1 / Ao * np.exp(-x / ld * np.sqrt(1 + 1j * tg * 2 * np.pi * f))
