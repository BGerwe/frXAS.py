import numpy as np
import matplotlib.pyplot as plt

from . import models


def plot_chi(axes, x, data, params=None, add_fit=False,
             model=None, x_units=r'$\mum$', **kwargs):
    """
    Plotting real and imaginary components of FR-XAS profiles on separate axes.

    Parameters
    ----------
    axes : list
        List of matplotlib.axes.Axes on which to plot chi profile
    x : array_like
        Array of distance points or list of arrays for multiple data sets.
    data : array_like
        Array of complex-type chi points or list of arrays for multiple data
        sets

    Returns
    -------
    axes : matplotlib.axes.Axes
        Plot axes.

    """
    if 'marker' in kwargs.keys():
        marker = kwargs.pop('marker')
    else:
        marker = 'x'

    if 'ls' in kwargs.keys():
        ls = kwargs.pop('ls')
    else:
        ls = '-'

    if 'fontsize' in kwargs.keys():
        fontsize = kwargs.pop('fontsize')
    else:
        fontsize = 18

    try:
        ax_re = axes[0]
        ax_im = axes[1]
    except TypeError:
        print('Incorrect number of matplotlib.axes passed.')
        raise

    if (np.shape(data[0]), np.shape(x[0])) == ((), ()):
        ax_re.plot(x, data.real, marker=marker, ls=ls, **kwargs)
        ax_im.plot(x, data.imag, marker=marker, ls=ls, **kwargs)

        if add_fit:
            fit = models.dataset_fun(params, 0, x, model)
            ax_re.plot(x, fit.real, ls=ls, **kwargs)
            ax_im.plot(x, fit.imag, ls=ls, **kwargs)

    elif np.shape(data[0]) != np.shape(x[0]):
        raise TypeError("The shapes of `data` and `x` are inconsistent")
    else:
        for i, (xa, dat) in enumerate(zip(x, data)):
            ax_re.plot(xa, dat.real, marker=marker, ls=ls, **kwargs)
            ax_im.plot(xa, dat.imag, marker=marker, ls=ls, **kwargs)

            if add_fit:
                fit = models.dataset_fun(params, i, xa, model)
                ax_re.plot(xa, fit.real, marker='', ls=ls, **kwargs)
                ax_im.plot(xa, fit.imag, marker='', ls=ls, **kwargs)

    ax_re.set_ylabel(r'Re[$\chi$]', fontsize=fontsize)
    ax_im.set_ylabel(r'Im[$\chi$]', fontsize=fontsize)

    ax_re.xaxis.set_major_formatter(plt.NullFormatter())
    ax_im.set_xlabel(r'Distance ($\mu m$)', fontsize=fontsize)
    for ax in axes:

        # Make the tick labels larger
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Change the number of labels on each axis to five
        ax.locator_params(axis='y', nbins=6, tight=True)

        # Add a light grid
        ax.grid(b=True, which='major', axis='both', alpha=.5)

    y_offset = ax_re.yaxis.get_offset_text()
    y_offset.set_size(18)
    t = ax_re.xaxis.get_offset_text()
    t.set_size(18)

    return ax_re, ax_im


def plot_chi_mag(ax, x, data, params=None, add_fit=False,
                 model=None, x_units=r'$\mum$', **kwargs):
    """
    Plotting magnitude of FR-XAS profiles.

    Parameters
    ----------
    ax :
        matplotlib.axes.Axes on which to plot chi magnitude profile
    x : array_like
        Array of distance points or list of arrays for multiple data sets.
    data : array_like
        Array of complex-type chi points or list of arrays for multiple data
        sets

    Returns
    -------
    ax : matplotlib.axes.Axes

    """
    try:
        marker = kwargs.pop('marker')
    except KeyError:
        marker = 'x'
    try:
        ls = kwargs.pop('ls')
    except KeyError:
        ls = '-'

    fontsize = 18

    if (np.shape(data[0]), np.shape(x[0])) == ((), ()):
        ax.plot(x, np.abs(data).real, marker=marker, ls=ls, **kwargs)

        if add_fit:
            fit = models.dataset_fun(params, 0, x, model)
            ax.plot(x, np.abs(fit), ls=ls, **kwargs)

    elif np.shape(data[0]) != np.shape(x[0]):
        raise TypeError("The shapes of `data` and `x` are inconsistent")
    else:
        for i, (xa, dat) in enumerate(zip(x, data)):
            ax.plot(xa, np.abs(dat), marker=marker, ls=ls, **kwargs)

            if add_fit:
                fit = models.dataset_fun(params, i, xa, model)
                ax.plot(xa, np.abs(fit), marker='', ls=ls, **kwargs)

    ax.set_ylabel(r'||$\chi$||', fontsize=fontsize)
    ax.set_xlabel(r'Distance ($\mu$m)', fontsize=fontsize)

    # Make the tick labels larger
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Change the number of labels on each axis to five
    ax.locator_params(axis='y', nbins=6, tight=True)

    # Add a light grid
    ax.grid(b=True, which='major', axis='both', alpha=.5)

    y_offset = ax.yaxis.get_offset_text()
    y_offset.set_size(18)
    t = ax.xaxis.get_offset_text()
    t.set_size(18)

    return ax
