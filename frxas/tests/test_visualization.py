import unittest

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters

from ..visualization import plot_chi
from .. import models


class TestCase(unittest.TestCase):
    def test_plot_chi(self):

        x = np.array([0., 5., 10., 20.])
        chi = np.array([1, .75, .5, 0]) + 1j*np.array([0, -.25, -.125, 0])

        chi_fit = np.array([-0.6666667, -0.39103533, -0.22409403, -0.0680762])\
            + 1j*np.array([0., 0.05926727, 0.06952679, 0.04674162])

        xs, chis, chi_fits = [], [], []
        for i in range(3):
            xs.append(x)
            chis.append(chi)
            chi_fits.append(chi_fit)

        params = Parameters()
        for iy, _ in enumerate(chis):
            params.add('ao_%i' % (iy+1), value=1.5, min=1, max=1000)
            params.add('ld_%i' % (iy+1), value=10, min=0.0, max=200)
            params.add('tg_%i' % (iy+1), value=.1, min=0.01, max=100.0)
            params.add('f_%i' % (iy+1), value=1, vary=False)

        # Passing single data set
        _, axes = plt.subplots(nrows=2)
        axes = plot_chi(axes, x, chi)

        xc, yc = axes[0].lines[0].get_xydata().T
        assert (xc == x).all() and (yc == chi.real).all()

        xc, yc = axes[1].lines[0].get_xydata().T
        assert (xc == x).all() and (yc == chi.imag).all()

        # Passing multiple data sets
        _, axes = plt.subplots(nrows=2)
        axes = plot_chi(axes, xs, chis)

        for line in axes[0].lines:
            xc, yc = line.get_xydata().T
            assert (xc == x).all() and (yc == chi.real).all()

        for line in axes[1].lines:
            xc, yc = line.get_xydata().T
            assert (xc == x).all() and (yc == chi.imag).all()

        # Incorrect number of axes
        with self.assertRaises(TypeError):
            plot_chi(axes[0], x, chi)

        # Data shapes don't agree
        with self.assertRaises(TypeError):
            plot_chi(axes, xs[0], chis)

        # Single data set with model fit
        _, axes = plt.subplots(nrows=2)
        axes = plot_chi(axes, x, chi, params=params, add_fit=True,
                        model=models.chi_ideal)
        for i, _ in enumerate(axes[0].lines[::2]):
            xc, yc = axes[0].lines[i].get_xydata().T
            xc_fit, yc_fit = axes[0].lines[i+1].get_xydata().T
            assert (xc == x).all() and (yc == chi.real).all()
            assert (xc == x).all() and np.isclose(yc_fit, chi_fit.real).all()

        for i, _ in enumerate(axes[0].lines[::2]):
            xc, yc = axes[1].lines[i].get_xydata().T
            xc_fit, yc_fit = axes[1].lines[i+1].get_xydata().T
            assert (xc == x).all() and (yc == chi.imag).all()
            assert (xc == x).all() and np.isclose(yc_fit, chi_fit.imag).all()

        # Multiple data set with model fit
        _, axes = plt.subplots(nrows=2)
        axes = plot_chi(axes, xs, chis, params=params, add_fit=True,
                        model=models.chi_ideal)

#        for i, line in enumerate(axes[0].lines):
#            xc, yc = axes[0].lines[i].get_xydata().T
#            xc_fit, yc_fit = axes[0].lines[2*i-1].get_xydata().T
#            print(2*i-1,yc_fit, chi_fit.real)
#            print(np.isclose(yc_fit, chi_fit.real).all())
#            assert (xc == x).all() and (yc == chi.real).all()
#            assert (xc == x).all() and np.isclose(yc_fit, chi_fit.real).all()
#            i+=1
#
#        for i, _ in enumerate(axes[0].lines[::2]):
#            xc, yc = axes[1].lines[i].get_xydata().T
#            xc_fit, yc_fit = axes[1].lines[i+1].get_xydata().T
#            assert (xc == x).all() and (yc == chi.imag).all()
#            assert (xc == x).all() and np.isclose(yc_fit, chi_fit.imag).all()
