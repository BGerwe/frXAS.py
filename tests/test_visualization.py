import unittest

import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameters

from frxas.visualization import plot_chi
from frxas import models


class TestCase(unittest.TestCase):
    def test_plot_chi(self):
        x = np.array([0.0, 5.0, 10.0, 20.0])
        chi = np.array([1, 0.75, 0.5, 0]) + 1j * np.array([0, -0.25, -0.125, 0])

        chi_fit = np.array([-0.6666667, -0.39103533, -0.22409403, -0.0680762]) + 1j * np.array(
            [0.0, 0.05926727, 0.06952679, 0.04674162]
        )

        xs, chis, chi_fits = [], [], []
        for i in range(3):
            xs.append(x)
            chis.append(chi)
            chi_fits.append(chi_fit)

        params = Parameters()
        for iy, _ in enumerate(chis):
            params.add("ao_%i" % (iy + 1), value=1.5, min=1, max=1000)
            params.add("ld_%i" % (iy + 1), value=10, min=0.0, max=200)
            params.add("tg_%i" % (iy + 1), value=0.1, min=0.01, max=100.0)
            params.add("f_%i" % (iy + 1), value=1, vary=False)

        _, axes = plt.subplots(nrows=2)

        # Incorrect number of axes
        with self.assertRaises(TypeError):
            plot_chi(axes[0], x, chi)

        # Data shapes don't agree
        with self.assertRaises(TypeError):
            plot_chi(axes, xs[0], chis)

        # Single data set with model fit
        _, axes = plt.subplots(nrows=2)
        axes = plot_chi(axes, x, chi, params=params, add_fit=True, model=models.chi_ideal)

        for i, line in enumerate(axes[0].lines):
            xc, yc = line.get_xydata().T
            if i % 2 == 0:
                assert (xc == x).all() and np.isclose(yc, chi.real).all()
            else:
                assert (xc == x).all() and np.isclose(yc, chi_fit.real).all()

        for i, line in enumerate(axes[1].lines):
            xc, yc = line.get_xydata().T
            if i % 2 == 0:
                assert (xc == x).all() and np.isclose(yc, chi.imag).all()
            else:
                assert (xc == x).all() and np.isclose(yc, chi_fit.imag).all()

        # Multiple data set with model fit
        _, axes = plt.subplots(nrows=2)
        axes = plot_chi(axes, xs, chis, params=params, add_fit=True, model=models.chi_ideal)

        for i, line in enumerate(axes[0].lines):
            xc, yc = line.get_xydata().T
            if i % 2 == 0:
                assert (xc == x).all() and np.isclose(yc, chi.real).all()
            else:
                assert (xc == x).all() and np.isclose(yc, chi_fit.real).all()

        for i, line in enumerate(axes[1].lines):
            xc, yc = line.get_xydata().T
            if i % 2 == 0:
                assert (xc == x).all() and np.isclose(yc, chi.imag).all()
            else:
                assert (xc == x).all() and np.isclose(yc, chi_fit.imag).all()
