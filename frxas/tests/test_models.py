from frxas import models
import numpy as np
from lmfit import Parameters


def test_dataset_fun():
    # Array for distances
    x = np.linspace(0, 20, 30)

    # Using lmfit to make a parameters dictionary
    params = Parameters()
    params.add('ld_1', value=20)
    params.add('tg_1', value=1.5)
    params.add('ao_1', value=1.2)
    params.add('f_1', value=2)

    correct = \
        np.array([-0.83333333 + 0.j, -0.74352671 + 0.076924j,
                 -0.65629759 + 0.13726812j, -0.57289869 + 0.18305707j,
                 -0.49426079 + 0.21621297j, -0.42103695 + 0.23853677j,
                 -0.35364362 + 0.25169557j, -0.29229845 + 0.25721527j,
                 -0.23705481 + 0.25647743j, -0.18783278 + 0.25071963j,
                 -0.14444679 + 0.24103871j, -0.10663006 + 0.22839617j,
                 -0.07405578 + 0.21362527j, -0.04635545 + 0.19743932j,
                 -0.02313443 + 0.1804407j, -0.00398501 + 0.16313049j,
                 0.01150282 + 0.1459181j,  0.02373271 + 0.12913099j,
                 0.03309502 + 0.11302407j,  0.03996155 + 0.09778874j,
                 0.04468174 + 0.08356144j,  0.04757994 + 0.07043168j,
                 0.04895381 + 0.05844936j,  0.04907355 + 0.04763152j,
                 0.0481818 + 0.03796845j,  0.04649417 + 0.02942906j,
                 0.04420015 + 0.02196573j,  0.04146442 + 0.01551847j,
                 0.03842837 + 0.01001854j,  0.03521183 + 0.00539159j])

    assert np.isclose(models.dataset_fun(params, 0, x, models.chi_ideal),
                      correct).all()


def test_calc_resid():
    # Data generated from ld=20, tg=1.5, a0=1.2, f=2
    data = \
        np.array([-0.83333333 + 0.j, -0.74352671 + 0.076924j,
                 -0.65629759 + 0.13726812j, -0.57289869 + 0.18305707j,
                 -0.49426079 + 0.21621297j, -0.42103695 + 0.23853677j,
                 -0.35364362 + 0.25169557j, -0.29229845 + 0.25721527j,
                 -0.23705481 + 0.25647743j, -0.18783278 + 0.25071963j,
                 -0.14444679 + 0.24103871j, -0.10663006 + 0.22839617j,
                 -0.07405578 + 0.21362527j, -0.04635545 + 0.19743932j,
                 -0.02313443 + 0.1804407j, -0.00398501 + 0.16313049j,
                 0.01150282 + 0.1459181j,  0.02373271 + 0.12913099j,
                 0.03309502 + 0.11302407j,  0.03996155 + 0.09778874j,
                 0.04468174 + 0.08356144j,  0.04757994 + 0.07043168j,
                 0.04895381 + 0.05844936j,  0.04907355 + 0.04763152j,
                 0.0481818 + 0.03796845j,  0.04649417 + 0.02942906j,
                 0.04420015 + 0.02196573j,  0.04146442 + 0.01551847j,
                 0.03842837 + 0.01001854j,  0.03521183 + 0.00539159j])

    model = \
        np.array([-0.90909091 + 0.j, -0.82186961 + 0.0739764j,
                 -0.73699686 + 0.1337577j, -0.65540245 + 0.18089694j,
                 -0.57780057 + 0.21687382j, -0.5047165 + 0.24308417j,
                 -0.43651153 + 0.26083266j, -0.37340611 + 0.27132819j,
                 -0.31550117 + 0.27568159j, -0.26279758 + 0.27490536j,
                 -0.21521372 + 0.2699149j, -0.17260131 + 0.26153115j,
                 -0.1347595 + 0.25048422j, -0.1014473 + 0.23741783j,
                 -0.07239445 + 0.22289432j, -0.04731087 + 0.20740011j,
                 -0.02589472 + 0.19135131j, -0.00783926 + 0.17509956j,
                 0.00716143 + 0.15893782j,  0.01940775 + 0.14310603j,
                 0.02919082 + 0.12779666j,  0.03678948 + 0.11316003j,
                 0.04246806 + 0.09930936j,  0.04647473 + 0.08632548j,
                 0.04904044 + 0.07426128j,  0.05037827 + 0.06314578j,
                 0.05068324 + 0.05298788j,  0.05013235 + 0.04377974j,
                 0.04888502 + 0.03549989j,  0.04708359 + 0.02811593j])

    correct = \
        np.array([0.07575758,  0., 0.0783429, 0.0029476, 0.08069927,
                 0.00351042, 0.08250376, 0.00216012, 0.08353978, -0.00066084,
                 0.08367955, -0.0045474, 0.08286791, -0.00913709, 0.08110765,
                 -0.01411292, 0.07844636, -0.01920417, 0.07496481, -0.02418573,
                 0.07076693, -0.02887619, 0.06597125, -0.03313498, 0.06070372,
                 -0.03685894, 0.05509185, -0.03997851, 0.04926002, -0.04245362,
                 0.04332585, -0.04426962, 0.03739754, -0.0454332, 0.03157197,
                 -0.04596857, 0.02593359, -0.04591375, 0.0205538, -0.04531729,
                 0.01549092, -0.04423521, 0.01079045, -0.04272835, 0.00648575,
                 -0.04086001, 0.00259881, -0.03869396, -0.00085864,
                 -0.03629284, -0.00388411, -0.03371672, -0.00648309,
                 -0.03102214, -0.00866793, -0.02826127, -0.01045664,
                 -0.02548134, -0.01187176, -0.02272435])

    assert np.isclose(models.calc_resid(data, model), correct).all()


def test_objective_fun():
    # List of distance arrays
    x = []
    x.append(np.linspace(0, 80, 20))
    x.append(np.linspace(0, 60, 26))

    # Using lmfit to make a parameters dictionary for one data set
    params = Parameters()
    for i in range(2):
        params.add('ld_%i' % (i+1), value=20)
        params.add('tg_%i' % (i+1), value=1.5)
        params.add('Ao_%i' % (i+1), value=1.2)
        params.add('f_%i' % (i+1), value=2)

    # Data generated from the first distance array with ld=19,
    # tg=1.4, a0=1.1, f=1.5
    d1 = [-9.09090909e-01 + 0.0j, -4.29626046e-01 + 2.62258921e-01j,
          -1.27378677e-01 + 2.47881179e-01j, 1.13122384e-02 + 1.53892646e-01j,
          4.97417267e-02 + 6.94645088e-02j,  4.35468313e-02 + 1.84784057e-02j,
          2.59104777e-02 - 3.82991467e-03j,  1.11401255e-02 - 9.28475351e-03j,
          2.58618647e-03 - 7.60162614e-03j, -9.70751323e-04 - 4.33851776e-03j,
          -1.71036254e-03 - 1.77028724e-03j, -1.31899891e-03 - 3.43204038e-04j,
          -7.22353067e-04 + 2.18316821e-04j, -2.78394774e-04 + 3.11561941e-04j,
          -4.16853221e-05 + 2.27553302e-04j,  4.59456616e-05 + 1.19564690e-04j,
          5.62059955e-05 + 4.32502899e-05j,  3.90393674e-05 + 4.22502004e-06j,
          1.96684160e-05 - 9.26556808e-06j,  6.62209452e-06 - 1.00528416e-05j]
    d1 = np.array(d1)

    # Data generated from the second distance array with ld=19,
    # tg=1.4, a0=1.1, f=1.5
    d2 = [-9.09090909e-01 + 0.00000000e+00j, -6.17618817e-01 + 1.99470905e-01j,
          -3.75830797e-01 + 2.71033366e-01j, -1.95862591e-01 + 2.66598878e-01j,
          -7.45686726e-02 + 2.24097909e-01j, -1.48940288e-03 + 1.68609502e-01j,
          3.59840875e-02 + 1.14876843e-01j,  4.96529412e-02 + 7.01495537e-02j,
          4.91253244e-02 + 3.67635039e-02j,  4.14413715e-02 + 1.41974347e-02j,
          3.12696406e-02 + 5.52480451e-04j,  2.13652144e-02 - 6.48577731e-03j,
          1.30920180e-02 - 9.09422445e-03j,  6.89902785e-03 - 9.05106491e-03j,
          2.70109984e-03 - 7.66288967e-03j,  1.53699201e-04 - 5.79868925e-03j,
          -1.16791650e-03 - 3.97324192e-03j, -1.66526071e-03 - 2.44308197e-03j,
          -1.66740213e-03 - 1.29439457e-03j, -1.41681429e-03 - 5.13528653e-04j,
          -1.07523371e-03 - 3.80069033e-05j, -7.38832426e-04 + 2.10104469e-04j,
          -4.55847788e-04 + 3.04854051e-04j, -2.42803724e-04 + 3.07133166e-04j,
          -9.75656197e-05 + 2.61935851e-04j, -8.81075936e-06 + 1.99361814e-04j]
    d2 = np.array(d2)

    # Correct residuals array from d1
    c1 = [-7.57575758e-02,  0.00000000e+00, -8.27267912e-02,  9.64775725e-03,
          -5.95466460e-02,  3.75676805e-02, -2.42037494e-02,  4.57813767e-02,
          2.18502680e-03,  3.52261284e-02,  1.33711743e-02,  1.86417074e-02,
          1.33984843e-02,  5.38531379e-03,  8.72508756e-03, -1.65584342e-03,
          3.89343409e-03, -3.69379163e-03,  7.58023693e-04, -3.10803832e-03,
          -6.17710212e-04, -1.78211380e-03, -8.67735600e-04, -6.79346601e-04,
          -6.36397617e-04, -5.84052842e-05, -3.26497028e-04,  1.70312343e-04,
          -1.04526756e-04,  1.83335445e-04,  6.38217680e-06,  1.20207057e-04,
          3.99312664e-05,  5.55107076e-05,  3.59810797e-05,  1.42621896e-05,
          2.14379161e-05, -4.16023499e-06,  8.90629739e-06, -8.46399272e-06]
    c1 = np.array(c1)

    # Correct residuals array from d1 and d2 combined
    c2 = [-7.57575758e-02,  0.00000000e+00, -8.27267912e-02,  9.64775725e-03,
          -5.95466460e-02,  3.75676805e-02, -2.42037494e-02,  4.57813767e-02,
          2.18502680e-03,  3.52261284e-02,  1.33711743e-02,  1.86417074e-02,
          1.33984843e-02,  5.38531379e-03,  8.72508756e-03, -1.65584342e-03,
          3.89343409e-03, -3.69379163e-03,  7.58023693e-04, -3.10803832e-03,
          -6.17710212e-04, -1.78211380e-03, -8.67735600e-04, -6.79346601e-04,
          -6.36397617e-04, -5.84052842e-05, -3.26497028e-04,  1.70312343e-04,
          -1.04526756e-04,  1.83335445e-04,  6.38217680e-06,  1.20207057e-04,
          3.99312664e-05,  5.55107076e-05,  3.59810797e-05,  1.42621896e-05,
          2.14379161e-05, -4.16023499e-06,  8.90629739e-06, -8.46399272e-06,
          -7.57575758e-02,  0.00000000e+00, -8.31067065e-02, -9.63710254e-04,
          -8.11958037e-02,  1.39100155e-02, -6.87231288e-02,  3.08101622e-02,
          -4.97318384e-02,  4.22798212e-02, -2.92898627e-02,  4.60149453e-02,
          -1.13341629e-02,  4.29295221e-02,  1.99744738e-03,  3.53825547e-02,
          1.01961860e-02,  2.59256182e-02,  1.38649115e-02,  1.66091536e-02,
          1.41617687e-02,  8.73212459e-03,  1.23593574e-02,  2.87557697e-03,
          9.56713779e-03, -9.23613177e-04,  6.60332643e-03, -2.96250649e-03,
          3.97586209e-03, -3.68647725e-03,  1.92776307e-03, -3.55476831e-03,
          5.09705158e-04, -2.96065900e-03, -3.45661792e-04, -2.19710071e-03,
          -7.61828420e-04, -1.45401058e-03, -8.74357236e-04, -8.33718662e-04,
          -8.04306465e-04, -3.73854033e-04, -6.45834224e-04, -7.04765979e-05,
          -4.63683180e-04,  1.02517206e-04, -2.96495827e-04,  1.79235954e-04,
          -1.62766549e-04,  1.92814828e-04, -6.72566778e-05,  1.70708813e-04]
    c2 = np.array(c2)

    # Testing if it works with one data set
    assert np.isclose(models.objective_fun(params, x[0], d1, models.chi_ideal),
                      c1).all()
    # Testing if it works with multiple data sets
    data = np.array([d1, d2])
    assert np.isclose(models.objective_fun(params, x, data, models.chi_ideal),
                      c2).all()


def test_calc_ao():
    aoo = 1.2
    po2 = 0.01
    po2_ref = 0.21

    ao_correct = 1.60891
    assert np.isclose(models.calc_ao(aoo, po2, po2_ref), ao_correct)


def test_chi_ideal():
    ld = 20
    tg = 1.5
    ao = 1.2
    f = 2
    x = np.linspace(0, 20, 30)

    correct = \
        np.array([-0.83333333 + 0.j, -0.74352671 + 0.076924j,
                 -0.65629759 + 0.13726812j, -0.57289869 + 0.18305707j,
                 -0.49426079 + 0.21621297j, -0.42103695 + 0.23853677j,
                 -0.35364362 + 0.25169557j, -0.29229845 + 0.25721527j,
                 -0.23705481 + 0.25647743j, -0.18783278 + 0.25071963j,
                 -0.14444679 + 0.24103871j, -0.10663006 + 0.22839617j,
                 -0.07405578 + 0.21362527j, -0.04635545 + 0.19743932j,
                 -0.02313443 + 0.1804407j, -0.00398501 + 0.16313049j,
                 0.01150282 + 0.1459181j,  0.02373271 + 0.12913099j,
                 0.03309502 + 0.11302407j,  0.03996155 + 0.09778874j,
                 0.04468174 + 0.08356144j,  0.04757994 + 0.07043168j,
                 0.04895381 + 0.05844936j,  0.04907355 + 0.04763152j,
                 0.0481818 + 0.03796845j,  0.04649417 + 0.02942906j,
                 0.04420015 + 0.02196573j,  0.04146442 + 0.01551847j,
                 0.03842837 + 0.01001854j,  0.03521183 + 0.00539159j])

    assert np.isclose(models.chi_ideal(x, ld, tg, ao, f), correct).all()
