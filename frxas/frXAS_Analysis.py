import numpy as np
import h5py
import lmfit
from lmfit import Parameters, minimize


def create_Exp_File(filename: str, Po2s: list, Temp:int=700):
    try:
        if type(filename) == str:
            f = h5py.File(filename + ".h5", "a")
        else:
            print(filename + " is not a string. File not created.")
            return

        for Cond in Po2s:
            f.create_group(Cond + "%_O2")

    except:
        print("An error occured while creating the file")
        f.close()
        return

    try:
        f.attrs.modify("Temperature", Temp)
    except:
        print("Temperature value not entered")

    f.close()
    return

def open_Exp_File(filename: str):
    try:
        f = h5py.File(filename + ".h5", "r+")

    except:
        print("Error encountered")
        return

    return f

def close_Exp_File(filename: str):
    try:
        f = h5py.File(filename + ".h5", "r+")
        f.close()

    except:
        print("Error encountered")
        return

    return

def add_frXAS_Profile(file, Po2, frequency, data):
    f = file
    group = str(Po2) + '%_O2'
    dset = str(frequency) + '_Hz'

    try:
        if dset in f[group].keys():
            del f[group][dset]
            f[group].create_dataset(dset, data=data)
        else:
            f[group].create_dataset(dset, data=data)

        f[group][dset].attrs['frequency'] = frequency

    except:
        print('Data entry unsuccessful')
        return

    return
    
def print_data_shapes(file):
    f = file

    for group in f.keys():
        for dset in f[group].keys():
            print(f[group][dset].name, f[group][dset].shape)

    return
    
def get_Po2_Cond(file):
    gas =[]

    for group in f1.keys():
        name = str(group).split("'")
        g = name[0].split('%')
        gas.append(g[0])
        
    return gas

def extr_adj_1Po2(obj, starts=None):
    gas = obj.attrs['Gas']
    adj_starts = True
    i=0

    if starts is None:
        starts = []
        adj_starts = False

    frequencies = []
    data = []
    data_adj = []
    for group in obj.keys():
        frequency = obj[group].attrs['frequency']

        if adj_starts:
            start = starts[i]
            dat = np.array(obj[group])
            rows = dat.shape[0]
            cols = dat.shape[1]
            dat_adj = np.zeros((rows+1, cols-start+1))
            dat_adj[:rows, :] = dat[:, start-1:].copy()
            dat_adj[0,:] = dat_adj[0, :] - dat_adj[0, 0]
            dat_adj[3,:] = np.sqrt(dat_adj[1,:]**2 + dat_adj[2,:]**2)
            i += 1
        else:
            start = obj[group].attrs['start']
            dat = np.array(obj[group])
            rows = dat.shape[0]
            cols = dat.shape[1]
            dat_adj = np.zeros((rows+1, cols-start+1))
            dat_adj[:rows, :] = dat[:, start-1:].copy()
            dat_adj[0,:] = dat_adj[0, :] - dat_adj[0, 0]
            dat_adj[3,:] = np.sqrt(dat_adj[1,:]**2 + dat_adj[2,:]**2)
            starts.append(start)

        frequencies.append(frequency)
        data.append(dat)
        data_adj.append(dat_adj)

    return dict([('gas', gas), ('frequencies', frequencies), 
                 ('starts', starts), ('data', data), ('data_adj', data_adj)])

def Single_Chi_Model(x, Amp, l_d, sig):
	return Amp * np.exp((-x / l_d) * np.sqrt(1 + 1j * sig))



class ChiModel(lmfit.model.Model):
    
    def __init__(self, *args, **kwargs):
        super(ChiModel, self).__init__(Comp_model2, *args, **kwargs)
        
        self.set_param_hint('Amp', min=0)
        self.set_param_hint('l_d', min=0)
        self.set_param_hint('sig', min=0)
 
    def guess(self, data, x=None, **kwargs):
        verbose = kwargs.pop('verbose', None)
        if x is None:
            return
        
        Amp_guess = data[0]
        Amp_min = 0
        Amp_max = 2
        
                
#         if verbose:
#             print("fmin=", fmin, "fmax=", fmax, "f_0_guess=", f_0_guess)
#             print("Qmin=", Q_min, "Q_max=", Q_max, "Q_guess=", Q_guess, "Q_e_real_guess=", Q_e_real_guess)
        params = self.make_params(Amp=Amp_guess, )
        params['%sAmp' % self.prefix].set(min=Amp_min, max=Amp_max)
        params['%s']
        return lmfit.models.update_param_vals(params, self.prefix, **kwargs)

