import numpy as np
import h5py


def create_hdf5_file(filename: str, po2s: list, temp=700):
    r""" Makes an HDF5 file to store data.

    Parameters
    ----------
    filename : str
        File path name excluding ".h5" extension.
    po2s : list
        List of strings or floats indicating experimental `pO_2` conditions in
        units % `O_2` for data sets being stored .
    temp : int or float
        Temperature of experiment for data sets being stored. Default is
        700 °C.
    """
    try:
        if type(filename) == str:
            f = h5py.File(filename + ".h5", "a")
        else:
            print(filename + " is not a string. File not created.")
            return
    except TypeError:
        print("An error occured while creating the file")
        return

    try:
        for po2 in po2s:
            f.create_group(str(po2) + "%_O2")
    except TypeError:
        print("pO2 conditions not added successfully. Check type in po2s")
        f.close()
        return

    try:
        f.attrs.modify("Temperature", temp)
    except KeyError:
        print("Temperature value not entered")
        f.close()
        return

    f.close()
    return


def open_frxas_file(filename: str):
    r"""Opens existing hdf5 file containing frXAS data.

    Parameters
    ----------
    filename : str
        File path name excluding ".h5" extension.

    Returns
    -------
    f : :class:`~h5py.File`
        Class representing an HDF5 file.
    """
    try:
        f = h5py.File(filename + ".h5", "r+")

    except TypeError:
        print("Error encountered. Check filename.")
        return

    return f


def close_frxas_file(filename: str):
    r"""Closes specified hdf5 file.

    Parameters
    ----------
    filename : str
        File path name excluding ".h5" extension.

    Returns
    -------
    f : :class:`~h5py.File`
        Class representing an HDF5 file.
    """
    try:
        f = h5py.File(filename + ".h5", "r+")
        f.close()

    except TypeError:
        print("Error encountered. Check filename.")
        pass

    return


def add_frxas_profile(file, po2, frequency, data):
    r"""Adds data for a single po2 and frequency experiment to existing file.

    Parameters
    ----------
    file : :class:`~h5py.File` or str
        Class representing an HDF5 file. If `file` is a string, it will attempt
        to open an HDF5 file with that name.
    po2 : str or int
        Experimental `pO_2` condition for data seta being added in units
        % `O_2`.
    frequency : str or int
        Frequency of experimentally applied voltage perturbation for data seta
        being added.
    data : np.ndarray
        Array of frXAS data being added.

    Notes
    -----
    The structure of the data files is:
        file
        |--- pO2 1
        |   |--- frequency 1
        |   |--- frequency 2
        |   ...
        |   |--- frequency n
        |--- pO2 2
        |   |--- frequency 1
        |   ...
        ...
        |---pO2 n
        |   |--- frequency 1
        |   ...

    Use :print_data_shapes: to retrieve existing structure in HDF5 file.
    
    The data is assumed to be organized where the first row is positions,
    the second row is real components of the X-ray signal, and the third row
    is imaginary components of the X-ray signal.
    """
    f = file
    group = str(po2) + '%_O2'
    # frequency is used both to find existing data and to label new data
    dset = str(frequency) + '_Hz'
    
    try:
        f.keys()

    except AttributeError:
        f = open_frxas_file(file)

    try:
        if dset in f[group].keys():
            # delete existing data and replace in case length changes
            del f[group][dset]
            f[group].create_dataset(dset, data=data)
        else:
            f[group].create_dataset(dset, data=data)

        # Not really necessary, but adding exp. conditions as attributes
        f[group][dset].attrs['Frequency'] = frequency
        f[group][dset].attrs['pO2'] = po2

    except (KeyError, TypeError):
        print('Data entry unsuccessful')
        return

    return


def print_data_shapes(file):
    r"""Prints the name of all data sets within each group of the HDF5 file.

    Parameters
    ----------
    file : :class:`~h5py.File` or str
        Class representing an HDF5 file. If `file` is a string, it will attempt
        to open an HDF5 file with that name.
    """
    f = file
    try:
        f.keys()

    except AttributeError:
        f = open_frxas_file(file)

    for group in f.keys():
        if f[group].keys():
            for dset in f[group].keys():
                print(f[group][dset].name, f[group][dset].shape)
        else:
            print(f[group], ' is empty.')
    return


def get_po2_Cond(file):
    r"""Prints the name of all pO2 conditions HDF5 file.

    Parameters
    ----------
    file : :class:`~h5py.File` or str
        Class representing an HDF5 file. If `file` is a string, it will attempt
        to open an HDF5 file with that name.
    """
    gas = []

    f = file
    try:
        f.keys()

    except AttributeError:
        f = open_frxas_file(file)

    for group in f.keys():
        name = str(group).split("'")
        
        # This was in my original code, but it doesn't seem useful
        # TODO: check if this is necessary
        # g = name[0].split('%')
        # gas.append(g[0])

        gas.append(name[0])

    return gas


def extr_adj_1po2(obj, starts=None):
    """Placeholder
"""
    gas = obj.attrs['Gas']
    adj_starts = True
    i = 0

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
            dat_adj[0, :] = dat_adj[0, :] - dat_adj[0, 0]
            dat_adj[3, :] = np.sqrt(dat_adj[1, :]**2 + dat_adj[2, :]**2)
            i += 1
        else:
            start = obj[group].attrs['start']
            dat = np.array(obj[group])
            rows = dat.shape[0]
            cols = dat.shape[1]
            dat_adj = np.zeros((rows+1, cols-start+1))
            dat_adj[:rows, :] = dat[:, start-1:].copy()
            dat_adj[0, :] = dat_adj[0, :] - dat_adj[0, 0]
            dat_adj[3, :] = np.sqrt(dat_adj[1, :]**2 + dat_adj[2, :]**2)
            starts.append(start)

        frequencies.append(frequency)
        data.append(dat)
        data_adj.append(dat_adj)

    return dict([('gas', gas), ('frequencies', frequencies),
                 ('starts', starts), ('data', data), ('data_adj', data_adj)])

def adjust_dataset():
    """
    """
    return