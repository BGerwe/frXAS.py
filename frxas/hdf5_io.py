import numpy as np
import h5py


def create_hdf5_file(filename: str, gases: list, temp=700):
    r""" Makes an HDF5 file to store data.

    Parameters
    ----------
    filename : str
        File path name excluding ".h5" extension.
    gases : list
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
        for gas in gases:
            f.create_group(str(gas))
    except TypeError:
        print("Gas conditions not added successfully. Check type in gases")
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


def add_frxas_profile(file, gas, frequency, data):
    r"""Adds data for a single gas and frequency experiment to existing file.

    Parameters
    ----------
    file : :class:`~h5py.File` or str
        Class representing an HDF5 file. If `file` is a string, it will attempt
        to open an HDF5 file with that name.
    gas : str or int
        Experimental gas condition for data seta being added e.g. `10%_O2`.
    frequency : str or int
        Frequency of experimentally applied voltage perturbation for data seta
        being added.
    data : np.ndarray
        Array of frXAS data being added.

    Notes
    -----
    The structure of the data files is

    ::

        file
        |--- gas 1
        |   |--- frequency 1
        |   |--- frequency 2
        |   ...
        |   |--- frequency n
        |--- gas 2
        |   |--- frequency 1
        |   ...
        ...
        |--- gas n
        |   |--- frequency 1
        |   ...

    Use :print_data_shapes: to retrieve existing structure in HDF5 file.

    The data is assumed to be organized where the first row is positions,
    the second row is real components of the X-ray signal, and the third row
    is imaginary components of the X-ray signal.
    """
    f = file
    group = str(gas)
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
        print(float(gas.split('%')[0]) / 100)
        f[group][dset].attrs['Gas'] = float(gas.split('%')[0]) / 100
        f[group].attrs['Gas'] = float(gas.split('%')[0]) / 100

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


def get_gas_condition(file):
    r"""Retrieves the gas conditions of all HDF5 file.

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


def adjust_dataset(data, start_index):
    """Truncates data sets to begin at `start_index` and makes the position
    values relative to the `start_index` position.

    Parameters
    ----------
    data : np.ndarray
        Array, usually of shape 3 x n, of fr-XAS profile data.
    start_index : int
        Index that data should be truncated to begin at, e.g. data point for
        the edge of the electrode gate in patterned SOFC samples.

    Returns
    -------
    np.ndarray
        Array of data truncated to start_index, has shape m x (x - start_index)
    """
    rows, cols = data.shape
    data_adj = np.zeros((rows, cols-start_index))
    data_adj[:rows, :] = data[:, start_index:].copy()
    data_adj[0, :] = data_adj[0, :] - data_adj[0, 0]

    return data_adj


def get_group_datasets(obj, start_indices=[]):
    """Retrieves all datasets stored in one gas group.

    Parameters
    ----------
    obj : :class:`~h5py.File` group reference
        Reference to a group of an HDF5 file, e.g. file['1%_O2'].
    start_indices : list of ints
        List of indices where the fr-XAS profile should begin

    Returns
    -------
    dict
        Dictionary of data stored in an HDF5 group, including metadata such as
        specified gas condition, and frequencies applied for each dataset.
    """
    gas = obj.attrs['Gas']
    starts = start_indices

    if start_indices:
        # starts = start_indices
        start_adj = True
    else:
        # starts = []
        start_adj = False

    frequencies = []
    data = []
    data_adj = []

    # Cycling through each data set stored in the HDF5 file.
    for i, dset in enumerate(obj.keys()):
        frequency = obj[dset].attrs['Frequency']
    # Start indices are used in passed to the function or previously stored
    # with each data set. Otherwise, assume starting at beginning. Makes new
    # array to store truncated data set.
        if start_adj:
            start = starts[i]
            obj[dset].attrs['Start_Index'] = start
            dat = np.array(obj[dset])
            dat_adj = adjust_dataset(dat, start)

        else:
            try:
                start = obj[dset].attrs['Start_Index']
            except KeyError:
                start = 0

            dat = np.array(obj[dset])
            dat_adj = adjust_dataset(dat, start)
            starts.append(start)

        frequencies.append(frequency)
        data.append(dat)
        data_adj.append(dat_adj)

    return dict([('gas', gas), ('frequencies', frequencies),
                 ('starts', starts), ('data', data), ('data_adj', data_adj)])


def get_all_datasets(file, start_indices=[]):
    """Retrieves all datasets stored in an HDF5 file.

    Parameters
    ----------
    file : :class:`~h5py.File` or str
        Class representing an HDF5 file. If `file` is a string, it will attempt
        to open an HDF5 file with that name.
    start_indices : list of lists
        Should have one entry of integer type for each dataset stored in HDF5.

    Returns
    -------
    list
        List of dicts corresponding to each group in the HDF5 `file`.

    See Also
    --------
    print_data_shapes : Displays structure of data in an HDF5 file.
    get_group_datasets : Retrieves data in a single group in an HDF5 file.
    """
    f = file
    try:
        f.keys()

    except AttributeError:
        f = open_frxas_file(file)

    data = []

    for i, group in enumerate(f.keys()):
        if f[group].keys() and len(start_indices) > 0:
            data.append(get_group_datasets(f[group], start_indices[i]))
        elif f[group].keys() and len(start_indices) == 0:
            data.append(get_group_datasets(f[group]))
        else:
            print(f[group], ' is empty.')

    return data


def unpack_data(hdf_file, kind='data_adj'):
    """
    """
    x, data, freqs, gas = [], [], [], []

    for group in hdf_file:
        for dset in group[kind]:
            x.append(np.array(dset[0]))
            data.append(np.array(dset[1] + 1j * dset[2]))
            gas.append(group['gas'])
        freqs.append(group['frequencies'])

    frequencies = [item for sublist in freqs for item in sublist]

    return x, data, frequencies, gas
