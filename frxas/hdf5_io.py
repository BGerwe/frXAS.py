import itertools
import re
import numpy as np
import h5py
import lmfit
from . import time_domain


def create_frxas_profile_hdf5(filename: str, gases: list, temp=700):
    """
    Makes an HDF5 file to store data for fr-XAS profiles.

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
        if isinstance(filename, str):
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


def open_hdf5(filename: str, mode="r+"):
    """Opens existing hdf5 file containing frXAS data.

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
        f = h5py.File(filename + ".h5", mode)

    except TypeError:
        print("Error encountered. Make sure filename is a valid path for an",
              "existing file.")
        return

    return f


def close_hdf5(filename: str):
    """Closes specified hdf5 file.

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


def add_frxas_profile(file, gas, frequency, positions, data, harmonic=1):
    """Adds data for a single gas and frequency experiment to existing file.

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
    positions : np.ndarray
        Array of measurement positions for data.
    data : np.ndarray
        Array of frXAS data being added.
    harmonic : int
        Index of harmonic for data added.

    Notes
    -----
    The structure of the data files is

    ::

        file
        |--- gas 1
        |   |--- frequency 1
        |   |   |--- harmonic 1
        |   |   |--- harmonic 2
        |   |   ...
        |   |--- frequency 2
        |   ...
        |   |--- frequency n
        |--- gas 2
        |   ...
        ...
        |--- gas n
        |   ...

    Use :print_data_shapes: to retrieve existing structure in HDF5 file.

    The data is assumed to be organized where the first row is positions,
    the second row is real components of the X-ray signal, and the third row
    is imaginary components of the X-ray signal.

    """
    if data.shape[-1] != positions.shape[-1]:
        raise ValueError('Data and position shapes don\'t match')

    f = file
    group = str(gas)
    # frequency is used both to find existing data and to label new data
    subgroup = str(frequency) + '_Hz'
    dset = 'Harmonic ' + str(harmonic)

    try:
        f.keys()

    except AttributeError:
        f = open_hdf5(file)

    try:
        if subgroup not in f[group].keys():
            f[group].create_group(subgroup)

        if dset in f[group][subgroup].keys():
            # delete existing data and replace in case length changes
            del f[group][subgroup][dset]
            f[group][subgroup].create_dataset(dset, data=data)
        else:
            f[group][subgroup].create_dataset(dset, data=data)

        # Positions will be shared across all harmonics within each frequency
        f[group][subgroup].attrs['Positions'] = positions
        # Not really necessary, but adding exp. conditions as attributes
        f[group][subgroup].attrs['Frequency'] = frequency
        print(float(gas.split('%')[0]) / 100)
        f[group][subgroup].attrs['Gas'] = float(gas.split('%')[0]) / 100
        f[group].attrs['Gas'] = float(gas.split('%')[0]) / 100

    except (KeyError, TypeError):
        print('Data entry unsuccessful')
        return

    return


def print_data_shapes(file):
    """Prints the name of all data sets within each group of the HDF5 file.

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
        f = open_hdf5(file)

    # Looking through Gas condition groups
    for group in f.keys():
        if f[group].keys():
            # Looking through frequencies groups
            for dset in f[group].keys():
                print(f[group][dset].name,
                      f[group][dset].attrs.get('Positions').shape)
        else:
            print(f[group], ' is empty.')
    return


def get_gas_condition(file):
    """Retrieves the gas conditions of all HDF5 file.

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
        f = open_hdf5(file)

    for group in f.keys():
        name = str(group).split("'")

        # This was in my original code, but it doesn't seem useful
        # TODO: check if this is necessary
        # g = name[0].split('%')
        # gas.append(g[0])

        gas.append(name[0])

    return gas


def adjust_dataset(data, positions, start_index):
    """Truncates data sets to begin at `start_index` and makes the position
    values relative to the `start_index` position.

    Parameters
    ----------
    data : np.ndarray
        Array, usually of shape 2 x n, of fr-XAS profile data.
    positions : np.ndarray
        Possss
    start_index : int
        Index that data should be truncated to begin at, e.g. data point for
        the edge of the electrode gate in patterned SOFC samples.

    Returns
    -------
    np.ndarray
        Array of data truncated to start_index, has shape m x (x - start_index)

    """
    pos = positions
    rows, cols = data.shape
    data_adj = np.zeros((rows, cols-start_index))
    data_adj[:rows, :] = data[:, start_index:].copy()
    # Make positions in first row relative to first value
    pos_adj = pos[start_index:].copy()
    pos_adj = pos_adj - pos_adj[0]

    return data_adj, pos_adj


def get_group_datasets(subgroup, harmonic=1, start_index=None):
    """Retrieves all datasets stored in one gas group.

    Parameters
    ----------
    subgroup : :class:`~h5py.File` group reference
        Reference to a group of an HDF5 file, e.g. file['1%_O2'].
    harmonic : int
        Index of harmonic data to retrieve.
    start_index : int
        Index where the fr-XAS profile should begin

    Returns
    -------
    dict
        Dictionary of data stored in an HDF5 group, including metadata such as
        specified gas condition, and frequencies applied for each dataset.

    """
    start = start_index
    if start is None:
        start_adj = False
        start = 0
    else:
        start_adj = True

    frequency = subgroup.attrs['Frequency']
    pos = subgroup.attrs.get('Positions')
    try:
        dset = subgroup[f'Harmonic {harmonic}']
    except KeyError:
        print(f'Harmonic {harmonic} not found in {subgroup}')
        return frequency, [], [], [], [], []

    # Start indices are used in passed to the function or previously stored
    # with each data set. Otherwise, assume starting at beginning. Makes new
    # array to store truncated data set.
    if start_adj:
        subgroup.attrs['Start_Index'] = start
        data = np.array(dset)
        data_adj, pos_adj = adjust_dataset(data, pos, start)
    else:
        try:
            start = subgroup.attrs['Start_Index']
        except KeyError:
            start = 0

        data = np.array(dset)
        data_adj, pos_adj = adjust_dataset(data, pos, start)

    return frequency, start, data, data_adj, pos, pos_adj


def get_all_datasets(file, harmonic=1, start_indices=[]):
    """Retrieves all datasets stored in an HDF5 file.

    Parameters
    ----------
    file : :class:`~h5py.File` or str
        Class representing an HDF5 file. If `file` is a string, it will attempt
        to open an HDF5 file with that name.
    harmonic : int
        Index of harmonic data to retrieve.
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
        f = open_hdf5(file)

    data_list = []

    for i, key in enumerate(f):
        group = f[key]
        data_dict = {}
        data_dict['gas'] = group.attrs.get('Gas')

        # Make lists of all info within a gass condition
        frequencies, starts, data, data_adj = [], [], [], []
        posi, posi_adj = [], []
        for j, subgroup in enumerate(group.keys()):
            if group[subgroup].keys() and len(start_indices) > 0:
                frequency, start, dat, dat_adj, pos, pos_adj = \
                    get_group_datasets(group[subgroup], harmonic=harmonic,
                                       start_index=start_indices[i][j])
            elif group[subgroup].keys() and len(start_indices) == 0:
                frequency, start, dat, dat_adj, pos, pos_adj = \
                    get_group_datasets(group[subgroup], harmonic=harmonic)
            else:
                print(group[subgroup], ' is empty.')
                continue
            frequencies.append(frequency)
            starts.append(start)
            data.append(dat)
            data_adj.append(dat_adj)
            posi.append(pos)
            posi_adj.append(pos_adj)

        data_dict['frequencies'] = frequencies
        data_dict['starts'] = starts
        data_dict['data'] = data
        data_dict['data_adj'] = data_adj
        data_dict['positions'] = posi
        data_dict['positions_adj'] = posi_adj
        data_list.append(data_dict)

    return data_list


def unpack_data(data_dict, kind='adj'):
    """
    data_dict : dict
        Contains all data of frxas profiles stored in hdf5 file.
    kind : str
        Options are 'raw' for unmodified data or 'adj' for arrays starting at
        specified index
    """
    xs, data, freqs, gas, sizes = [], [], [], [], []

    if kind == 'adj':
        kind = '_adj'
    elif kind == 'raw':
        kind = ''
    else:
        raise ValueError(f"Invalid `kind` selection. Valid choices are \'adj\' \
                         or \'raw\', but {kind} was provided.")

    for group in data_dict:
        x, dat = [], []
        x = group[f'positions{kind}']
        for dset in group[f'data{kind}']:
            dat.append(np.array(dset[0] + 1j * dset[1]))
            gas.append(group['gas'])
        # Adding individual data sets to list, sorted by frequency
        xs.append([a for _, a in sorted(zip(group['frequencies'], x))])
        data.append([b for _, b in sorted(zip(group['frequencies'], dat))])
        freqs.append(sorted(group['frequencies']))
        # Tracks how many datasets per gas condition
        sizes.append(sum(1 for c in dat))

    # Flatten list of lists into list of arrays
    data = list(itertools.chain.from_iterable(a for a in data))
    xs = list(itertools.chain.from_iterable(b for b in xs))

    frequencies = [item for sublist in freqs for item in sublist]

    return xs, data, frequencies, gas, sizes


def save_lmfit_fit_statistics(hdf_file, fit_model):
    """Store fit statistics from `lmfit.ModelResult` in open hdf5 file.

    Parameters
    ----------
    hdf_file : :class:`~h5py.File`
        Class representing an HDF5 file.
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.

    """
    f = hdf_file
    try:
        f.create_group("Fit Statistics")
    except ValueError:
        pass

    # Adding these parts as attributes seems easier than making into a dataset
    fit_stats = f["Fit Statistics"]
    fit_stats.attrs['Fitting Method'] = fit_model.method
    fit_stats.attrs['Function Evals'] = fit_model.nfev
    fit_stats.attrs['Data Points'] = fit_model.ndata
    fit_stats.attrs['Number of Variables'] = fit_model.nvarys
    fit_stats.attrs['Chi Squared'] = fit_model.chisqr
    fit_stats.attrs['Reduced Chi Squared'] = fit_model.redchi
    fit_stats.attrs['Akaike Info Criteria'] = fit_model.aic
    fit_stats.attrs['Bayesian Info Criteria'] = fit_model.bic

    # Kind of clunky way to get lmfit model info out, but it works for now
    mod_str = str(fit_model.model).split(": ")[-1][:-1]
    mod_comps = mod_str.split('Model(')
    fcns, prefixes = [], []
    for mod in mod_comps[1:]:
        fcn = mod.split(",")[0]
        match = re.search(r'h\d+_', mod)
        if match:
            prefix = match.group()
        fcns.append(fcn)
        prefixes.append(prefix)

    fit_stats.attrs['Model Functions'] = fcns
    fit_stats.attrs['Model Prefixes'] = prefixes
    return


def save_lmfit_ind_varis(hdf_file, fit_model):
    """Store independent variables from `lmfit.ModelResult` in open hdf5 file.

    Parameters
    ----------
    hdf_file : :class:`~h5py.File`
        Class representing an HDF5 file.
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.

    """
    f = hdf_file
    try:
        f.create_group("Independent Variables")
    except ValueError:
        pass

    # Store independent variables: frequencies, freq_in, window_param
    # Using datasets since frequencies will be too large for storing as attr
    ind_varis = f['Independent Variables']
    for kw in fit_model.userkws:
        try:
            if kw in ind_varis.keys():
                del ind_varis[kw]
                ind_varis.create_dataset(kw, data=fit_model.userkws[kw])
            else:
                ind_varis.create_dataset(kw, data=fit_model.userkws[kw])
        except (KeyError, RuntimeError):
            print(f'Data entry for {kw} was unsuccessful. Check data',
                  f'type of {kw}')
    return


def save_lmfit_varis(hdf_file, fit_model, save_data):
    """Store variables from `lmfit.ModelResult` in open hdf5 file.

    Parameters
    ----------
    hdf_file : :class:`~h5py.File`
        Class representing an HDF5 file.
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.

    """
    f = hdf_file
    try:
        f.create_group("Variables")
    except ValueError:
        pass
    # Store variables AKA parameters from lmfit model
    varis = f['Variables']

    param_info = np.zeros((len(fit_model.params), 6))
    param_names = []
    for i, kw in enumerate(fit_model.params.keys()):
        param_names.append(kw)
        param = np.array([fit_model.params[kw].value, fit_model.params[kw].min,
                         fit_model.params[kw].max, fit_model.params[kw].vary,
                         fit_model.params[kw].stderr,
                         fit_model.params[kw].init_value])
        param_info[i, :] = param

    # Store names as attribute because HDF doesn't play nice with arrays
    # of strings
    varis.attrs['Parameter Names'] = param_names
    try:
        if 'Parameter Info' in varis.keys():
            del varis['Parameter Info']
            varis.create_dataset('Parameter Info', data=np.array(param_info))
            varis['Parameter Info'].attrs['Values Order'] = \
                ['Value', 'Min', 'Max', 'Varies', 'Std Err', 'Initial Value']
        else:
            varis.create_dataset('Parameter Info', data=np.array(param_info))
            varis['Parameter Info'].attrs['Values Order'] = \
                ['Value', 'Min', 'Max', 'Varies', 'Std Err', 'Initial Value']
        if save_data:
            if 'Data' in varis.keys():
                del varis['Data']
                varis.create_dataset('Data', data=fit_model.data)
            else:
                varis.create_dataset('Data', data=fit_model.data)
    except (KeyError, RuntimeError):
        print('Parameter names and value entry unsuccessful.')

    return


def save_time_domain_fit(filename: str, fit_model, save_data=False):
    """Store most important information from `lmfit.ModelResult` in hdf5 file.
    Parameters
    ----------
    filename : str
        File path name excluding ".h5" extension.
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.
    save_data : bool, optional
        Option to save data used for fitting into file. May result in large
        file sizes.

    """
    f = open_hdf5(filename, mode="a")

    save_lmfit_fit_statistics(f, fit_model)
    save_lmfit_ind_varis(f, fit_model)
    save_lmfit_varis(f, fit_model, save_data)

    f.close()
    return


def load_lmfit_fit_statistics(hdf_file):
    """Loads and unpacks data in hdf5 file stored by save_time_domain_fit().

    Parameters
    ----------
    hdf_file : :class:`~h5py.File`
        Class representing an HDF5 file.
    Returns
    -------
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.

    """
    f = hdf_file

    fit_stats = f['Fit Statistics']
    fcns = fit_stats.attrs.get('Model Functions')
    prefixes = fit_stats.attrs.get('Model Prefixes')

    model_form = None
    for fcn, prefix in zip(fcns, prefixes):
        if model_form:
            model_form += \
                lmfit.Model(getattr(time_domain, fcn), prefix=prefix,
                            independent_vars=['frequencies', 'freq_in',
                                              'window_param'])
        else:
            model_form = \
                lmfit.Model(getattr(time_domain, fcn), prefix=prefix,
                            independent_vars=['frequencies', 'freq_in',
                                              'window_param'])

    # Initialize `ModelResult` class with empty parameters
    params = lmfit.Parameters()
    fit_model = lmfit.model.ModelResult(model_form, params)
    # Load in fit statistics
    fit_model.method = fit_stats.attrs['Fitting Method']
    fit_model.nfev = fit_stats.attrs['Function Evals']
    fit_model.ndata = fit_stats.attrs['Data Points']
    fit_model.nvarys = fit_stats.attrs['Number of Variables']
    fit_model.chisqr = fit_stats.attrs['Chi Squared']
    fit_model.redchi = fit_stats.attrs['Reduced Chi Squared']
    fit_model.aic = fit_stats.attrs['Akaike Info Criteria']
    fit_model.bic = fit_stats.attrs['Bayesian Info Criteria']

    return fit_model


def load_lmfit_ind_varis(hdf_file, fit_model):
    """Loads and unpacks data in hdf5 file stored by save_time_domain_fit().

    Parameters
    ----------
    hdf_file : :class:`~h5py.File`
        Class representing an HDF5 file.
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.

    """
    f = hdf_file
    # Load in independent variables
    ind_varis = {}
    for name, val in f['Independent Variables'].items():
        ind_varis[name] = np.array(val)
    fit_model.userkws = ind_varis

    return


def load_lmfit_varis(hdf_file, fit_model):
    """Loads and unpacks data in hdf5 file stored by save_time_domain_fit().

    Parameters
    ----------
    hdf_file : :class:`~h5py.File`
        Class representing an HDF5 file.
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.

    """
    # Load in parameters
    f = hdf_file
    params = lmfit.Parameters()
    varis = f['Variables']
    for name, info in zip(varis.attrs['Parameter Names'],
                          varis['Parameter Info']):
        params.add(name, value=info[0], min=info[1], max=info[2], vary=info[3])
        # For compatibility with previous version of saving lmfit
        if len(info) == 6:
            params[name].stderr = info[4]
            params[name].init_value = info[5]
            # If stderr exists for one parameters, uncertainties were evaluated
            # So turn off warning
            if info[4]:
                fit_model.errorbars = True

    fit_model.params = params
    if 'Data' in varis.keys():
        fit_model.data = np.array(varis['Data'])

    return


def load_time_domain_fit(filename: str):
    """Loads hdf5 fit file into lmfit.ModelResult object.

    Parameters
    ----------
    filename : str
        File path name excluding ".h5" extension.
    Returns
    -------
    fit_model : lmfit.model.ModelResult
        The object returned by lmfit.Model.fit() containing statistics, data,
        and best fit parameters.

    """
    f = open_hdf5(filename)

    fit_model = load_lmfit_fit_statistics(f)
    load_lmfit_ind_varis(f, fit_model)
    load_lmfit_varis(f, fit_model)

    f.close()
    return fit_model


def extract_time_domain_fit(file, suffix, harmonics=1, fit_dict=None):
    """Extracts essential data from HDF5 fit files needed for modeling.

    Parameters
    ----------
    file : :class:`~h5py.File`
        Class representing an HDF5 file.
    suffix : str
        Used for associating position values and absorbance data.
    harmonics : int
        Index of highest harmonic coefficients to extract. Should not exceed
        value of harmonic index time domain data is fit to.
    fit_dict : dict
        Object for storing extracted data. A unique dictionary should be used
        for each gas and frequency condition measured.

    Returns
    -------
    fit_dict
    """

    if not fit_dict:
        fit_dict = {}

    if 'positions' in fit_dict.keys():
        fit_dict['positions'][suffix] = float(file.userkws['position'])
    else:
        fit_dict['positions'] = {suffix: float(file.userkws['position'])}

    if 'ir_avg' in fit_dict.keys():
        fit_dict['ir_avg'][suffix] = float(file.userkws['ir_avg'])
    else:
        fit_dict['ir_avg'] = {suffix: float(file.userkws['ir_avg'])}

    for k in range(1, harmonics+1):
        if f'h{k}_re' in fit_dict.keys():
            fit_dict[f'h{k}_re'][suffix] = file.params[f'h{k}_re_comp'].value
            fit_dict[f'h{k}_im'][suffix] = file.params[f'h{k}_im_comp'].value
        else:
            fit_dict[f'h{k}_re'] = {suffix: file.params[f'h{k}_re_comp'].value}
            fit_dict[f'h{k}_im'] = {suffix: file.params[f'h{k}_im_comp'].value}

    return fit_dict


def dict_vals_to_array(vals):
    """Converts `dict_values` type to numpy array.

    Parameters
    ----------
    vals : dict_values

    Returns
    -------
    numpy array
    """
    return np.array(list(vals.values()))
