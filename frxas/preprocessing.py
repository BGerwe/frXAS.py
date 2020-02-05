import h5py


def create_Exp_File(filename: str, Po2s: list, Temp: int = 700):
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


def add_frxas_Profile(file, Po2, frequency, data):
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
