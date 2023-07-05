# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Convenience class used to save features to an HDF5 file.
"""
import h5py
import numpy as np


class HDF5Writer(object):
    """Convenience class used to save features to an HDF5 file.

    https://docs.h5py.org/en/stable/high/dataset.html

    Args:
        filename (string): Path to the destination file
        dataset_name (string): name of the dataset to write.
        shape (int List): shape of the features written to disks, the actual shape of the dataset
            is `[-1,] + shape` (since the total number of features written to disk is not known at initialisation time.)
        dtype (np.dtype): dtype specifying the features precision.
        attrs {dictionary}: dictionary of attributes for the dataset. It consists in metadata that needs to be
            contained in the result file.
        mode (string): mode for opening the file. Defaults to write `"w"`.
        store_as_uint8 (boolean): if True, casts to byte before storing to save space.
            Only supports 0-1 normalized data.

    Attributes:
        index (int): correspond to the number of feature already written to HDF5.

    Examples:
        >>> f = HDF5Writer("example.h5", "data", [2, 480, 320], dtype=np.uint8)
        >>> f.write(np.empty((15, 2, 480, 320), dtype=np.uint8))
        >>> f.write(np.zeros((12, 2, 480, 320), dtype=np.uint8))
        >>> f.close()
    """

    def __init__(self, filename, dataset_name, shape, dtype=np.uint8, attrs={}, mode="w", store_as_uint8=False):
        self.filename = filename
        self.shape = tuple(shape)
        self.dataset_size_increment = 100
        self.file = h5py.File(filename, mode)
        self.store_as_uint8 = store_as_uint8
        self.dset = self.file.create_dataset(
            dataset_name, (self.dataset_size_increment,) + self.shape, maxshape=(None,) + self.shape, dtype=np.uint8
            if store_as_uint8 else dtype, compression='gzip', chunks=(1,) + self.shape)
        
        for key, value in attrs.items():
            if value is not None:
                self.dset.attrs[key] = value
        if store_as_uint8:
            self.dset.attrs['store_as_uint8'] = True

        #add defualt delta_t mode
        if  self.dset.attrs.get("mode") is None:
            self.dset.attrs["mode"] = "delta_t"
        # for n_event mode we need to keep track of frames timestamps
        # last_timestamps corresponds to time of the last event in the slice 
        if self.dset.attrs["mode"] == "n_events":
            self.ts_dset = self.file.create_dataset(
                "last_timestamps", (self.dataset_size_increment,), maxshape=(None,), dtype=np.int64
                , compression='gzip', chunks=(1,))

        self.index = 0
        assert mode != "r", "mode r doesn't allow to write a file !"

    def __repr__(self):
        """String representation of a `HDF5Writer` object.

        Returns:
            string describing the MetavisionWriter state and attributes
        """
        string = f"HDF5Writer({self.filename})\n"
        string += f"Current dataset shape : {self.dset.shape}, dtype: { str(self.dset.dtype)}\n"
        string += "Attributes:\n{"
        for key in self.dset.attrs:
            string += f"    {key}: {self.dset.attrs[key]},\n"
        string += "}"
        return string

    def write(self, array, last_timestamps=None):
        """Appends an array of features to the dataset.

        The underlying hdf5 dataset gets extended when necessary.

        Args:
            array (np.ndarray): feature array, its shape must be `[*,] + self.shape` and its dtype must be convertible
                to the dtype of the dataset.
            last_timestamps (np.ndarray): timestamps of the last events for each feature slice, its shape must be equal to array.shape[0]
        """
        while self.index + len(array) >= len(self.dset):
            # if go past the dataset size, we increase it further
            self.dset.resize(((self.dset.shape[0] + self.dataset_size_increment,) + self.shape))
            if self.dset.attrs["mode"] == "n_events":
                self.ts_dset.resize(((self.ts_dset.shape[0] + self.dataset_size_increment,)))
        if self.store_as_uint8:
            assert array.max() <= 1, "store_as_uint8 option requires [0 -> 1] normalized data"
            assert array.min() >= 0, "store_as_uint8 option requires [0 -> 1] normalized data"
            self.dset[self.index: self.index + len(array)] = np.uint8(np.around(255 * array))
        else:
            self.dset[self.index: self.index + len(array)] = array

        if self.dset.attrs["mode"] == "n_events":
            assert last_timestamps is not None, "you must write the timestamps of the last events if mode == n_events"
            assert len(last_timestamps) == len(array), f"array and last_timestamps shapes must be the same: {len(last_timestamps)}, {len(array)}"
            self.ts_dset[self.index: self.index + len(array)] = np.int64(last_timestamps)
        
        if last_timestamps is not None and self.dset.attrs["mode"] == "delta_t":
            print("Warning: if mode == delta_t, last_timestamps are ignored!")

        self.index += len(array)

    def flush(self):
        self.file.flush()

    def close(self):
        self.dset.resize(((self.index,) + self.shape))
        if self.dset.attrs["mode"] == "n_events":
            self.ts_dset.resize(((self.index,)))
        self.file.close()

    def set_cursor(self, index):
        """Sets the cursor of where to write the next features. Use with caution !

        Can be used to overwrite or drop some already written data.

        Args:
            index (int): new cursor position.

        Examples:
            >>> # ... some feature were written
            >>> hdf5_writer.set_cursor(hdf5_writer.index - 1)  # drop the last frame
            >>> hdf5_writer.set_cursor(0)  #  ! drop all previously written features !
        """
        assert index <= self.index, f"new index {index} must be below current index : {self.index}"
        self.index = max(int(index), 0)

    def __exit__(self, type, value, traceback):
        self.close()

    def __enter__(self):
        return self

    def __del__(self):
        self.file.close()
