import numpy as np
import h5py

class HDF5Writer():
    def __init__(self, data_size, dataset_file='dataset.hd5'):
        self.dataset_file = dataset_file
        self.dataset_fp = h5py.File(self.dataset_file, "w")
        self.dsfp = self.dataset_fp.create_dataset('data', data_size, dtype='i8')
        
    def write(self, i, data):
        self.dsfp[i] = data
    
    def close():
        self.dataset_fp.close()
        
class HDF5Reader():
    def __init__(self, dataset_file='dataset.hd5'):
        self.dataset_file = dataset_file
    
    def get_batches(self, batch_size):
        dataset_fp = h5py.File(self.dataset_file, "r")
        dsfp = dataset['data']
        for i in xrange(len(dsfp) / batch_size):
            yield dsfp[i * batch_size: i * batch_size + batch_size]
        dataset_fp.close()
