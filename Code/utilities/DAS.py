import h5py
import datetime
import numpy as np



def open_H5_file(file_path):
    """
    open h5 files and return DAS data 
    """

    f_in = h5py.File(file_path, 'r')
    key = list(f_in.keys())

    DAS_data = f_in['DAS'][:]
    length = DAS_data.shape[1]
    # channels betwee 81 and 4978 are at distances from interigator of 0 to 20km
    DAS_data = DAS_data[:, 81:4978]
    DAS_data = DAS_data[:,0:length:2]
    
    # creating time stamp
    tmp = file_path.split('.h5')[0].split('/')[-1].split('_')[2:5]  #a bit convoluted. 
    start_datetime = datetime.datetime.strptime(tmp[1] + ' ' + tmp[2],'%Y%m%d %H%M%S.%f')  # in datetime
    return DAS_data, start_datetime

def one_bit_cross_cor(source, reciever, offset):
    offset
    n_samples = source.shape[0]
    source_ones = np.sign(source)
    reciever_ones= np.sign(reciever)
    conv = source_ones[:n_samples-offset] * reciever_ones[:,offset:]
    return np.sum(conv, axis=1)