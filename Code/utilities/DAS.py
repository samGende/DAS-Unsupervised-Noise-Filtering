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
    DAS_data = DAS_data[:, 81:]
    DAS_data = DAS_data[:,0:length:2]
    
    # creating time stamp
    tmp = file_path.split('.h5')[0].split('/')[-1].split('_')[2:5]  #a bit convoluted. 
    start_datetime = datetime.datetime.strptime(tmp[1] + ' ' + tmp[2],'%Y%m%d %H%M%S.%f')  # in datetime
    return DAS_data, start_datetime