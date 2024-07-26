from utilities import cwt
from utilities import DAS 
import os
import numpy as np

#directory where DAS data is located 
DAS_data_directory = "../../../data/earthquakes/sissle/eq_data_50Hz"

#dir where transforms will be saved 
out_dir = "./Data/CWT_NZ_NOSUB/"
dir_list = os.listdir(DAS_data_directory)
#Last 5 files in data are not files with DAS data
dir_list = sorted(dir_list)[1:-5]

print(len(dir_list))

file_list = []
for folder in dir_list:
    files_in_folder = os.listdir(DAS_data_directory +'/'+ folder)
    file_list.extend([folder+'/' + file for file in os.listdir(DAS_data_directory +'/'+ folder)])

print(len(file_list))



#sub sample 50hz data to 2hz data  
def sub_sample(data):
    data =np.reshape(data, (data.shape[0], 25, -1))
    return np.mean(data, axis=1)


#sorted_list = [ "20160905_06:17:54.npy"]
#load one file to check n_channels and n_samples
sample_data,_ = DAS.open_H5_file(DAS_data_directory +'/'+file_list[0])
sample_data = sample_data.T

#parameters for cwt
dt =0.5
dj =0.5
w0 =8
minSpaceFrq = 0.002
maxSpaceFrq = 0.12
n_features = 30
n_samples = sample_data.shape[1]

n_channels = sample_data.shape[0]
samples_per_second = 50
samples_per_sub_sample = 25
space_log = np.logspace(np.log10(minSpaceFrq), np.log10(maxSpaceFrq), n_features)
time_scales= cwt.get_scales(dt, dj, w0, n_samples)

count = 0
transform_data = np.zeros((3704, 4*n_samples))

file_list.sort()

for file in file_list:
    data, start_time = DAS.open_H5_file(DAS_data_directory + '/' + file)
    #data = np.load(file)
    #take a 4 minute window 
    transform_data[:, count*n_samples:(count+1)*n_samples] = data.T
    if(count == 3):
        print(data.shape)
        transform = cwt.transform_window(transform_data, n_channels, samples_per_second, samples_per_sub_sample, space_log, time_scales, start_window=0, end_window=11950, window_length=478, subsampling = False, derivative = False)
        filename = file.split("/")[0]
        np.save(out_dir  + "cwt_" + filename, transform)
        print(transform.shape)
        count= 0
    else:
        count+=1


start_window = 0 
end_window= 11950
window_length = 478

cwt.save_cwt_info(sample_data.shape, samples_per_second, samples_per_sub_sample, space_log, time_scales, .2, 24, w0, start_window, end_window, window_length, True, file_list[0], file_list[-1])


