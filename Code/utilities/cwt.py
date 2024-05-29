from obspy.core.trace import Trace
import time
import os 
import torch

if torch.cuda.is_available():
    import cupy as np
    gpu = True
else:
    import numpy as np
    gpu = False




def cwt_time_vec(signal, log_space, omega_0, dt, space=False):
    #signal shape is (channels , samples)
    #calculate the scales for cwt
    if(gpu):
      log_space = np.array(log_space)
      signal = np.array(signal)
    scales = (log_space*(omega_0 + np.sqrt(2 + omega_0**2)))/ (4 * np.pi)
    #calculate angular frequencies
    angular_freq = ((2*np.pi * np.arange(signal.shape[1])) / (signal.shape[1]* dt))
    # reflect the second half of the array
    angular_freq[(signal.shape[1] //2):] *= -1
    #H(w)
    heavy_step = angular_freq > 0
 
    #compute wavelet point 
    wavelet = np.zeros((signal.shape[1], scales.shape[0]), dtype=np.complex128)
    wavelet[heavy_step,:]= np.sqrt((scales *2 * np.pi)/ dt) * (np.pi ** -.25) * np.exp(-((np.outer(angular_freq[heavy_step], scales) - omega_0)**2) / 2)

    #zero mean 
    signal -= np.mean(signal)
    
    fft_signal = np.fft.fft(signal)
    
    wavelet = wavelet[:, np.newaxis, :]
    signal = signal[np.newaxis, :,:]
    result = np.array((wavelet.shape[1], fft_signal.shape[1]), dtype=np.complex128)
    multplied_start = time.time()
    multiplied = wavelet.T * fft_signal
    multi_end = time.time()
    print("multiplication took", multplied_start -multi_end)
    
    
    if(not space):
        multiplied = multiplied.transpose(0,2,1)
        
    print(multiplied.shape)
    
    inverse_start = time.time()
    result = np.fft.ifft(multiplied, axis=1)
    inverse_end = time.time()
    print("invers fft took", inverse_end - inverse_start)
    
    return result




def cwt_space_vec(signal, time_scales, omega_0, dt):
    """take the cwt across the channels of an array

    Args:
        signal (channels, samples): the signal that will be transformed 
        log_space (np.array(scales,)): the log space that is used to generate scales 
        omega_0 (): parameter for the cwt 
        dt (float): time intervale the the samples = 1/samples per second

    Returns:
        _type_: _description_
    """
    if (gpu):
      log_space = np.array(log_space)
      signal = np.array(signal)
    #calculate the scales for cwt
    #scales = (log_space*(omega_0 + np.sqrt(2 + omega_0**2)))/ (4 * np.pi)
    scales = time_scales
    #calculate angular frequencies
    angular_freq = ((2*np.pi * np.arange(signal.shape[1])) / (signal.shape[1]* dt))
    # reflect the second half of the array
    angular_freq[(signal.shape[1] //2):] *= -1
    #H(w)
    heavy_step = angular_freq > 0
 
    #compute wavelet point 
    wavelet = np.zeros((signal.shape[1], scales.shape[0]), dtype=np.complex128)
    wavelet[heavy_step,:]= np.sqrt((scales *2 * np.pi)/ dt) * (np.pi ** -.25) * np.exp(-((np.outer(angular_freq[heavy_step], scales) - omega_0)**2) / 2)

    #zero mean 
    signal -= np.mean(signal)
    
    
    fft_signal = np.fft.fft(signal)
    
    wavelet = wavelet[:, np.newaxis, :]
    signal = signal[np.newaxis, :,:]
    
    multplied_start = time.time()
    multiplied = wavelet.T * fft_signal
    multi_end = time.time()
    print("multiplication took", multplied_start -multi_end)
    print("shape after multiplication in space", multiplied.shape)
    
    
    result = np.array((wavelet.shape[1], fft_signal.shape[0]), dtype=np.complex128)
    inverse_start = time.time()
    result = np.fft.ifft(multiplied, axis=2)
    inverse_end = time.time()
    print("invers fft took", inverse_end - inverse_start)
    return result
    


def transform_window(data,n_channels, samples_per_second, samples_per_subSample, space_log, time_scales, freq_min=0.2, freq_max=24.0, w0=8, n_features=60
                     ,start_window=250, end_window=7750):
    """transform a window of data from stanford array 

    Args:
        data (np.array(286, 15000)): window size should be 5 min/ 15000 samples 

    Returns:
        cwt transformed data : np.array(286, 300,60) only samples from 250:7750 are used so result is 2.5 minutes long
    """
    n_samples=data.shape[1]
    delta = 1 /samples_per_second
    
    #take time derivative 
    data_derivative = data[:, 1:] - data[:, :-1]
    if(gpu):
      data_derivative = np.asnumpy(data_derivative)
 
    """
    nyquist = 0.5 * samples_per_second
    low = freq_min / nyquist
    high = freq_max / nyquist
    b, a = butter(4, Wn=[low, high], btype="bandpass")
    """

    # perform bandpass filter to remove high and low frequencies
    # TODO can this be done faster 
    filter_start = time.time()
    for channel in range(n_channels):
        trace = Trace(data=data_derivative[channel, :], header={'delta':1.0/float(samples_per_second),'sampling_rate':samples_per_second})
        trace = trace.filter("bandpass", freqmin= freq_min, freqmax= freq_max, corners=4, zerophase=True)
        data_derivative[channel, :] = trace.data
        #data_derivative[channel, :] = filtfilt(b, a, data_derivative[channel, :])
    filter_end = time.time()
    print("filter took", filter_end - filter_start)
    
    if(gpu):
        data_derivative = np.array(data_derivative)

    #remove lazer drift/ remove median
    data_derivative = data_derivative - np.median(data_derivative, axis=0)


    transformed_data = np.empty((n_channels, n_samples-1, n_features), dtype=np.float64)
    #new_transformed_data = np.empty((n_channels, n_samples-1, 60), dtype=np.float64)
    
    #TODO how can I get rid of these loops 
    space_start = time.time()
    #for index, channel in enumerate(data_derivative.T):
        #transformed_data[:, index, 30:] = np.abs(cwt_space_vec(channel, space_log, w0, delta).T)
    transformed_data[:, : , n_features//2:] = np.abs(cwt_space_vec(data_derivative.T, space_log, w0, delta).T) 
    space_end = time.time()
    print("space cwt took", space_end - space_start)
    
    time_start = time.time()
    #for index, sample in enumerate(data_derivative):
    #    transformed_data[index, : , :30] = np.abs(cwt_space_vec(sample, time_log, w0, delta).T) 
    transformed_data[:, : , :n_features//2] = np.abs(cwt_time_vec(data_derivative, time_scales, w0, delta).T) 
    time_end = time.time()
    print("time cwt took", time_end - time_start)
    
   

    #average samples over 0.5 second intervals 

    #start is at 250 since stanford windows start at 54.9 so add 5 seconds to start at the top of 
    
    reshaped_data = np.reshape(transformed_data[:, start_window: end_window, :], (n_channels, 300, samples_per_subSample, n_features))

    averaged_data = np.mean(reshaped_data, axis=2)
    return averaged_data


