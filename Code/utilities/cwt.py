import time
import os
import torch
import pickle
import numpy as np
from obspy.core.trace import Trace

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

def get_scales(dt,dj, w0, n_samples):
    s0 = (2*dt* (w0 + np.sqrt(2 + w0))) / (4 *np.pi)
    J = np.ceil((1/dj )* np.log2((n_samples) / s0))
    scales = s0 * 2**(np.arange(J) *dj)
    return scales



def cwt_time_vec(signal, scales, omega_0, dt):
    signal = torch.tensor(signal, dtype=torch.float32, device=device)
    scales = torch.tensor(scales, dtype=torch.float32, device=device)
    
    #scales = (log_space * (omega_0 + np.sqrt(2 + omega_0 ** 2))) / (4 * np.pi)
    angular_frequencies = (2 * np.pi * torch.arange(signal.shape[1], device=device)) / (signal.shape[1] * dt)
    angular_frequencies[(signal.shape[1] // 2):] *= -1
   
    heavy_step = angular_frequencies > 0

    wavelet = torch.zeros((signal.shape[1], scales.shape[0]), dtype=torch.complex128, device=device)
    wavelet[heavy_step, :] = (torch.sqrt((scales * 2 * np.pi) / dt) * (np.pi ** -.25) * \
                             torch.exp(-((torch.outer(angular_frequencies[heavy_step], scales) - omega_0) ** 2) / 2)).type(torch.complex128)

    signal -= torch.mean(signal, dim=1, keepdim=True)
    fft_signal = torch.fft.fft(signal, dim=1)
    
    wavelet = wavelet[:, None, :]
    multiplied = wavelet.T * fft_signal[None,:, :]
    multiplied = multiplied.permute(0, 2, 1)

    result = torch.fft.ifft(multiplied, dim=1)
    return result

def cwt_space_vec(signal, log_space, omega_0, dt):
    signal = torch.tensor(signal, dtype=torch.float32, device=device)
    log_space = torch.tensor(log_space, dtype=torch.float32, device=device)

    scales = (log_space * (omega_0 + np.sqrt(2 + omega_0 ** 2))) / (4 * np.pi)
    
    angular_frequencies = (2 * np.pi * torch.arange(signal.shape[1], device=device)) / (signal.shape[1] * dt)
    angular_frequencies[(signal.shape[1] // 2):] *= -1
    heavy_step = angular_frequencies > 0

    wavelet = torch.zeros((signal.shape[1], scales.shape[0]), dtype=torch.complex128, device=device)
    wavelet[heavy_step, :] = (torch.sqrt((scales * 2 * np.pi) / dt) * (np.pi ** -.25) * \
                             torch.exp(-((torch.outer(angular_frequencies[heavy_step], scales) - omega_0) ** 2) / 2)).type(torch.complex128)

    signal -= torch.mean(signal, dim=1, keepdim=True)
    fft_signal = torch.fft.fft(signal, dim=1)

    wavelet = wavelet[:, None, :]
    multiplied = wavelet.permute(2,1,0) * fft_signal[None,:,:]

    result = torch.fft.ifft(multiplied, dim=2)
    return result

def transform_window(data, n_channels, samples_per_second, samples_per_subSample, space_log, time_scales, freq_min=0.2, freq_max=24.0, w0=8, start_window=250, end_window=7750, window_length=300, subsampling = True, derivative = True, space_dt=False):
    print(device)
    n_samples = data.shape[1]
    delta = 1 / samples_per_second
    n_features = len(space_log) + len(time_scales)

    if(derivative):
        data_derivative = torch.tensor(data[:, 1:] - data[:, :-1], dtype=torch.float32, device=device)
    else:
        #add one since we no longer take the derivative 
        n_samples +=1
        data_derivative = torch.tensor(data, dtype=torch.float32)
    if(space_dt):
        data_derivative = data_derivative[1:,:]- data_derivative[:-1,:]
        n_channels=data_derivative.shape[0]

    for channel in range(n_channels):
        trace = Trace(data=data_derivative[channel, :].cpu().numpy(), header={'delta': 1.0 / float(samples_per_second), 'sampling_rate': samples_per_second})
        trace = trace.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)
        data_derivative[channel, :] = torch.tensor(trace.data, dtype=torch.float32, device=device)

    data_derivative = data_derivative - torch.median(data_derivative, dim=0).values

    transformed_data = torch.empty((n_channels, n_samples - 1, n_features), dtype=torch.float32, device=device)

    # add in torch.abs() and remove .real to recreate paper implementation
    transformed_data[:, :, len(time_scales):] = cwt_space_vec(data_derivative.T.cpu().numpy(), space_log, w0, delta).T.real

    transformed_data[:, :, :len(time_scales)] = cwt_time_vec(data_derivative.cpu().numpy(), time_scales, w0, delta).T.real
    
    print(transformed_data.shape)
    if(subsampling):
        reshaped_data = transformed_data[:, start_window:end_window, :].reshape(n_channels,  window_length, samples_per_subSample, n_features)
        averaged_data = torch.mean(reshaped_data, dim=2)
    else:
        averaged_data = transformed_data
    return averaged_data

def inverse_cwt(transform, scales, dj, dt, w0):
    #shape of transform should be scales, samples 
    if w0 != 8:
        print("err only w0 = 8 is implemented")
    

    scales_graph = torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(0)
    transform = torch.tensor(transform, dtype=torch.float32, device=device)
    inverse = transform / torch.sqrt(scales_graph.T) + 10e-10 
    inverse = torch.sum(inverse, dim=0).squeeze()

    colorado_factor = (dj * torch.sqrt(torch.tensor(dt, dtype=torch.float32, device=device))) / (0.7511 * 0.5758)
    inverse_w_factor = colorado_factor * inverse
    return inverse_w_factor.cpu().numpy()

def inverse_DAS(transform, scales ,dj, dt, w0):
    inverse = np.zeros((transform.shape[0], transform.shape[1]))
    for i in range(transform.shape[0]):
        inverse[i,:] = inverse_cwt(transform[i,:,:len(scales)].T, scales, dj, dt, w0)
    return inverse
        

def mute(transform, scales, mute_mask, scales_to_mute, dj, dt, w0, mute_level = 0.0001):
    """
    inputs 
    transform: the cwt transform that will be muted shape is (channels, samples, scales)
    scales: the scales used for the cwt 
    mute_mask: locations in the DAS that where itentified as noise and should be muted
    scales_to_mute: the scales that will be set to mute level and muted out
    dj: dj param used for invese cwt
    dt: dt param used for inverse cwt 
    w0: w0 param used for inverse cwt
    """
    transform = np.copy(transform)
    muted_inverse = np.empty((transform.shape[0], transform.shape[1]))

    mute = np.ones(scales.shape)
    mute[np.ma.make_mask(scales_to_mute)] = mute_level
 
    transform[mute_mask,:] *= mute.astype(np.float32)

    for i in range(transform.shape[0]):
        muted_inverse[i, :] = inverse_cwt(transform[i].T ,scales, dj, dt, w0)
    return muted_inverse

def save_cwt_info(data_shape, samples_per_second, samples_per_subSample, space_log, time_scales, freq_min, freq_max, w0, start_window, end_window, window_length, subsampling, start_file, end_file, filename):

    cwt_info = {
        "data_shape": data_shape,
        "samples_per_second": samples_per_second,
        "samples_per_subSample": samples_per_subSample,
        "space_log": space_log,
        "time_scales": time_scales,
        "freq_min": freq_min,
        "freq_max": freq_max,
        "w0": w0,
        "start_window": start_window,
        "end_window": end_window,
        "window_length": window_length,
        "subsampling": subsampling,
        "start_file": start_file,
        "end_file": end_file
    }
    
    # Pickle the dictionary
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(cwt_info, f)
        f.close()
