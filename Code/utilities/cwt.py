import time
import os
import torch
import numpy as np
from obspy.core.trace import Trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cwt_time_vec(signal, scales, omega_0, dt):
    signal = torch.tensor(signal, dtype=torch.float32, device=device)
    print(f'signal shape is {signal.shape}')
    scales = torch.tensor(scales, dtype=torch.float32, device=device)
    print(f'scales shape is {scales.shape}')
    #scales = (log_space * (omega_0 + np.sqrt(2 + omega_0 ** 2))) / (4 * np.pi)
    angular_freq = (2 * np.pi * torch.arange(signal.shape[1], device=device)) / (signal.shape[1] * dt)
    angular_freq[(signal.shape[1] // 2):] *= -1
    heavy_step = angular_freq > 0

    wavelet = torch.zeros((signal.shape[1], scales.shape[0]), dtype=torch.complex128, device=device)
    wavelet[heavy_step, :] = (torch.sqrt((scales * 2 * np.pi) / dt) * (np.pi ** -.25) * \
                             torch.exp(-((torch.outer(angular_freq[heavy_step], scales) - omega_0) ** 2) / 2)).type(torch.complex128)

    signal -= torch.mean(signal, dim=1, keepdim=True)
    fft_signal = torch.fft.fft(signal, dim=1)
    print(fft_signal.shape)
    
    print(f'wavelet shape is {wavelet.shape}')
    print(f'fft shape is {fft_signal.shape}')

    wavelet = wavelet[:, None, :]
    multiplied = wavelet.T * fft_signal[None,:, :]
    
   
    multiplied = multiplied.permute(0, 2, 1)

    result = torch.fft.ifft(multiplied, dim=1)
    return result

def cwt_space_vec(signal, log_space, omega_0, dt):
    signal = torch.tensor(signal, dtype=torch.float32, device=device)
    log_space = torch.tensor(log_space, dtype=torch.float32, device=device)

    scales = (log_space * (omega_0 + np.sqrt(2 + omega_0 ** 2))) / (4 * np.pi)
    angular_freq = (2 * np.pi * torch.arange(signal.shape[1], device=device)) / (signal.shape[1] * dt)
    angular_freq[(signal.shape[1] // 2):] *= -1
    heavy_step = angular_freq > 0

    wavelet = torch.zeros((signal.shape[1], scales.shape[0]), dtype=torch.complex128, device=device)
    wavelet[heavy_step, :] = (torch.sqrt((scales * 2 * np.pi) / dt) * (np.pi ** -.25) * \
                             torch.exp(-((torch.outer(angular_freq[heavy_step], scales) - omega_0) ** 2) / 2)).type(torch.complex128)

    signal -= torch.mean(signal, dim=1, keepdim=True)
    fft_signal = torch.fft.fft(signal, dim=1)

    wavelet = wavelet[:, None, :]
    print(wavelet.shape)
    multiplied = wavelet.T * fft_signal[None,:,:]

    result = torch.fft.ifft(multiplied, dim=2)
    return result

def transform_window(data, n_channels, samples_per_second, samples_per_subSample, space_log, time_scales, freq_min=0.2, freq_max=24.0, w0=8, start_window=250, end_window=7750, window_length=300, subsampling = True):
    n_samples = data.shape[1]
    delta = 1 / samples_per_second
    n_features = len(space_log) + len(time_scales)

    data_derivative = torch.tensor(data[:, 1:] - data[:, :-1], dtype=torch.float32, device=device)

    filter_start = time.time()
    for channel in range(n_channels):
        trace = Trace(data=data_derivative[channel, :].cpu().numpy(), header={'delta': 1.0 / float(samples_per_second), 'sampling_rate': samples_per_second})
        trace = trace.filter("bandpass", freqmin=freq_min, freqmax=freq_max, corners=4, zerophase=True)
        data_derivative[channel, :] = torch.tensor(trace.data, dtype=torch.float32, device=device)
    filter_end = time.time()
    print("filter took", filter_end - filter_start)

    data_derivative = data_derivative - torch.median(data_derivative, dim=0).values

    transformed_data = torch.empty((n_channels, n_samples - 1, n_features), dtype=torch.float32, device=device)

    space_start = time.time()
    transformed_data[:, :, len(time_scales):] = cwt_space_vec(data_derivative.T.cpu().numpy(), space_log, w0, delta).T.real
    space_end = time.time()
    print("space cwt took", space_end - space_start)

    time_start = time.time()
    transformed_data[:, :, :len(time_scales)] = cwt_time_vec(data_derivative.cpu().numpy(), time_scales, w0, delta).T.real
    time_end = time.time()
    print("time cwt took", time_end - time_start)
    
    if(subsampling):
        reshaped_data = transformed_data[:, start_window:end_window, :].reshape(n_channels,  window_length, samples_per_subSample, n_features)
        averaged_data = torch.mean(reshaped_data, dim=2)
    else:
        averaged_data = transformed_data
    return averaged_data

def inverse_cwt(transform, scales, dj, dt, w0):
    if w0 != 8:
        print("err only w0 = 8 is implemented")

    scales_graph = torch.tensor(scales, dtype=torch.float32, device=device).unsqueeze(0)
    inverse = transform / torch.sqrt(scales_graph.T)
    inverse = torch.sum(inverse, dim=0).squeeze()

    colorado_factor = (dj * torch.sqrt(torch.tensor(dt, dtype=torch.float32, device=device))) / (0.7511 * 0.5758)
    inverse_w_factor = colorado_factor * inverse
    return inverse_w_factor.cpu().numpy()