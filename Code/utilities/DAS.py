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


def zero_time_lag_cross_cor(channel_a, channel_b):
    return np.sum(channel_a * channel_b)

def semblance(DAS_sample):
    sum = 0
    divisor = 0
    for i in range(DAS_sample.shape[0]):
        divisor += zero_time_lag_cross_cor(DAS_sample[i], DAS_sample[i])
        for j in range(DAS_sample.shape[0]):
            sum += zero_time_lag_cross_cor(DAS_sample[i], DAS_sample[j])
    DAS_sample.shape[0] * divisor
    return sum / divisor

def SNR_sem(DAS_sample):
    s = semblance(DAS_Stanford)

    print(f'semblance is {s}')

    snr = s / (1 -s)
    return snr



# Code from  https://github.com/sachalapins/DAS-N2N/blob/main/results.ipynb 
# Calculats the semblance correctly from this paper https://onlinelibrary.wiley.com/doi/10.1111/1365-2478.13178 
# Works on older versions of scipy
def correlate_func(x, idx, cc_thresh = 0.9):
    correlation = correlate(x[idx,:], x[(idx+1),:], mode="full")
    lags = np.arange(-(x[idx,:].size - 1), x[(idx+1),:].size)
    lag_idx = np.argmax(correlation)
    lag = lags[lag_idx]
    if lag > 0:
        if np.corrcoef(x[idx,lag:], x[(idx+1),:-lag], rowvar=False)[0,1] > cc_thresh:
            x = np.concatenate(
                [np.concatenate([x[:(idx+1),:], np.zeros((x[:(idx+1),:].shape[0], lag))], axis=1),
                 np.concatenate([np.zeros((x[(idx+1):,:].shape[0], lag)), x[(idx+1):,:]], axis=1)],
                axis=0)
    if lag < 0:
        if np.corrcoef(x[idx,:-lag], x[(idx+1),lag:], rowvar=False)[0,1] > cc_thresh:
            x = np.concatenate(
                [np.concatenate([np.zeros((x[:(idx+1),:].shape[0], abs(lag))), x[:(idx+1),:]], axis=1),
                 np.concatenate([x[(idx+1):,:], np.zeros((x[(idx+1):,:].shape[0], abs(lag)))], axis=1)],
                axis=0)

    return(x)

# Looks at all previous channels in window and shifts relative to channel with highest xcorr
def correlate_func2(x, idx, cc_thresh = 0.9):
    correlation = correlate(x[:idx,:], x[(idx+1):(idx+2),:], mode="full", method="direct")
    idx_max_xcorr = np.argmax(np.amax(correlation, axis=1))

    lags = np.arange(-(x[idx_max_xcorr,:].size - 1), x[(idx+1),:].size)
    lag_idx = np.argmax(correlation[idx_max_xcorr,:])
    lag = lags[lag_idx]

    if lag > 0:
        if np.corrcoef(x[idx_max_xcorr,lag:], x[(idx+1),:-lag], rowvar=False)[0,1] > cc_thresh:
            # Need to check difference in zeros at start
            idx_max_xcorr_start = np.amin(np.where(x[idx_max_xcorr,:] != 0))
            idx_plus1_start = np.amin(np.where(x[(idx+1),:] != 0))
            lag = lag - (idx_plus1_start - idx_max_xcorr_start)
            if lag > 0:
                # Concatenate zeros to shift signal
                x = np.concatenate(
                    [np.concatenate([x[:(idx+1),:], np.zeros((x[:(idx+1),:].shape[0], lag))], axis=1),
                     np.concatenate([np.zeros((x[(idx+1):,:].shape[0], lag)), x[(idx+1):,:]], axis=1)],
                    axis=0)
    if lag < 0:
        if np.corrcoef(x[idx,:-lag], x[(idx+1),lag:], rowvar=False)[0,1] > cc_thresh:
            # Need to check difference in zeros at start
            idx_max_xcorr_start = np.amin(np.where(x[idx_max_xcorr,:] != 0))
            idx_plus1_start = np.amin(np.where(x[(idx+1),:] != 0))
            lag = lag + (idx_plus1_start - idx_max_xcorr_start)
            if lag < 0:
                # Concatenate zeros to shift signal
                x = np.concatenate(
                    [np.concatenate([np.zeros((x[:(idx+1),:].shape[0], abs(lag))), x[:(idx+1),:]], axis=1),
                     np.concatenate([x[(idx+1):,:], np.zeros((x[(idx+1):,:].shape[0], abs(lag)))], axis=1)],
                    axis=0)

    return(x)


# This is equal to first part of Eq 7 in https://doi.org/10.1111/1365-2478.13178
def marfurt_semblance(region):
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape

    # Cross correlation and shift
    for i in range(ntraces-1):
        region = correlate_func(region, i, cc_thresh = 0.7)

    square_of_sums = np.sum(region, axis=0)**2
    sum_of_squares = np.sum(region**2, axis=0)
    sembl = square_of_sums.sum() / sum_of_squares.sum()
    return sembl / ntraces
