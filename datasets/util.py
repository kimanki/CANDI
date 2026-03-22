import numpy as np


def downsample(data: np.ndarray, rate: int):
    assert data.ndim in (1, 2)
    
    if data.ndim == 1:
        data_downsampled = np.zeros(data.shape[0] // rate, dtype=data.dtype)
    elif data.ndim == 2:
        data_downsampled = np.zeros((data.shape[0] // rate, data.shape[1]), dtype=data.dtype)
    
    for i in range(len(data_downsampled)):
        data_downsampled[i] = np.median(data[rate*i:rate*(i+1)], axis=0)
    
    return data_downsampled
