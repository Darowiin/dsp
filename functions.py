import numpy as np
import pyreaper
import scipy.signal.windows as windows

def my_acf(x, m):
    N = len(x)
    mu = np.mean(x)
    if m >= N: return 0
    
    x_centered = x - mu
    sum_val = np.sum(x_centered[:N-m] * x_centered[m:])
    return sum_val / (N - m)

def my_dtft(x, fs, f):
    x = np.array(x)
    n = np.arange(len(x))
    
    def calc_point(freq):
        omega = 2 * np.pi * freq / fs
        kernel = np.exp(-1j * omega * n)
        return np.abs(np.dot(x, kernel))

    if np.isscalar(f):
        return calc_point(f)
    else:
        return np.array([calc_point(freq) for freq in f])

def psola(x, fs, k):
    x_int16 = (x * 32767).astype(np.int16)
    pm_times, _, _, _, _ = pyreaper.reaper(x_int16, fs)
    pm_samples = (pm_times * fs).astype(int)

    new_len = int(len(x) / k)
    y = np.zeros(new_len)

    for i in range(1, len(pm_samples) - 1):
        center = pm_samples[i]

        T_prev = center - pm_samples[i-1]
        T_next = pm_samples[i+1] - center
        T_samples = (T_prev + T_next) // 2

        half_len = T_samples
        start_idx = center - half_len
        end_idx = center + half_len

        if start_idx < 0 or end_idx >= len(x):
            continue

        segment = x[start_idx:end_idx].copy()

        window = windows.triang(len(segment))
        windowed_segment = segment * window

        new_center = int(center / k)
        new_start = new_center - half_len
        new_end = new_center + half_len

        if new_start < 0 or new_end > new_len:
            continue

        act_len = len(windowed_segment)
        if new_start + act_len <= new_len:
            y[new_start:new_start+act_len] += windowed_segment
        else:
            act_len = new_len - new_start
            y[new_start:new_start+act_len] += windowed_segment[:act_len]

    return y