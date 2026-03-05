import numpy as np
import pyreaper
from scipy import signal

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
    pm_times, pm, _, _, _ = pyreaper.reaper(x_int16, fs)
    
    marks = (pm_times * fs).astype(int)
    marks = marks[pm == 1]
    
    output = np.zeros(int(len(x) / k) + fs)
    last_idx = 0
    
    for i in range(1, len(marks) - 1):
        T = marks[i] - marks[i-1]
        start, end = marks[i] - T, marks[i] + T
        if start < 0 or end > len(x): continue
        
        segment = x[start:end].copy()
        window = signal.windows.triang(len(segment))
        segment *= window
        
        new_pos = int(last_idx)
        if new_pos + len(segment) < len(output):
            output[new_pos:new_pos + len(segment)] += segment
        
        last_idx += T / k
            
    return output / np.max(np.abs(output))