import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import sys

def shift(x, fs, dt, at, f):
    t = np.arange(len(x)) / fs
    
    t_new = t + dt + at * np.sin(2 * np.pi * f * t)
    
    if not np.all(np.diff(t_new) > 0):
        print(f"Предупреждение: не строгое возрастание для dt={dt}, at={at}, f={f}")
        
    x_shifted = np.interp(t, t_new, x, left=0, right=0)
    return x_shifted

def apply_chorus(input_file, output_file):
    fs, x = wavfile.read(input_file)
    
    x_float = x.astype(np.float64)
    if x.dtype == np.int16:
        x_float /= 32768.0
    elif x.dtype == np.uint8:
        x_float = (x_float - 128.0) / 128.0
        
    copies = [
        {'dt': 0.020, 'at': 0.010, 'f': 3.0},
        {'dt': 0.025, 'at': 0.007, 'f': 2.7},
        {'dt': 0.015, 'at': 0.012, 'f': 3.2}
    ]
    
    mono = True
    if x_float.ndim > 1:
        mono = False
        channels = x_float.shape[1]
    else:
        channels = 1
        x_float = x_float[:, np.newaxis]
        
    y = np.zeros_like(x_float)
    
    for c in range(channels):
        x_channel = x_float[:, c]
        y_channel = np.copy(x_channel)
        
        for params in copies:
            shifted = shift(x_channel, fs, params['dt'], params['at'], params['f'])
            y_channel += shifted
            
        y[:, c] = y_channel
        
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y *= 0.9 / max_val
        
    y_int16 = np.int16(y * 32767)
    
    if mono:
        y_int16 = y_int16.flatten()
        
    wavfile.write(output_file, fs, y_int16)
    print(f"Эффект хоруса успешно применен. Оригинал добавлен с {len(copies)} задержанными копиями. Результат сохранен как {output_file}.")

if __name__ == "__main__":
    apply_chorus("voice/input.wav", "voice/output.wav")

    n1 = np.arange(-5, 10)
    x_n1 = np.where((n1 >= 0) & (n1 < 4), 1.0, 0.0)

    t1 = np.linspace(-5, 10, 1000)

    x_linear1 = np.interp(t1, n1, x_n1)

    x_shannon1 = np.zeros_like(t1)
    for i, n_val in enumerate(n1):
        x_shannon1 += x_n1[i] * np.sinc(t1 - n_val)

    fig1, (ax1_5, ax2_5) = plt.subplots(2, 1, figsize=(10, 8))

    ax1_5.stem(n1, x_n1, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax1_5.plot(t1, x_linear1, 'g-', label='Линейная (numpy.interp)')
    ax1_5.set_title('Задание 5: Линейная интерполяция импульса')
    ax1_5.grid(True)
    ax1_5.legend()

    ax2_5.stem(n1, x_n1, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax2_5.plot(t1, x_shannon1, 'b-', label='Идеальное восстановление (Уиттекер-Шеннон)')
    ax2_5.set_title('Задание 5: Идеальное восстановление импульса')
    ax2_5.grid(True)
    ax2_5.legend()

    plt.tight_layout()
    plt.show()

    n2 = np.arange(-5, 20)
    x_n2 = np.zeros_like(n2, dtype=float)

    for i, val in enumerate(n2):
        if 0 <= val < 5:
            x_n2[i] = 1
        elif 5 <= val < 10:
            x_n2[i] = val - 5
        elif 10 <= val < 15:
            x_n2[i] = val - 10
        else:
            x_n2[i] = 0

    t2 = np.linspace(-5, 20, 1500)

    x_linear = np.interp(t2, n2, x_n2)

    x_shannon2 = np.zeros_like(t2)
    for i, val in enumerate(n2):
        x_shannon2 += x_n2[i] * np.sinc(t2 - val)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.stem(n2, x_n2, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax1.plot(t2, x_linear, 'g-', label='Линейная (numpy.interp)')
    ax1.set_title('Задание 6: Линейная интерполяция')
    ax1.grid(True)
    ax1.legend()

    ax2.stem(n2, x_n2, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax2.plot(t2, x_shannon2, 'b-', label='Восстановление sinc-функциями')
    ax2.set_title('Задание 6: Идеальное восстановление (Уиттекер-Шеннон)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()