import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import sys

def shift(x, fs, dt, at, f):
    """
    Создание задержанной копии сигнала (сдвиг).
    x - входной сигнал
    fs - частота дискретизации
    dt - постоянная часть задержки (в секундах)
    at - амплитуда переменной части задержки (в секундах)
    f - частота изменения переменной части задержки (в герцах)
    """
    t = np.arange(len(x)) / fs
    
    # Вычисляем задержку для каждого момента времени t
    t_new = t + dt + at * np.sin(2 * np.pi * f * t)
    
    # Проверка на строгое возрастание значений t_new
    if not np.all(np.diff(t_new) > 0):
        print(f"Предупреждение: не строгое возрастание для dt={dt}, at={at}, f={f}")
        
    # Интерполяция
    x_shifted = np.interp(t, t_new, x, left=0, right=0)
    return x_shifted

def apply_chorus(input_file, output_file):
    """
    Применяет эффект хоруса к аудиофайлу и сохраняет результат в новый файл.
    
    Читает входной аудиофайл, создает несколько его задержанных копий с 
    меняющимся во времени временем задержки (используя синусоидальный закон),
    суммирует их с оригинальным сигналом, нормализует амплитуду (чтобы
    предотвратить искажения) и сохраняет полученный результат.
    
    Параметры:
    input_file (str): Путь к входному wav-файлу.
    output_file (str): Путь для вывода (сохранения) итогового wav-файла.
    """
    # Чтение входного файла
    fs, x = wavfile.read(input_file)
    
    # Преобразуем в float64 для безопасных вычислений и отсутствия клиппинга на этапах сложения
    x_float = x.astype(np.float64)
    if x.dtype == np.int16:
        x_float /= 32768.0
    elif x.dtype == np.uint8:
        x_float = (x_float - 128.0) / 128.0
        
    # Параметры задержки для копий (отталкиваясь от стартовых: 20мс, 10мс, 3Гц)
    copies = [
        {'dt': 0.020, 'at': 0.010, 'f': 3.0},   # 1-я копия
        {'dt': 0.025, 'at': 0.007, 'f': 2.7},   # 2-я копия (чуть бОльшая задержка, меньшая амплитуда)
        {'dt': 0.015, 'at': 0.012, 'f': 3.2}    # 3-я копия (чуть меньшая задержка, бОльшая амплитуда)
    ]
    
    # Учитываем возможную многоканальность (стерео)
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
        y_channel = np.copy(x_channel) # Начинаем с оригинального сигнала
        
        # Накладываем задержанные копии
        for params in copies:
            shifted = shift(x_channel, fs, params['dt'], params['at'], params['f'])
            y_channel += shifted
            
        y[:, c] = y_channel
        
    # Нормализация амплитуды выходного сигнала для предотвращения искажений
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y *= 0.9 / max_val
        
    # Возвращаем в стандартный 16-битный формат PCM
    y_int16 = np.int16(y * 32767)
    
    if mono:
        y_int16 = y_int16.flatten()
        
    # Запись в файл
    wavfile.write(output_file, fs, y_int16)
    print(f"Эффект хоруса успешно применен. Оригинал добавлен с {len(copies)} задержанными копиями. Результат сохранен как {output_file}.")

if __name__ == "__main__":
    apply_chorus("voice/input.wav", "voice/output.wav")

    # === Задание 5: Восстановление прямоугольного импульса ===
    # Исходный дискретный сигнал (4 единичных отсчета)
    n1 = np.arange(-5, 10)
    x_n1 = np.where((n1 >= 0) & (n1 < 4), 1.0, 0.0)

    # Создаем "непрерывное" время (мелкий шаг дискретизации)
    t1 = np.linspace(-5, 10, 1000)

    # 1. Упрощение с линейной интерполяцией
    x_linear1 = np.interp(t1, n1, x_n1)

    # 2. Восстановление по формуле Уиттекера-Шеннона
    x_shannon1 = np.zeros_like(t1)
    for i, n_val in enumerate(n1):
        x_shannon1 += x_n1[i] * np.sinc(t1 - n_val)

    # Отрисовка двух методов восстановления
    fig1, (ax1_5, ax2_5) = plt.subplots(2, 1, figsize=(10, 8))

    # График 1: Линейная интерполяция
    ax1_5.stem(n1, x_n1, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax1_5.plot(t1, x_linear1, 'g-', label='Линейная (numpy.interp)')
    ax1_5.set_title('Задание 5: Линейная интерполяция импульса')
    ax1_5.grid(True)
    ax1_5.legend()

    # График 2: Формула Уиттекера-Шеннона
    ax2_5.stem(n1, x_n1, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax2_5.plot(t1, x_shannon1, 'b-', label='Идеальное восстановление (Уиттекер-Шеннон)')
    ax2_5.set_title('Задание 5: Идеальное восстановление импульса')
    ax2_5.grid(True)
    ax2_5.legend()

    plt.tight_layout()
    plt.show()

    # === Задание 6: Восстановление кастомного дискретного сигнала ===
    
    # 1. Формирование дискретного сигнала по условиям
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

    # 2. Мелкая сетка "непрерывного" времени
    t2 = np.linspace(-5, 20, 1500)

    # 3. Линейная интерполяция
    x_linear = np.interp(t2, n2, x_n2)

    # 4. Восстановление (Уиттекер-Шеннон)
    x_shannon2 = np.zeros_like(t2)
    for i, val in enumerate(n2):
        x_shannon2 += x_n2[i] * np.sinc(t2 - val)

    # 5. Отрисовка двух графиков
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # График 1: Линейная интерполяция
    ax1.stem(n2, x_n2, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax1.plot(t2, x_linear, 'g-', label='Линейная (numpy.interp)')
    ax1.set_title('Линейная интерполяция')
    ax1.grid(True)
    ax1.legend()

    # График 2: Формула Уиттекера-Шеннона
    ax2.stem(n2, x_n2, linefmt='k-', markerfmt='ko', basefmt='k-', label='Дискретный x[n]')
    ax2.plot(t2, x_shannon2, 'b-', label='Восстановление sinc-функциями')
    ax2.set_title('Идеальное восстановление (Уиттекер-Шеннон)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()