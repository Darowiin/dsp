import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve, deconvolve
import itertools
from morse_encode import morse_encode, MORSE_CODES

def build_lowpass_filter(w0, N=51):
    h_lp = np.zeros(N)
    center = N // 2
    for i in range(N):
        n = i - center
        if n == 0:
            h_lp[i] = w0 / np.pi
        else:
            h_lp[i] = np.sin(w0 * n) / (np.pi * n)
    return h_lp

def main():
    # === 1. ЗАГРУЗКА ДАННЫХ ===
    print("Загрузка данных...")
    data = np.load("data/6412-27.npy")
    y = np.ravel(data[0, :])
    v = np.ravel(data[1, :])
    h_noisy_samples = data[2:, :]

    # === 2. ОЦЕНКА ИМПУЛЬСНОЙ ХАРАКТЕРИСТИКИ ===
    h_estimated = np.mean(h_noisy_samples, axis=0)[:200]  # Берем первые 200 отсчетов
    threshold_h = 0.02 * np.max(np.abs(h_estimated))
    h_cleaned = np.where(np.abs(h_estimated) < threshold_h, 0.0, h_estimated)
    h_final = np.trim_zeros(h_cleaned, trim='fb')

    # === 3. ОПРЕДЕЛЕНИЕ ПАРАМЕТРОВ МОРЗЕ ===
    chunk_size = min(50000, len(y))
    y_chunk = y[:chunk_size]
    y_zero_mean = y_chunk - np.mean(y_chunk)
    spectrum_y = np.abs(np.fft.rfft(y_zero_mean))
    frequencies = np.fft.rfftfreq(len(y_zero_mean), d=1.0)
    
    peak_idx = np.argmax(spectrum_y[1:]) + 1
    f_peak = frequencies[peak_idx]
    
    M = round(1.0 / (2.0 * f_peak))
    w0 = np.pi / M
    h_lp = build_lowpass_filter(w0, N=51)

    # === 4. ФИЛЬТРАЦИЯ И ДЕКОНВОЛЮЦИЯ ВСЕГО СИГНАЛА ===
    print("Фильтрация и развёртка ВСЕГО сигнала...")
    y_filtered = convolve(y, h_lp, mode='full')
    x_hat, _ = deconvolve(y_filtered, h_final)

    # === 5. АВТОМАТИЧЕСКОЕ ДЕКОДИРОВАНИЕ МОРЗЕ ===
    # Применяем пороговую обработку ко всей длине
    x_hat_thresholded = np.where(x_hat >= 0.5, 1.0, 0.0)

    # Группируем непрерывные блоки нулей и единиц
    groups = [(key, len(list(group))) for key, group in itertools.groupby(x_hat_thresholded)]

    # Отрезаем длинную тишину в самом начале и конце файла, если она есть
    if groups and groups[0][0] == 0 and groups[0][1] > 5 * M:
        groups.pop(0)
    if groups and groups[-1][0] == 0 and groups[-1][1] > 5 * M:
        groups.pop()

    morse_elements = []
    for val, length in groups:
        if val == 1:
            if length < 2 * M:
                morse_elements.append('.')
            else:
                morse_elements.append('-')
        else:
            if length > 5 * M:
                morse_elements.append(' / ')  # Разделитель слов (7 у.е.)
            elif length > 2 * M:
                morse_elements.append(' ')    # Разделитель букв (3 у.е.)

    morse_str = "".join(morse_elements).strip()
    
    # Словарик для обратного декодирования
    REVERSE_MORSE = {v: k for k, v in MORSE_CODES.items() if v not in [" ", "   ", "       "]}
    
    decoded_words = []
    for word in morse_str.split(' / '):
        decoded_word = "".join([REVERSE_MORSE.get(letter, '?') for letter in word.split(' ') if letter])
        decoded_words.append(decoded_word)
    
    final_sentence = " ".join(decoded_words).upper()

    # === 6. ГЕНЕРАЦИЯ ИДЕАЛА И ВЫЧИСЛЕНИЕ ЧЕСТНОГО MSE ===
    x_ideal = morse_encode(final_sentence.lower(), unit_size=M)

    # Линейно масштабируем амплитуду x_hat для честного сравнения
    x_min, x_max = np.min(x_hat), np.max(x_hat)
    x_hat_scaled = (x_hat - x_min) / (x_max - x_min) if (x_max - x_min) > 0 else x_hat

    # Ищем идеальное совмещение по минимальному MSE (сканируем область задержки)
    best_mse = float('inf')
    best_delay = 0
    for test_delay in range(0, 150):
        x_hat_trimmed = x_hat_scaled[test_delay : test_delay + len(x_ideal)]
        if len(x_hat_trimmed) == len(x_ideal):
            current_mse = np.mean((x_ideal - x_hat_trimmed) ** 2)
            if current_mse < best_mse:
                best_mse = current_mse
                best_delay = test_delay

    x_hat_final_aligned = x_hat_scaled[best_delay : best_delay + len(x_ideal)]
    x_hat_thresholded_final = np.where(x_hat_final_aligned >= 0.5, 1.0, 0.0)

    # ========================================
    # === ВЫВОД ДАННЫХ В КОНСОЛЬ ДЛЯ ОТЧЕТА ===
    # ========================================
    print("\n" + "="*40)
    print("=== ИТОГОВЫЕ ДАННЫЕ ДЛЯ ТВОЕГО ОТЧЕТА ===")
    print("="*40)
    print(f"1. Размер одной точки (M): {M} отсчётов [cite: 367, 391]")
    print(f"2. Частота среза ФНЧ (w0): {w0:.4f} рад/отсчёт [cite: 369, 391]")
    print(f"3. РАСШИФРОВАННОЕ ПРЕДЛОЖЕНИЕ: {final_sentence} [cite: 344, 418]")
    print(f"4. Строка Морзе: {morse_str}")
    print(f"5. Оптимальный временной сдвиг: {best_delay} отсчётов [cite: 389, 458]")
    print(f"6. Минимальная ошибка (MSE_ФНЧ): {best_mse:.5f} [cite: 344, 469]")
    print("="*40)

    print("\nОтображение графиков... Закрой окно с графиками, чтобы завершить программу.")
    
    plt.figure(figsize=(12, 8))

    # Верхний график: Сравнение восстановленной волны (до порога) и идеала
    plt.subplot(2, 1, 1)
    plt.plot(x_ideal, color='blue', alpha=0.5, linewidth=2, label="Идеальный сигнал x[n]")
    plt.plot(x_hat_final_aligned, color='orange', label="Восстановленная оценка x_hat[n]")
    plt.title("Восстановленный сигнал после ФНЧ и развёртки (Выровненный)")
    plt.grid(True)
    plt.legend()

    # Нижний график: Пороговый сигнал (бинарный 0 или 1) готовый к чтению
    plt.subplot(2, 1, 2)
    plt.plot(x_ideal, color='blue', alpha=0.5, linewidth=2, label="Идеальный сигнал x[n]")
    plt.plot(x_hat_thresholded_final, color='green', label="Пороговый сигнал (>= 0.5)")
    plt.title("Сигнал, готовый к декодированию Морзе")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()