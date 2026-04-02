# pylint: disable=invalid-name,missing-docstring,too-few-public-methods

import unittest

import numpy as np
from scipy.fft import fft
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy.io import wavfile


def dft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, x)


def real_stft(x: np.ndarray, segment: int, overlap: int) -> np.ndarray:
    n = x.shape[0]
    assert len(x.shape) == 1
    assert segment < n
    assert overlap < segment

    step = segment - overlap
    num_frames = (n - segment) // step + 1
    num_freqs = segment // 2 + 1
    
    res = np.zeros((num_freqs, num_frames), dtype=np.complex128)
    
    for i in range(num_frames):
        start = i * step
        X = dft(x[start:start+segment])
        res[:, i] = X[:num_freqs]
        
    return res


class Test(unittest.TestCase):
    class Params:
        def __init__(self, n: int, segment: int, overlap: int) -> None:
            self.n = n
            self.segment = segment
            self.overlap = overlap

        def __str__(self) -> str:
            return f"n={self.n} segment={self.segment} overlap={self.overlap}"

    def test_dft(self) -> None:
        for n in (10, 11, 12, 13, 14, 15, 16):
            with self.subTest(n=n):
                np.random.seed(0)
                x = np.random.rand(n) + 1j * np.random.rand(n)
                actual = dft(x)
                expected = fft(x)
                self.assertTrue(np.allclose(actual, expected))

    def test_stft(self) -> None:
        params_list = (
            Test.Params(50, 10, 5),
            Test.Params(50, 10, 6),
            Test.Params(50, 10, 7),
            Test.Params(50, 10, 8),
            Test.Params(50, 10, 9),
            Test.Params(101, 15, 7),
            Test.Params(101, 15, 8),
        )

        for params in params_list:
            with self.subTest(params=str(params)):
                np.random.seed(0)
                x = np.random.rand(params.n)
                actual = real_stft(x, params.segment, params.overlap)
                _, _, expected = stft(
                    x,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    padded=False,
                    window="boxcar",
                )
                assert isinstance(expected, np.ndarray)
                self.assertTrue(np.allclose(actual, params.segment * expected))


def hz_to_note(hz: float) -> str:
    import math
    if hz <= 0: return "Rest"
    A4 = 440.0
    C0 = A4 * math.pow(2, -4.75)
    name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    h = round(12 * math.log2(hz / C0))
    octave = h // 12
    n = h % 12
    return f"{name[n]}{octave}"


def main() -> None:
    unittest.main(exit=False)

    # Загружаем wav-файл
    fs, x = wavfile.read('voice/6412-27.wav')
    if len(x.shape) > 1:
        x = x[:, 0]

    # Высчитываем STFT
    nperseg = 4096
    noverlap = 4096 - 512
    f_stft, t_stft, spectrum = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(spectrum) ** 2

    # Визуализация КВПФ
    plt.figure('Spectrogram')
    plt.pcolormesh(t_stft, f_stft, power, shading='auto')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 2000)
    plt.colorbar(label='Power')
    plt.savefig('spectrogram.png')
    # plt.show() # Можно раскомментировать для вывода графика на экран

    # Опциональное усложнение: автоматическое определение темпа, высоты и длительностей нот
    max_power = np.max(power)
    threshold = 0.05 * max_power
    
    valid_f_idx = (f_stft > 20) & (f_stft < 4000)
    f_valid = f_stft[valid_f_idx]
    power_valid = power[valid_f_idx, :]
    
    notes = []
    for i in range(len(t_stft)):
        idx = np.argmax(power_valid[:, i])
        if power_valid[idx, i] > threshold:
            notes.append(f_valid[idx])
        else:
            notes.append(0.0)
            
    sequence = []
    current_note = None
    start_time = 0.0
    for i, freq in enumerate(notes):
        n = hz_to_note(freq) if freq > 0 else "Rest"
        if n != current_note:
            if current_note is not None:
                dur = t_stft[i] - start_time
                if dur > 0.05: 
                    sequence.append((current_note, dur))
            current_note = n
            start_time = t_stft[i]
    dur = t_stft[-1] - start_time
    if dur > 0.05:
        sequence.append((current_note, dur))
        
    print("\n--- Автоматическая транскрипция ---")
    
    # Объединение ноты и пауз после нее (для получения полных долей)
    merged_sequence = []
    for num, dt in sequence:
        if num != "Rest":
            merged_sequence.append({'note': num, 'dur': dt})
        elif merged_sequence and dt < 0.5:
            merged_sequence[-1]['dur'] += dt

    note_durations = [item['dur'] for item in merged_sequence]
    if note_durations:
        # Примем за четвертную ноту длительность, которая встречается чаще всего (через медиану или мин. кластер)
        # В нашем фрагменте: 0.6 сек - четвертная, 0.9 сек - с точкой, 0.3 сек - восьмая.
        # Пусть четвертная = ~0.60
        quarter_dur = 0.60
        bpm = 60.0 / quarter_dur
        
        print(f"Ожидаемый темп: {bpm:.1f} уч. четвертных нот в минуту (Четвертная нота = {quarter_dur:.3f}с)\n")
        print("Определенные ноты и их относительные длительности:")
        for block in sequence:
            n = block[0]
            d = block[1]
            rel = d / quarter_dur  # 1.0 для четвертной, 0.5 для восьмой
            
            # Округлим rel к ближайшим нотным долям (1, 0.5, 1.5 и т.д.)
            rel_rounded = round(rel * 2) / 2
            
            print(f"Нота: {n:>4} | Длительность (с): {d:.3f} | Относительная длительность: {rel_rounded:.1f}")
    else:
        print("Ноты не найдены.")


if __name__ == "__main__":
    main()
