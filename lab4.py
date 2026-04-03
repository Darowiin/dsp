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
    m = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * m * n / N)
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

    fs, x = wavfile.read('voice/6412-27.wav')
    if len(x.shape) > 1:
        x = x[:, 0]

    nperseg = 4096
    noverlap = 4096 - 512
    f_stft, t_stft, spectrum = stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(spectrum) ** 2

    plt.figure('Spectrogram')
    plt.pcolormesh(t_stft, f_stft, power, shading='auto')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, 1500)
    plt.colorbar(label='Power')
    plt.savefig('spectrogram.png')
    plt.show()


if __name__ == "__main__":
    main()