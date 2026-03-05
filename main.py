from matplotlib import pyplot as plt
import numpy as np
from scipy.io.wavfile import write

from functions import create_melody, tone, musical_tone
from waveform import Waveform

if __name__ == "__main__":
    fs = 44100

    print("Генерация чистого тона (440 Гц, 2 секунды)...")
    pure_tone_audio = tone(440, 2.0, Waveform.SIN, fs)
    write('lab1_pure_tone.wav', fs, pure_tone_audio)
    print("Файл lab1_pure_tone.wav успешно сгенерирован.")
    
    t_pure = np.linspace(0, 2.0, len(pure_tone_audio), endpoint=False)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_pure, pure_tone_audio)
    plt.xlim(0, 0.005)
    plt.title("Задание 3.1: Чистый тон (Форма волны на интервале 5 мс)")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.show()

    print("Генерация затухающего составного тона (440 Гц, 2 секунды)...")
    compound_tone_audio = musical_tone(440, 2.0, Waveform.SIN, fs, db=-20)
    write('lab1_musical_tone.wav', fs, compound_tone_audio)
    print("Файл lab1_musical_tone.wav успешно сгенерирован.")

    t_comp = np.linspace(0, 2.0, len(compound_tone_audio), endpoint=False)

    plt.figure(figsize=(10, 4))
    plt.plot(t_comp, compound_tone_audio)
    plt.title("Задание 3.2: Затухающий составной тон (Огибающая затухания, 2 сек)")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.show()

    melody_text = """
     A#2 A3 F3 A#2 A3 F3 A#2 A3
     C3 A3 F3 C3 A3 F3 C3 A3
     A#2 A3 F3 A#2 A3 F3 A#2 A3
     C3 A3 F3 C3 A3 F3 C3 A3
     A#2 A3 F3 A#2 A3 F3 A#2 A3
     C3 A3 F3 C3 A3 F3 C3 A3
     D3 A3 F3 D3 A3 F3 D3 A3
     F2 A3 F3 F2 A3 F3 F2 A3
     A#2 [F3A3] [F3C4] [A#2F4] A3 F3 [A#2C5] A3
     [C3A4] A3 F3 C3 A3 F3 C3 A3
     A#2 [A3C4] [F3F4] [A#2A4] A3 F3 [A#2G4] A3
     [C3F4] A3 F3 [C3G4] A3 F3 C3 A3
     A#2 [F3A3] [F3C4] [A#2F4] A3 F3 [A#2C5] A3
     [C3A4] A3 F3 C3 A3 F3 C3 A3
     D3 [G3A3] [F3C4] [D3G4] A3 F3 [D3A4] A3
     [F2E4] [A3E4] F3 [F2F4] A3 F3 F2 A3
     A#2 [F3A3] [F3C4] [A#2F4] A3 F3 [A#2C5] A3
     [C3A4] A3 F3 C3 A3 F3 C3 A3
     A#2 [A3C4] [F3F4] [A#2A4] A3 F3 [A#2G4] A3
     [C3C5] A3 F3 [C3A4] A3 F3 C3 A3
     A#2 [F3A3] [F3C4] [A#2F4] A3 F3 [A#2C5] A3
     [C3A4] A3 F3 C3 A3 F3 C3 A3
     D3 [G3A3] [F3C4] [D3G4] A3 F3 [D3A4] A3
     [F2E4] [A3E4] F3 [F2F4] A3 F3 F2 [G2A3]
     [A#2A5] A3 F3 A#2 A3 F3 A#2 A3
     [C3G5] A3 F3 [C3D5] A3 F3 C3 [A3E5]
     [A#2C5] A3 F3 A#2 A3 F3 A#2 [A3G4]
     [C3D5] A3 F3 [C3A4] A3 F3 C3 [A3C4]
     [A#2A#3] A3 F3 A#2 A3 F3 A#2 A3
     [C3C5] A3 F3 C3 A3 F3 C3 [A3G5] A5
     [D3D5G5] A3 F3 [D3E5] A3 F3 D3 A3
     F2 [A3F4] [F3E5] [F2D5] A3 F3 F2 A3
     A#2 A3 F3 A#2 A3 F3 A#2 A3
     C3 A3 F3 C3 A3 F3 C3 A3
     A#2 A3 F3 A#2 A3 F3 A#2 A3
     C3 A3 F3 C3 A3 F3 C3 A3
     A#2 A3 F3 A#2 A3 F3 A#2 A3
     C3 A3 F3 C3 A3 F3 C3 A3
     D3 A3 F3 D3 A3 F3 D3 A3
     F2 A3 F3 F2 A3 F3 F2 A3 
    """
    
    print("Генерация произведения...")
    melody_audio = create_melody(melody_text, note_duration=0.35, fs=fs, a4_freq=440.0)    
    write('lab1_melody.wav', fs, melody_audio)
    print("Файл lab1_melody.wav успешно сгенерирован.\n")

    t_melody = np.linspace(0, len(melody_audio)/fs, len(melody_audio), endpoint=False)

    plt.figure(figsize=(10, 4))
    plt.plot(t_melody, melody_audio)
    plt.xlim(0, 0.35)
    plt.title("Задание 3.3: Форма сигнала первой ноты (350 мс мелодии)")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.show()