import re

import numpy as np
from scipy import signal

from waveform import Waveform

def tone(f, t, waveform=Waveform.SIN, fs=44100):
    time = np.linspace(0, t, int(fs * t), endpoint=False)
    
    if waveform == Waveform.SIN:
        return np.sin(2 * np.pi * f * time).astype(np.float32)
    elif waveform == Waveform.SQUARE:
        return signal.square(2 * np.pi * f * time).astype(np.float32)
    elif waveform == Waveform.TRIANGLE:
        return signal.sawtooth(2 * np.pi * f * time, width=0.5).astype(np.float32)
    elif waveform == Waveform.SAWTOOTH:
        return signal.sawtooth(2 * np.pi * f * time).astype(np.float32)
    return np.zeros_like(time)

def musical_tone(f, t, waveform=Waveform.SIN, fs=44100, db=-20):
    if f == 0: return np.zeros(int(fs * t), dtype=np.float32)

    amplitudes = [1.0, 0.4, 0.2, 0.1, 0.05]
    combined_signal = np.zeros(int(fs * t), dtype=np.float32)
    
    for i, amp in enumerate(amplitudes):
        freq = f * (i + 1)
        if freq > 20000: break
        combined_signal += amp * tone(freq, t, waveform, fs)
            
    max_val = np.max(np.abs(combined_signal))
    if max_val > 0:
        combined_signal /= max_val
    
    num_samples = len(combined_signal)
    if db != 0:
        target_amplitude = 10**(db / 20)
        a = target_amplitude**(1 / num_samples)
        envelope = a**np.arange(num_samples)
        combined_signal *= envelope
    
    return combined_signal.astype(np.float32)

def get_freq(note_str, a4_freq=440.0):
    if not note_str: return 0
    notes_map = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    match = re.match(r"([A-G]#?)(\d)", note_str)
    if not match: return 0
    
    name, octave = match.groups()
    
    note_index = notes_map.index(name)
    
    semitones = (note_index - 9) + (int(octave) - 4) * 12
    
    return a4_freq * (2**(semitones / 12))

def create_melody(melody_text, note_duration=0.2, fs=44100, a4_freq=440.0):
    pattern = r'\[(.*?)\]|(\S+)'
    tokens = re.findall(pattern, melody_text)
    
    full_audio = []
    
    for chord_group, single_note in tokens:
        if chord_group:
            notes = re.findall(r'[A-G]#?\d', chord_group)
        else:
            notes = [single_note]
            
        chord_signal = np.zeros(int(fs * note_duration), dtype=np.float32)
        for n in notes:
            freq = get_freq(n, a4_freq=a4_freq)
            chord_signal += musical_tone(freq, note_duration, Waveform.SIN, fs, db=-18)
        
        m = np.max(np.abs(chord_signal))
        if m > 1.0:
            chord_signal /= m
            
        full_audio.append(chord_signal)
        
    return np.concatenate(full_audio)