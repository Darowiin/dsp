import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
from statsmodels.tsa.stattools import acf as stats_acf
import pyreaper
from functions import my_acf, my_dtft, psola

fs, data = wavfile.read('./voice/input1.wav')
if len(data.shape) > 1:
    data = data[:, 0]
x = data.astype(np.float32) / np.max(np.abs(data))
t = np.linspace(0, (len(x) - 1) / fs, len(x))

print(f"Частота дискретизации: {fs} Гц")
print(f"Длина сигнала: {len(x)} отсчётов ({len(x)/fs:.2f} сек)")
print("=" * 60)

print("\n1. ТЕСТ ЭКВИВАЛЕНТНОСТИ ФУНКЦИЙ ACF")
print("-" * 60)

max_lag_test = int(fs / 80)
acf_lib = stats_acf(x, adjusted=True, fft=True, nlags=max_lag_test)

R0 = my_acf(x, 0)
my_acf_vals = np.array([my_acf(x, m) / R0 for m in range(max_lag_test + 1)])

print(f"{'m':>3} | {'my_acf':>12} | {'statsmodels':>12} | {'разница':>10}")
print("-" * 50)
for m in range(11):
    diff = abs(my_acf_vals[m] - acf_lib[m])
    print(f"{m:3d} | {my_acf_vals[m]:12.8f} | {acf_lib[m]:12.8f} | {diff:.2e}")

is_equivalent = np.allclose(my_acf_vals, acf_lib, atol=1e-5)
print("-" * 50)
print(f"Результат теста: {'PASSED ✓' if is_equivalent else 'FAILED ✗'}")
print(f"Максимальное отклонение: {np.max(np.abs(my_acf_vals - acf_lib)):.2e}")

print("\n2. ОЦЕНКА ОСНОВНОГО ТОНА МЕТОДОМ АКФ")
print("-" * 60)

min_f0, max_f0 = 20, 500
min_lag = int(fs / max_f0)
max_lag = int(fs / min_f0)

acf_full = stats_acf(x, adjusted=True, fft=True, nlags=max_lag)
acf_search = acf_full.copy()
acf_search[0] = 0

search_region = acf_search[min_lag:max_lag]
peak_lag_acf = np.argmax(search_region) + min_lag
f0_acf = fs / peak_lag_acf

print(f"Диапазон поиска: {min_f0}-{max_f0} Гц (лаги {min_lag}-{max_lag})")
print(f"Найденный пик: лаг = {peak_lag_acf}")
print(f"Оценка F0 (АКФ): {f0_acf:.2f} Гц")

print("\n3. ОЦЕНКА ОСНОВНОГО ТОНА МЕТОДОМ ДВПФ")
print("-" * 60)

segment_duration = 3
N_segment = min(int(segment_duration * fs), len(x))
x_segment = x[:N_segment]

freqs = np.arange(40, 500, 1)
spectrum = my_dtft(x_segment, fs, freqs)

peak_idx_dtft = np.argmax(spectrum)
f0_dtft = freqs[peak_idx_dtft]

print(f"Длина анализируемого сегмента: {N_segment} отсчётов ({N_segment/fs:.2f} сек)")
print(f"Диапазон частот: {freqs[0]}-{freqs[-1]} Гц")
print(f"Оценка F0 (ДВПФ): {f0_dtft:.2f} Гц")

print("\n4. ОЦЕНКА ОСНОВНОГО ТОНА МЕТОДОМ GOOGLE REAPER")
print("-" * 60)

x_int16 = (x * 32767).astype(np.int16)
pm_times, pm, f0_times, f0_reaper, corr = pyreaper.reaper(x_int16, fs)

f0_voiced = f0_reaper[f0_reaper != -1]
f0_reaper_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0

print(f"Количество pitch marks: {np.sum(pm == 1)}")
print(f"Voiced участков: {len(f0_voiced)} из {len(f0_reaper)}")
print(f"Оценка F0 (REAPER, средняя): {f0_reaper_mean:.2f} Гц")
print(f"F0 диапазон: {np.min(f0_voiced):.2f} - {np.max(f0_voiced):.2f} Гц")

print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА ОЦЕНОК ОСНОВНОЙ ЧАСТОТЫ ГОЛОСА")
print("=" * 60)
print(f"{'Метод':<25} | {'F0, Гц':>10}")
print("-" * 40)
print(f"{'АКФ':<25} | {f0_acf:>10.2f}")
print(f"{'ДВПФ':<25} | {f0_dtft:>10.2f}")
print(f"{'Google REAPER (средняя)':<25} | {f0_reaper_mean:>10.2f}")
print("-" * 40)
print(f"{'Среднее значение':<25} | {np.mean([f0_acf, f0_dtft, f0_reaper_mean]):>10.2f}")
print("=" * 60)

print("\n6. PSOLA ПРЕОБРАЗОВАНИЕ")
print("-" * 60)
x_high = psola(x, fs, k=1.5)
wavfile.write('./voice/output_high.wav', fs, (x_high * 32767).astype(np.int16))
print("Файл output_high.wav сохранён (k=1.5, повышение тона)")

fig = plt.figure(figsize=(16, 14))
fig.suptitle('Анализ основного тона речевого сигнала', fontsize=14, fontweight='bold')

ax1 = plt.subplot(3, 2, 1)
test_lags = np.arange(min(100, len(my_acf_vals)))
ax1.plot(test_lags, my_acf_vals[:len(test_lags)], 'b-', label='my_acf', linewidth=2)
ax1.plot(test_lags, acf_lib[:len(test_lags)], 'r--', label='statsmodels.acf', linewidth=2)
ax1.set_xlabel('Лаг m')
ax1.set_ylabel('ACF(m)')
ax1.set_title('Тест эквивалентности: my_acf vs statsmodels.acf')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 2, 2)
diff_vals = np.abs(my_acf_vals - acf_lib)
ax2.semilogy(diff_vals, 'g-', linewidth=1.5)
ax2.axhline(y=1e-5, color='r', linestyle='--', label='Порог (1e-5)')
ax2.set_xlabel('Лаг m')
ax2.set_ylabel('|my_acf - statsmodels.acf|')
ax2.set_title('Абсолютная разница между реализациями ACF')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 2, 3)
lags_plot = np.arange(len(acf_search))
ax3.plot(lags_plot, acf_search, 'b-', linewidth=1)
ax3.axvline(x=min_lag, color='gray', linestyle='--', alpha=0.7, label=f'Мин. лаг ({min_lag})')
ax3.axvline(x=max_lag, color='gray', linestyle='--', alpha=0.7, label=f'Макс. лаг ({max_lag})')
ax3.scatter([peak_lag_acf], [acf_search[peak_lag_acf]], color='red', s=100, zorder=5, 
            label=f'Пик: лаг={peak_lag_acf}, F0={f0_acf:.1f} Гц')
ax3.set_xlabel('Лаг (отсчёты)')
ax3.set_ylabel('АКФ')
ax3.set_title(f'Автокорреляционная функция (оценка F0 = {f0_acf:.1f} Гц)')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, max_lag + 50])

ax4 = plt.subplot(3, 2, 4)
ax4.plot(freqs, spectrum, 'b-', linewidth=1)
ax4.scatter([f0_dtft], [spectrum[peak_idx_dtft]], color='red', s=100, zorder=5,
            label=f'Пик: F0={f0_dtft:.1f} Гц')
ax4.set_xlabel('Частота (Гц)')
ax4.set_ylabel('Амплитуда')
ax4.set_title(f'Амплитудный спектр ДВПФ (оценка F0 = {f0_dtft:.1f} Гц)')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(3, 2, 5)
ax5.plot(t, x, 'b-', linewidth=0.5, alpha=0.7, label='Сигнал')
pm_voiced = pm == 1
pm_indices = (pm_times[pm_voiced] * fs).astype(int)
pm_indices = pm_indices[pm_indices < len(x)]
ax5.scatter(pm_times[pm_voiced][:len(pm_indices)], x[pm_indices], 
            color='red', s=20, marker='x', label=f'Pitch marks ({np.sum(pm_voiced)})', zorder=5)
ax5.set_xlabel('Время (с)')
ax5.set_ylabel('Амплитуда')
ax5.set_title('Google REAPER: Pitch Marks')
ax5.legend(loc='upper right')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(3, 2, 6)
ax6.plot(f0_times, f0_reaper, 'b-', linewidth=1.5)
ax6.axhline(y=f0_reaper_mean, color='red', linestyle='--', linewidth=2,
            label=f'Средняя F0 = {f0_reaper_mean:.1f} Гц')
ax6.fill_between(f0_times, 0, f0_reaper, where=(f0_reaper > 0), alpha=0.3)
ax6.set_xlabel('Время (с)')
ax6.set_ylabel('F0 (Гц)')
ax6.set_title('Google REAPER: Контур основного тона')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_ylim([0, max(f0_voiced) * 1.2 if len(f0_voiced) > 0 else 500])

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.savefig('./voice/analysis_results.png', dpi=150, bbox_inches='tight')
print("\nГрафик сохранён в ./voice/analysis_results.png")
plt.show()