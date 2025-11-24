import os
import glob
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

def load_ecg(path):
    data = np.load(path)
    fs   = float(data["fs"])
    hsp  = data["hsp"]
    #Handle both 1-D and 2-D arrays
    if hsp.ndim == 1:
        ecg = hsp
    else:
        ecg = hsp[:, 0]
    return ecg, fs

def bandpass(x, fs, low=0.5, high=40, order=3):
    nyq = 0.5*fs
    b,a = butter(order, [low/nyq, high/nyq], btype='bandpass')
    return filtfilt(b, a, x)

def compute_qrs_durations(ecg_raw, fs):
    ecg = bandpass(ecg_raw, fs)

    #Find R peaks
    r_peaks, _ = find_peaks(ecg, distance=int(0.2 * fs), prominence=0.05)

    #Search window
    win_hi = int(0.100 * fs)
    win_lo = int(0.020 * fs)
    win_post_lo = int(0.020 * fs)
    win_post_hi = int(0.100 * fs)

    durs = []
    for r in r_peaks:
        #Find Q-dip
        start_q = max(r - win_hi, 0)
        end_q   = max(r - win_lo, start_q + 1)
        seg_q   = ecg[start_q:end_q]
        if len(seg_q) == 0:
            continue
        q_idx = start_q + np.argmin(seg_q)

        #Find S wave
        start_s = min(r + win_post_lo, len(ecg) - 1)
        end_s   = min(r + win_post_hi, len(ecg))
        seg_s   = ecg[start_s:end_s]
        if len(seg_s) == 0:
            continue
        s_idx = start_s + np.argmin(seg_s)

        qrs_dur = (s_idx - q_idx) / fs * 1000.0
        durs.append(qrs_dur)

    return durs


real_dirs   = ["gan_segments/CHARLES_PSTOV-12-07-27",
               "gan_segments/CHARLES_PSTOV-12-07-28"]
synth_dir   = "synth_samples"

real_paths  = []
for d in real_dirs:
    real_paths += glob.glob(os.path.join(d, "*.npz"))

synth_paths = glob.glob(os.path.join(synth_dir, "*.npz"))

def aggregate(paths):
    all_durs = []
    for p in paths:
        ecg, fs = load_ecg(p)
        all_durs.extend(compute_qrs_durations(ecg, fs))
    return np.array(all_durs)

real_qrs  = aggregate(real_paths)
synth_qrs = aggregate(synth_paths)

print(f"Real data:")
print(f"  → Average QRS: {np.mean(real_qrs):.1f} ms (±{np.std(real_qrs):.1f})")
print(f"Synthetic data:")
print(f"  → Average QRS: {np.mean(synth_qrs):.1f} ms (±{np.std(synth_qrs):.1f})")

plt.figure()
plt.boxplot([real_qrs, synth_qrs], tick_labels=['Real', 'Synthetic'])
plt.title('QRS Duration Comparison')
plt.ylabel('Duration (ms)')
plt.tight_layout()
plt.show()
