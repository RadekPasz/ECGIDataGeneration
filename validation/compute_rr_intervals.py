import os
import glob
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt

def load_ecg(path):
    data = np.load(path)
    fs   = float(data["fs"])
    hsp  = data["hsp"]
    if hsp.ndim == 1:
        ecg = hsp
    else:
        ecg = hsp[:, 0]
    return ecg, fs

def bandpass(x, fs, low=0.5, high=40, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, x)

def compute_rr_intervals(ecg_raw, fs):
    #Bandpass filter
    ecg = bandpass(ecg_raw, fs)

    #R peaks
    min_distance = int(0.2 * fs)
    r_peaks, _ = find_peaks(ecg, distance=min_distance, prominence=0.05)

    if len(r_peaks) < 2:
        return []

    rr_samples = np.diff(r_peaks) 

    rr_ms = (rr_samples / fs) * 1000.0

    return rr_ms.tolist()

real_dirs = [
    "gan_segments/CHARLES_PSTOV-12-07-27",
    "gan_segments/CHARLES_PSTOV-12-07-28"
]
real_paths = []
for d in real_dirs:
    real_paths += glob.glob(os.path.join(d, "*.npz"))

synth_dir   = "synth_samples"
synth_paths = glob.glob(os.path.join(synth_dir, "*.npz"))

def aggregate_rr(paths):
    all_rr = []
    for p in paths:
        ecg, fs = load_ecg(p)
        rr_list = compute_rr_intervals(ecg, fs)
        all_rr.extend(rr_list)
    return np.array(all_rr)

real_rr   = aggregate_rr(real_paths)
synth_rr  = aggregate_rr(synth_paths)

print(f"Real data:")
print(f"  → Average RR: {np.mean(real_rr):.1f} ms (±{np.std(real_rr):.1f})")
print(f"Synthetic data:")
print(f"  → Average RR: {np.mean(synth_rr):.1f} ms (±{np.std(synth_rr):.1f})")

plt.figure(figsize=(6,4))
plt.boxplot([real_rr, synth_rr], tick_labels=["Real", "Synthetic"])
plt.title("RR Interval Comparison")
plt.ylabel("Interval Duration (ms)")
plt.tight_layout()
plt.show()
