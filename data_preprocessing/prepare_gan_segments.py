import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import resample

# ===== USER SETTINGS =====
PAIRED_DIR   = 'paired_data/CHARLES_PSTOV-12-07-29'  
OUTPUT_DIR   = 'gan_segments/CHARLES_PSTOV-12-07-29'
WINDOW_SEC   = 2.0    
OVERLAP      = 0.5    # 50% overlap
PLOT_EXAMPLE = True   #Plot the first segment for sanity check
# ==========================

#Concatenate all runs end-to-end
fs = None
bsp_list = []
hsp_list = []
for fname in sorted(os.listdir(PAIRED_DIR)):
    if not fname.endswith('.npz'): continue
    data = np.load(os.path.join(PAIRED_DIR, fname))
    bsp = data['bsp']
    hsp = data['hsp']
    fs0 = float(data['fs'])
    if fs is None:
        fs = fs0
    elif abs(fs - fs0) > 1e-6:
        raise RuntimeError(f"Sampling rate mismatch: {fs} vs {fs0}")
    bsp_list.append(bsp)
    hsp_list.append(hsp)

bsp_all = np.vstack(bsp_list)
hsp_all = np.vstack(hsp_list)
print(f"Concatenated {len(bsp_list)} runs: total {bsp_all.shape[0]} samples at {fs} Hz")

#Sliding window parameters
win_samps = int(WINDOW_SEC * fs)
step = int(win_samps * (1 - OVERLAP)) or win_samps
n_total = bsp_all.shape[0]

#Create output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Extract and save segments
count = 0
for start in range(0, n_total - win_samps + 1, step):
    end = start + win_samps
    bsp_seg = bsp_all[start:end]
    hsp_seg = hsp_all[start:end]
    out_name = f"window{count:04d}.npz"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    np.savez(out_path, bsp=bsp_seg, hsp=hsp_seg, fs=fs)
    count += 1
print(f"Saved {count} segments of {WINDOW_SEC}s each to '{OUTPUT_DIR}'")

if PLOT_EXAMPLE and count>0:
    seg = np.load(os.path.join(OUTPUT_DIR, 'window0000.npz'))
    bsp0, hsp0 = seg['bsp'], seg['hsp']
    t = np.arange(bsp0.shape[0]) / fs
    fig, axes = plt.subplots(2,1, figsize=(10,4), sharex=True)
    axes[0].plot(t, bsp0[:,0], 'k'); axes[0].set_title('BSP Channel 0')
    axes[1].plot(t, hsp0[:,0], 'k'); axes[1].set_title('HSP Node 0')
    for ax in axes:
        ax.set_ylabel('mV')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        ax.grid(which='minor', linestyle=':', alpha=0.5)
        ax.grid(which='major', linestyle='-', alpha=0.7)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
