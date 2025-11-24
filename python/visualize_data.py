"""
visualize_synth_samples.py

Load and visualize one or more synthetic BSP–HSP samples from your cGAN output.
Adjust the USER SETTINGS below to pick which samples and channels/nodes to plot.
This version handles 1D arrays (single-channel output) or multi-channel.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import resample

SYNTH_DIR    = 'synth_samples'   
INDEX        = 9                 
BSP_CHANNEL  = 0                 
HSP_NODE     = 0                 


#filename = f'synth_{INDEX:03d}.npz'
filename = "synth_000.npz"
path = os.path.join(SYNTH_DIR, filename)
if not os.path.exists(path):
    raise FileNotFoundError(f"'{path}' not found")

data = np.load(path)

data = np.load(path)

bsp = data['bsp']  
hsp = data['hsp']  
fs  = float(data['fs'])

if bsp.ndim == 1:
    sig_bsp = bsp
else:
    sig_bsp = bsp[:, BSP_CHANNEL]

if hsp.ndim == 1:
    sig_hsp = hsp
else:
    sig_hsp = hsp[:, HSP_NODE]

n = sig_bsp.shape[0]
t = np.arange(n) / fs

def plot_ecg(ax, sig, title):
    ax.plot(t, sig, linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel('mV')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.04))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(which='minor', linestyle=':', alpha=0.5)
    ax.grid(which='major', linestyle='-', alpha=0.7)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
plot_ecg(axes[0], sig_bsp, f'Synthetic BSP Channel {"" if bsp.ndim==1 else BSP_CHANNEL}')
plot_ecg(axes[1], sig_hsp, f'Synthetic HSP Node {"" if hsp.ndim==1 else HSP_NODE}')
axes[1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
