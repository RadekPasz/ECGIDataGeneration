import numpy as np
import torch
from pathlib import Path
from fastdtw import fastdtw
import matplotlib.pyplot as plt  

REAL_DIR   = "gan_segments_augmented"
SYNTH_DIR  = "synth_samples"
SEL_BSP_CH = 0
SEL_HSP_ND = 0
N_REAL     = 50
N_SYN      = 50

def load_signal(npz_path, bsp_ch, hsp_nd):
    data = np.load(npz_path)
    hsp = data["hsp"]
    if hsp.ndim > 1:
        sig = hsp[:, hsp_nd]
    else:
        sig = hsp
    return sig.flatten()

def compute_dtw(sig1, sig2):
    s1 = np.asarray(sig1).ravel().tolist()
    s2 = np.asarray(sig2).ravel().tolist()
    dist, _ = fastdtw(s1, s2, dist=lambda a, b: abs(a - b))
    return dist

#Gather file lists
real_files  = sorted(Path(REAL_DIR).rglob("window*.npz"))[:N_REAL]
synth_files = sorted(Path(SYNTH_DIR).rglob("synth_*.npz"))[:N_SYN]

#Load signals
real_sigs  = [load_signal(str(p), SEL_BSP_CH, SEL_HSP_ND) for p in real_files]
synth_sigs = [load_signal(str(p), SEL_BSP_CH, SEL_HSP_ND) for p in synth_files]

#Compute pairwise DTW for Real vs Real
real_vs_real = np.zeros((N_REAL, N_REAL))
for i in range(N_REAL):
    for j in range(i + 1, N_REAL):
        d = compute_dtw(real_sigs[i], real_sigs[j])
        real_vs_real[i, j] = real_vs_real[j, i] = d

#Compute DTW for Synthetic vs Real
synth_vs_real = np.zeros((N_SYN, N_REAL))
for i in range(N_SYN):
    for j in range(N_REAL):
        synth_vs_real[i, j] = compute_dtw(synth_sigs[i], real_sigs[j])

#Normalize
real_pairwise_raw = real_vs_real[np.triu_indices(N_REAL, k=1)]
all_dists_raw      = np.concatenate([real_pairwise_raw, synth_vs_real.ravel()])
mxd = all_dists_raw.max()
if mxd == 0:
    mxd = 1.0  # avoid division by zero

real_vs_real /= mxd
synth_vs_real /= mxd

#Recompute normalized real_pairwise
real_pairwise = real_vs_real[np.triu_indices(N_REAL, k=1)]

print(f"Real↔Real DTW:   mean = {real_pairwise.mean():.4f}, std = {real_pairwise.std():.4f}")
print(f"Synth↔Real DTW:  mean = {synth_vs_real.mean():.4f}, std = {synth_vs_real.std():.4f}")

plt.figure(figsize=(6, 4))
plt.boxplot(
    [real_pairwise, synth_vs_real.ravel()],
    tick_labels=["Real vs Real", "Synthetic vs Real"],
    patch_artist=True,
    boxprops=dict(facecolor="lightgray", color="black"),
    medianprops=dict(color="orange"),
)
plt.title("Normalized DTW Distance Comparison")
plt.ylabel("Normalized DTW Distance")
plt.tight_layout()
plt.show()
