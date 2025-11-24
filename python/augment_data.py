import os
import random
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

INPUT_ROOT    = Path("gan_segments")
OUTPUT_ROOT   = Path("gan_segments_augmented")
BSP_CHANNEL   = 0      
MIN_BEATS     = 2
MAX_BEATS     = 6
HEIGHT_FRAC   = 0.5    

def extract_and_repack(sig: np.ndarray, peaks: np.ndarray,
                       orig_beats: int, target_beats: int) -> np.ndarray:
    """
    Keep the first `target_beats` beat segments from `sig`, evenly spread
    over the original length by inserting flat baselines between them.
    """
    L = sig.shape[-1]
    period = L / orig_beats
    seglen = int(round(period))
    #Baseline = median of lowest half of values
    base = np.median(np.sort(sig)[: L // 2])

    chosen = peaks[:target_beats]
    segments = []
    for p in chosen:
        start = int(round(p - seglen/2))
        end   = start + seglen
        if start < 0:
            seg = np.pad(sig[0:end], (abs(start), 0), 'constant', constant_values=base)
        elif end > L:
            seg = np.pad(sig[start:L],    (0, end-L),    'constant', constant_values=base)
        else:
            seg = sig[start:end]
        segments.append(seg)

    #Build equal gaps
    gap_len = int(round((L - seglen * target_beats) / target_beats))
    gap = np.full(gap_len, base)

    out = []
    for seg in segments:
        out.append(seg)
        out.append(gap)
    out = np.concatenate(out)

    #Trim to original length
    if out.shape[-1] > L:
        out = out[:L]
    elif out.shape[-1] < L:
        out = np.pad(out, (0, L - out.shape[-1]), 'constant', constant_values=base)
    return out

for in_path in INPUT_ROOT.rglob("window*.npz"):
    data = np.load(in_path)
    bsp_all = data["bsp"]   
    hsp_all = data["hsp"]

    bsp = bsp_all[:, BSP_CHANNEL] if bsp_all.ndim > 1 else bsp_all
    hsp = hsp_all[:, 0]      if hsp_all.ndim > 1 else hsp_all

    L = bsp.shape[0]

    distance = int(round(L / MAX_BEATS * 0.5))
    height   = np.max(bsp) * HEIGHT_FRAC
    peaks, _ = find_peaks(bsp, distance=distance, height=height)

    orig_beats = min(len(peaks), MAX_BEATS)
    if orig_beats < MIN_BEATS: #skip if too few beats
        print(f"Skipping {in_path}: only {orig_beats} peaks found")
        continue

    #Choose a random target
    target = random.randint(MIN_BEATS, orig_beats)

    #Repack both signals
    bsp_new = extract_and_repack(bsp, peaks, orig_beats, target)
    hsp_new = extract_and_repack(hsp, peaks, orig_beats, target)

    rel = in_path.relative_to(INPUT_ROOT)
    out_path = OUTPUT_ROOT / rel.parent
    out_path.mkdir(parents=True, exist_ok=True)

    np.savez(out_path / rel.name, bsp=bsp_new, hsp=hsp_new, fs=data["fs"])
    print(f"Converted {rel} → beats={target}")

print("Augmentation complete.")
