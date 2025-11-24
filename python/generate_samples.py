import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train_cgan import Generator  

CHECKPOINT   = "python/generator_advanced.pth"
OUTPUT_DIR   = "synth_samples"
NUM_SAMPLES  = 500
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SHIFT_MS = 50  

def generate():
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    G.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(NUM_SAMPLES):
        # 1) Randomly pick a real segment as the conditioning BSP
        fname = np.random.choice(list(Path("gan_segments_augmented").rglob("window*.npz")))
        data = np.load(str(fname))
        
        bsp_arr = data["bsp"]
        hsp_arr = data["hsp"]
        fs      = float(data["fs"])

        # Handle both 1D and 2D BSP/HSP arrays
        if bsp_arr.ndim == 1:
            bsp = bsp_arr.flatten()
        else:
            bsp = bsp_arr[:, 0]  # first BSP channel

        if hsp_arr.ndim == 1:
            hsp_cond = hsp_arr.flatten()
        else:
            hsp_cond = hsp_arr[:, 0]  # first HSP channel

        # 2) Build the generator input: [BSP, zeros]
        bsp_t = torch.from_numpy(bsp).float()
        cond = torch.stack([
                     bsp_t,
                     torch.zeros_like(bsp_t)
                 ], dim=0).unsqueeze(0).to(DEVICE)  # shape (1, 2, T)

        # 3) Generate new BSP–HSP pair
        with torch.no_grad():
            out = G(cond)  # shape (1, 2, T)
        bsp_gen, hsp_gen = out.squeeze(0).cpu().numpy()  # each length T

        # 4) Apply a random time shift in [0, MAX_SHIFT_MS] milliseconds
        shift_ms = np.random.uniform(0, MAX_SHIFT_MS)
        shift_samples = int((shift_ms / 1000.0) * fs)
        bsp_shifted = np.roll(bsp_gen, shift_samples)
        hsp_shifted = np.roll(hsp_gen, shift_samples)

        # 5) Save synthetic .npz (including metadata)
        out_path = os.path.join(OUTPUT_DIR, f"synth_{i:03d}.npz")
        np.savez(
            out_path,
            bsp=bsp_shifted,
            hsp=hsp_shifted,
            fs=fs,
            shift_ms=shift_ms
        )

        # 6) For the first sample only, also save a comparison plot
        if i == 0:
            real_min, real_max = hsp_cond.min(),    hsp_cond.max()
            gen_min,  gen_max  = hsp_shifted.min(), hsp_shifted.max()
            scale_vis = (real_max - real_min) / (gen_max - gen_min + 1e-12)
            hsp_vis = (hsp_shifted - gen_min) * scale_vis + real_min

            fig, (ax_bsp, ax_hsp) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            ax_bsp.plot(bsp, label="Real BSP")
            ax_bsp.plot(bsp_shifted, label=f"Gen BSP (shift {shift_ms:.1f} ms)")
            ax_bsp.set_ylabel("BSP amplitude")
            ax_bsp.legend(loc="upper right")

            ax_hsp.plot(hsp_cond, label="Real HSP")
            ax_hsp.plot(hsp_vis, label="Gen HSP (vis-scaled)")
            ax_hsp.set_xlabel("Sample index")
            ax_hsp.set_ylabel("HSP amplitude")
            ax_hsp.legend(loc="upper right")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "plot_000.png"))
            plt.close(fig)
            print(f"Also saved plot: {OUTPUT_DIR}/plot_000.png")

        print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate()
