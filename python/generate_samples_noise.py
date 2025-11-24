import os
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from train_cgan import Generator
import torch.nn as nn

CHECKPOINT   = "python/generator_advanced.pth"
OUTPUT_DIR   = "synth_samples_latent_unet"
NUM_SAMPLES  = 500
BEAT_LENGTH  = 2000       
LATENT_DIM   = 100        
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionMapperUNet(nn.Module):
    """
    U-Net style 1D conv mapper: projects a latent vector into a multiscale
    2-channel time series [B,2,BEAT_LENGTH] for conditioning the generator.
    """
    def __init__(self, latent_dim, beat_length):
        super().__init__()
        self.beat_length = beat_length
        #Encoder: latent vector -> feature map
        self.fc = nn.Linear(latent_dim, 128 * (beat_length // 16))
        #Downsampling conv blocks
        self.down1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )  
        self.down2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )  
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )
        #Upsampling
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ) 
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ) 
        self.final = nn.Sequential(
            nn.Conv1d(256, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z).view(B, 128, self.beat_length // 16)
        # Downsample
        d1 = self.down1(x)     
        d2 = self.down2(d1)  
        b = self.bottleneck(d2)
        # Upsample + skip
        u2 = self.up2(b)     
        if d1.shape[2] != u2.shape[2]:
            minlen = min(d1.shape[2], u2.shape[2])
            d1 = d1[:, :, :minlen]
            u2 = u2[:, :, :minlen]
        u2 = torch.cat([u2, d1], dim=1)

        u1 = self.up1(u2)       
        if x.shape[2] != u1.shape[2]:
            minlen = min(x.shape[2], u1.shape[2])
            x_crop = x[:, :, :minlen]
            u1 = u1[:, :, :minlen]
        else:
            x_crop = x
        u1 = torch.cat([u1, x_crop], dim=1)  
        #Final upsample
        out = self.final(u1)  
        out = nn.functional.interpolate(out, size=self.beat_length, mode='linear', align_corners=True)
        return out



def generate_latent_unet_samples():
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    G.eval()

    #Unet mapper
    mapper = ConditionMapperUNet(LATENT_DIM, BEAT_LENGTH).to(DEVICE)
    mapper.eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in range(NUM_SAMPLES):
        #Sample latent code
        z = torch.randn(1, LATENT_DIM, device=DEVICE)
        cond = mapper(z)  

        with torch.no_grad():
            out = G(cond)
        bsp_gen, hsp_gen = out.squeeze(0).cpu().numpy()

        fname = os.path.join(OUTPUT_DIR, f"synth_unet_{i:03d}.npz")
        np.savez(fname, bsp=bsp_gen, hsp=hsp_gen)
        print(f"Saved: {fname}")

if __name__ == "__main__":
    generate_latent_unet_samples()
    #Plot first sample
    first = os.path.join(OUTPUT_DIR, "synth_unet_000.npz")
    if os.path.exists(first):
        data = np.load(first)
        bsp, hsp = data['bsp'], data['hsp']
        plt.figure(figsize=(12,6))
        plt.subplot(2,1,1)
        plt.plot(bsp)
        plt.title('BSP')
        plt.subplot(2,1,2)
        plt.plot(hsp)
        plt.title('HSP')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Missing: {first}")
