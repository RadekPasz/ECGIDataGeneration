import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─── USER SETTINGS ────────────────────────────────────────────────────────────
PAIRS_DIR        = "gan_segments"   
SEL_BSP_CHANNEL  = 0                
SEL_HSP_NODE     = 0                
BEAT_LENGTH      = 2000             
BATCH_SIZE       = 8
LR_G             = 1e-4
LR_D             = 1e-4
EPOCHS           = 50
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Loss weights
theta_L1         = 100.0           
theta_CC         = 10.0            
theta_FM         = 1.0             
# ───────────────────────────────────────────────────────────────────────────────

class ResidualTCNBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3,
                               padding=dilation, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Identity()
        self.bn1 = nn.BatchNorm1d(ch)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(self.relu1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu2(out + residual)  #residual skip


class BSPHSPDataset(Dataset):
    def __init__(self, data_dir, bsp_ch, hsp_nd, beat_length):
        self.files = sorted(Path(data_dir).rglob("window*.npz"))
        self.bsp_ch     = bsp_ch
        self.hsp_nd     = hsp_nd
        self.beat_length= beat_length
        if not self.files:
            raise RuntimeError(f"No .npz windows found under {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        bsp  = data["bsp"] 
        hsp  = data["hsp"]  
        lead = bsp[:, self.bsp_ch]
        node = hsp[:, self.hsp_nd]
        x = np.stack([lead, node], axis=0)
        return torch.from_numpy(x).float()

class Generator(nn.Module):
    def __init__(self, in_ch=2, base_ch=32):
        super().__init__()
        #Encoder
        self.enc1 = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(base_ch)
        )
        self.enc2 = nn.Sequential(
            nn.ReflectionPad1d(6),
            nn.Conv1d(base_ch, base_ch*2, kernel_size=7, padding=0, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(base_ch*2)
        )
        self.enc3 = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(base_ch*2, base_ch*4, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(base_ch*4)
        )

        #Decoder
        self.dec3 = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(base_ch*4, base_ch*2, kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(base_ch*2)
        )
        self.dec2 = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(base_ch*4, base_ch,   kernel_size=7, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(base_ch)
        )
        self.dec1 = nn.Sequential(
            nn.ReflectionPad1d(3),
            nn.Conv1d(base_ch*2, 2,         kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)      
        e2 = self.enc2(e1)    
        e3 = self.enc3(e2)   
        d3 = self.dec3(e3)   
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)    
        d2 = torch.cat([d2, e1], dim=1)
        out = self.dec1(d2)    
        return out

class Discriminator(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.layer1 = nn.Sequential(
          nn.Conv1d(in_ch,  32, kernel_size=15, stride=4, padding=7),
          nn.LeakyReLU(0.2),
          nn.BatchNorm1d(32),
        )
        self.layer2 = nn.Sequential(
          nn.Conv1d(32,     64, kernel_size=15, stride=4, padding=7),
          nn.LeakyReLU(0.2),
          nn.BatchNorm1d(64),
        )
        self.layer3 = nn.Sequential(
          nn.Conv1d(64,    128, kernel_size=15, stride=4, padding=7),
          nn.LeakyReLU(0.2),
          nn.BatchNorm1d(128),
        )
        self.final  = nn.Conv1d(128, 1, 7, padding=3)
        self.sig    = nn.Sigmoid()

    def forward(self, x, feat_match=False):
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = self.sig(self.final(f3))
        if feat_match:
            return out, [f1, f2, f3]
        return out

#Loss helpers
def pearson_corr(x, y):
    xm = x.mean(-1, keepdim=True)
    ym = y.mean(-1, keepdim=True)
    xn = x - xm
    yn = y - ym
    r = (xn * yn).sum(-1) / (torch.sqrt((xn**2).sum(-1)) * torch.sqrt((yn**2).sum(-1)) + 1e-8)
    return r.mean()

def train():
    dataset = BSPHSPDataset(PAIRS_DIR, SEL_BSP_CHANNEL, SEL_HSP_NODE, BEAT_LENGTH)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    optG = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5,0.9))
    optD = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5,0.9))
    bce = nn.BCELoss()
    l1  = nn.L1Loss()

    for epoch in range(1, EPOCHS+1):
        for real in loader:
            real = real.to(DEVICE)    
            B = real.size(0)
            D.zero_grad()
            out_real = D(real).view(-1)
            lossD_real = bce(out_real, torch.ones_like(out_real))
            #fake
            bsp = real[:,0:1,:]
            fake = G(real)
            out_fake = D(fake.detach()).view(-1)
            lossD_fake = bce(out_fake, torch.zeros_like(out_fake))
            lossD = 0.5*(lossD_real + lossD_fake)
            lossD.backward(); optD.step()

            G.zero_grad()
            #adversarial
            out, feats_fake = D(fake, feat_match=True)
            loss_adv = bce(out.view(-1), torch.ones_like(out.view(-1)))
            #L1 reconstruction
            loss_l1 = l1(fake, real)
            #correlation loss
            loss_cc = 1 - pearson_corr(fake[:,1,:], real[:,1,:])
            #feature matching
            with torch.no_grad(): _, feats_real = D(real, feat_match=True)
            fm_loss = 0
            for fr, ff in zip(feats_real, feats_fake):
                fm_loss += l1(ff, fr)
            #combine
            lossG = loss_adv + theta_L1*loss_l1 + theta_CC*loss_cc + theta_FM*fm_loss
            lossG.backward(); optG.step()

        print(f"Epoch {epoch}/{EPOCHS} — D_loss: {lossD.item():.4f}, G_loss: {lossG.item():.4f}")

    torch.save(G.state_dict(), "checkpoints/generator_advanced.pth")
    torch.save(D.state_dict(), "checkpoints/discriminator_advanced.pth")
    print("✓ Training complete, advanced models saved")

if __name__ == "__main__":
    train()

