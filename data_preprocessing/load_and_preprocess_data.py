import os
import argparse
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, iirnotch, resample
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_and_filter_bsp(matfile, hp=0.5, lp=40.0, notch=50.0):
    """
    Load raw BSP from a .mat file ('bspm.potvals', shape 120 x N) and
    apply bandpass (hp–lp) and notch filtering. Returns (N x 120) array.
    """
    data = scipy.io.loadmat(matfile, struct_as_record=False, squeeze_me=True)
    bspm = data.get('bspm')
    if bspm is None:
        raise KeyError(f"'bspm' struct not found in {matfile}")
    raw = np.asarray(bspm.potvals)  
    fs = float(data.get('fs', 1000.0))

    def bandpass(x):
        #High-pass
        b, a = butter(1, hp/(fs/2), btype='high'); y = filtfilt(b, a, x)
        #Notch
        bn, an = iirnotch(notch/(fs/2), Q=30); y = filtfilt(bn, an, y)
        #Low-pass
        b, a = butter(4, lp/(fs/2), btype='low'); return filtfilt(b, a, y)

    #Filter all channels
    filtered = np.stack([bandpass(raw[i, :]) for i in range(raw.shape[0])], axis=1)
    return filtered, fs


def build_inverse_operator(A_full, channels, lam):
    """
    Subset A_full (352 x 350) to rows given by 'channels' (120 indices, 1-based),
    build Tikhonov inverse operator G = (A_sub^T A_sub + lam I)^-1 A_sub^T.
    Returns G shape (350, 120).
    """
    #Convert 1-based to 0-based indices
    idx = np.array(channels, dtype=int) - 1
    A_sub = A_full[idx, :]  
    #G = (A_sub^T A_sub + lam*I)^-1 * A_sub^T -> shape (350, 120)
    AtA = A_sub.T.dot(A_sub)
    G = np.linalg.inv(AtA + lam * np.eye(AtA.shape[0])).dot(A_sub.T)
    return G


def process_session(session_dir, output_dir, lam, downsample):
    #Load geometry
    geom_search_dir = os.path.join(session_dir, 'Meshes')
    if not os.path.isdir(geom_search_dir):
        geom_search_dir = session_dir
    geom_file = next((f for f in os.listdir(geom_search_dir)
                      if f.endswith('daltorso_registered.mat')), None)
    if not geom_file:
        print(f"[WARN] No geometry file in {geom_search_dir}"); return
    geom = scipy.io.loadmat(os.path.join(geom_search_dir, geom_file),
                            struct_as_record=False, squeeze_me=True)['geom']
    channels = geom.channels  

    #Load forward matrix
    fwd_dir = os.path.join(session_dir, 'FwdInvTransforms')
    fwd_file = next(f for f in os.listdir(fwd_dir)
                    if f.endswith('_forward_matrix.mat'))
    A_full = scipy.io.loadmat(os.path.join(fwd_dir, fwd_file))['A']  

    #Build inverse operator G
    G = build_inverse_operator(A_full, channels, lam)

    #Go over each .mat in Interventions
    int_root = os.path.join(session_dir, 'Interventions')
    for pace in sorted(os.listdir(int_root)):
        pace_dir = os.path.join(int_root, pace)
        if not os.path.isdir(pace_dir): continue
        for fn in sorted(os.listdir(pace_dir)):
            if not fn.endswith('.mat'): continue
            mat_path = os.path.join(pace_dir, fn)
            rec_id = f"{os.path.basename(session_dir)}_{pace}_{fn[:-4]}"

            #Load and filter BSP
            bsp, fs = load_and_filter_bsp(mat_path)
            #Compute HSP
            hsp = G.dot(bsp.T).T

            fs_out = fs
            if downsample and downsample < fs:
                n = int(bsp.shape[0] * downsample / fs)
                bsp = resample(bsp, n, axis=0)
                hsp = resample(hsp, n, axis=0)
                fs_out = downsample

            #Save
            dest = os.path.join(output_dir, os.path.basename(session_dir))
            os.makedirs(dest, exist_ok=True)
            out_file = os.path.join(dest, rec_id + '.npz')
            np.savez(out_file, bsp=bsp, hsp=hsp, fs=fs, fs_out=fs_out)
            print(f"Saved {out_file} | BSP {bsp.shape} | HSP {hsp.shape}")


def visualize_pair(npz_path, duration=5.0):
    data = np.load(npz_path, allow_pickle=True)
    bsp, hsp, fs = data['bsp'], data['hsp'], float(data['fs'])
    t = np.arange(bsp.shape[0]) / fs
    mask = t < duration
    fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax[0].plot(t[mask], bsp[mask,0]); ax[0].set_title('BSP: channel 0')
    ax[1].plot(t[mask], hsp[mask,0]); ax[1].set_title('HSP: node 0')
    for a in ax:
        a.set_ylabel('mV'); a.grid(which='both', linestyle=':', color='lightgray')
    ax[1].set_xlabel('Time (s)')
    plt.tight_layout(); plt.show()


def main():
    p = argparse.ArgumentParser(description='Prepare paired BSP–HSP dataset')
    p.add_argument('--root_dir',   default='.', help='Parent directory of CHARLES_PSTOV sessions')
    p.add_argument('--output_dir', default='paired_data', help='Where to save .npz')
    p.add_argument('--lambda', dest='lam', type=float, default=1e-4,
                   help='Tikhonov regularization parameter')
    p.add_argument('--downsample', type=int, default=None,
                   help='Resample output to this Hz')
    p.add_argument('--visualize',  help='Path to .npz to plot')
    p.add_argument('--vis_duration', type=float, default=5.0,
                   help='Seconds to visualize')
    args = p.parse_args()

    for session in sorted(os.listdir(args.root_dir)):
        sd = os.path.join(args.root_dir, session)
        if os.path.isdir(sd):
            print(f"Processing {sd}")
            process_session(sd, args.output_dir, args.lam, args.downsample)

    if args.visualize:
        print(f"Visualizing {args.visualize}")
        visualize_pair(args.visualize, args.vis_duration)

if __name__ == '__main__':
    main()
