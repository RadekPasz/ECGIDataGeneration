Generating a Synthetic Dataset Based on Real Data for ECGi
Konrad Paszynski

This repository contains code and data for my Bachelor thesis (January–June 2025) on generating synthetic BSP–HSP (body‐surface and heart‐surface potential) pairs via a conditional sequential GAN.

Required dependencies are numpy, scipy, matplotlib, scikit-learn, torch, fastdtw, tqdm, plotly, tsnecuda
They can be downloading by running: pip install numpy scipy matplotlib scikit-learn torch fastdtw tqdm plotly tsnecuda

For data preprocessing, first load_and_preprocess_data should be ran, followed by prepare_gan_segments

The files for training the GAN ad generating samples can be found under python/

The validation/ folder contains methods that were used to assess the quality of the generated data

