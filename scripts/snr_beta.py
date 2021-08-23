# Compute SNR for beta

from utils.snr import snr

snr(
    'beta_mask',
    nb_files=1,
    save=False,
    output_path='snr/beta.npy'
)
