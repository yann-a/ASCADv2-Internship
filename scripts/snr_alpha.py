# Compute SNR for alpha

from utils.snr import snr

snr(
    'alpha_mask',
    nb_classes=255,
    can_be_zero=False,
    nb_files=1,
    save=False,
    output_path='snr/alpha.npy'
)
