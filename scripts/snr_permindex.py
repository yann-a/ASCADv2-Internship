# Comopute SNR for the permutation indexes

from utils.snr import snr

snr(
    'perm_index',
    nb_files=1,
    nb_classes=16,
    nb_bytes=16,
    save=False,
    output_path='snr/permindex/permindex{}.npy'
)
