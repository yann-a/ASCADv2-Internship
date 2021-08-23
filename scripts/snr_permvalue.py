# Compute SNR for the permuted values

from utils.snr import snr

snr(
    'sbox_masked',
    nb_files=1,
    nb_bytes=16,
    save=False,
    output_path='snr/permvalue/permvalue{}.npy'
)
