# Compute and pickle LDA for the permuted values

from utils.lda import lda, pi

lda(
    'sbox_masked',
    'snr/permvalue/permvalue{}.npy',
    nb_bytes=16,
    save=False,
    output_path='lda/permvalue'
)
