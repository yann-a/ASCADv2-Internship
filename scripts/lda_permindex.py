# Compute and pickle LDA for the permutation indexes

from utils.lda import lda, pi

lda(
    'perm_index',
    'snr/permindex/permindex{}.npy',
    nb_bytes=16,
    nb_classes=16,
    lda_dim=5,
    save=False,
    output_path='lda/permindex'
)
