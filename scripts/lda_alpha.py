# Compute and pickle LDA for alpha

from utils.lda import lda, pi

lda(
    'alpha_mask',
    'snr/alpha.npy',
    nb_classes=255,
    can_be_zero=False,
    save=False,
    output_path='lda/alpha'
)
