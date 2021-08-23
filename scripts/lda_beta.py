# Compute and pickle LDA for beta

from utils.lda import lda, pi

lda(
    'beta_mask',
    'snr/beta.npy',
    save=False,
    output_path='lda/beta'
)
