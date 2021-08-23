# Run the attack and draw key rank accuracy by number of traces
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.multilabelize import multilabelize as multilabelizer, multGF256, AES_Sbox
import scalib.postprocessing.rankestimation as rk

traces = np.load('data/traces1.npy', mmap_mode='r')
metadata = np.load('data/metadata1.npz')

key_ranks = []
min_traces = 0
max_traces = 3000
traces_step = 10

for nb_traces in tqdm(range(min_traces, max_traces + 1, traces_step)):
    multilabelize = multilabelizer(
        metadata['masks'][:nb_traces],
        metadata['keys'][:nb_traces],
        metadata['plaintext'][:nb_traces]
    )

    # Draw key and artificially replace the dataset's key with it
    k = np.random.randint(0, 256, 16)
    plaintexts = metadata['keys'][:nb_traces] ^ metadata['plaintext'][:nb_traces] ^ k

    # Get alpha
    lda_alpha = pickle.load(open('lda/alpha', 'rb'))
    prs_alpha = list(lda_alpha.predict_proba(traces[:nb_traces].astype(np.int16)))[0]

    max_alpha = np.argmax(prs_alpha, axis=1)+1
    true_alpha = metadata['masks'][:nb_traces, 18]

    # Get beta
    lda_beta = pickle.load(open('lda/beta', 'rb'))
    prs_beta = list(lda_beta.predict_proba(traces[:nb_traces].astype(np.int16)))[0]

    max_beta = np.argmax(prs_beta, axis=1)
    true_beta = metadata['masks'][:nb_traces, 17]

    # Get permindex
    lda_permindex = pickle.load(open('lda/permindex', 'rb'))
    prs_permindex = np.array(list(lda_permindex.predict_proba(traces[:nb_traces].astype(np.int16))))

    max_permindexes = np.argmax(prs_permindex, axis=2).T
    true_permindexes = multilabelize['perm_index']

    # Get permvalues
    lda_permvalue = pickle.load(open('lda/permvalue', 'rb'))
    prs_permvalue = np.array(list(lda_permvalue.predict_proba(traces[:nb_traces].astype(np.int16))))

    max_permvalue = np.argmax(prs_permvalue, axis=2).T
    true_permvalue = multilabelize['sbox_masked']

    probas = np.zeros((nb_traces, 256, 16))
    index_manip = np.argmax(prs_permindex, axis=0)

    # Attack
    for target_byte in range(16):
        for key_byte in range(256):
            C = multGF256(max_alpha, AES_Sbox[plaintexts[:, target_byte] ^ key_byte]) ^ max_beta
            proba_c = prs_permvalue[index_manip[:, target_byte], np.arange(nb_traces), C]

            probas[:, key_byte, target_byte] = proba_c

    probas  = probas / np.sum(probas, axis=1)[:, None]
    key_probability = np.sum(np.log(probas), axis=0)

    # Estimate key rank and store it
    lmin, l, lmax = rk.rank_accuracy(-key_probability.T, k)

    key_ranks.append(l)

# Plot key rank accuracy
plt.plot(range(min_traces, max_traces + 1, traces_step), key_ranks)
print(key_ranks)
plt.xlabel('Number of traces')
plt.ylabel('Key rank estimation')
plt.yscale('log', base=2)
plt.yticks([np.float(2**0), np.float(2**16), np.float(2**32), np.float(2**48), np.float(2**64), np.float(2**80), np.float(2**96), np.float(2**112), np.float(2**128)])
plt.show()
