# Transform ANSSI's h5 files to npy/npz files, used for the attack

import numpy as np
import h5py as h5
from tqdm import tqdm

base_filename = 'ascadv2-stm32-conso-raw-traces{}.{}'
N = 8

for index_file in tqdm(range(N, N+1)):
    with h5.File(base_filename.format(index_file, 'h5')) as data:
        traces = np.array(data['traces'])
        masks = np.array(data['metadata'][:, 'masks'])
        keys = np.array(data['metadata'][:, 'key'])
        plaintext = np.array(data['metadata'][:, 'plaintext'])
        ciphertext = np.array(data['metadata'][:, 'ciphertext'])

    with open('data/traces{}.npy'.format(index_file), 'wb') as npyfile:
        np.save(npyfile, traces)
    with open('data/metadata{}.npz'.format(index_file), 'wb') as npyfile:
        np.savez(npyfile, masks=masks, keys=keys, plaintext=plaintext, ciphertext=ciphertext)
