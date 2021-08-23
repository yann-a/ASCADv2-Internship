# Compute SNR
import numpy as np
import scalib.metrics as mts
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.multilabelize import multilabelize

# Variables
traces_path = 'data/traces{}.npy'
metadata_path = 'data/metadata{}.npz'
nb_values_per_trace = 1_000_000

# SNR function
def snr(
    target_label,
    nb_traces=99_999,
    nb_classes=256,
    nb_bytes=1,
    can_be_zero=True,
    nb_files=8,
    plot=True,
    save=False,
    output_path=None
):
    # Extract SNR
    snr = mts.SNR(nc=nb_classes, ns=nb_values_per_trace, np=nb_bytes)

    for index_file in tqdm(range(nb_files)):
        traces = np.load(traces_path.format(index_file + 1), mmap_mode='r')
        metadata = np.load(metadata_path.format(index_file + 1))

        multilabel = multilabelize(metadata['masks'][:nb_traces], metadata['keys'][:nb_traces], metadata['plaintext'][:nb_traces])

        traces = traces[:nb_traces].astype(np.int16)
        labels = np.reshape(multilabel[target_label][:nb_traces], newshape=(nb_traces, nb_bytes)).astype(np.uint16)

        if not can_be_zero:
            labels -= 1

        snr.fit_u(traces, labels)

    snr_val = snr.get_snr()

    if save:
        if output_path is None:
            print('No filename provided. Skipping saving')
        else:
            for index_byte in range(nb_bytes):
                np.save(output_path.format(index_byte), snr_val[index_byte])

    if plot:
        for index_byte in range(nb_bytes):
            plt.plot(snr_val[index_byte])
        plt.show()
