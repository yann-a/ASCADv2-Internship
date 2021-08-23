# Computes LDA and PI
import numpy as np
import scalib.modeling as mdl
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm

from utils.multilabelize import multilabelize

# Variables
traces_path = 'data/traces{}.npy'
metadata_path = 'data/metadata{}.npz'
nb_traces_per_file = 99_999
nb_files = 1
lda_batch_size = 10_000

def lda(
    target_label,
    snr_path,
    nb_poi=2800,
    lda_dim=28,
    nb_bytes=1,
    nb_classes=256,
    can_be_zero=True,
    nb_files_train=1,
    gemm_mode=1,
    save=False,
    output_path=None
):
    pois = [np.argsort(np.load(snr_path.format(index_byte)))[-nb_poi:] for index_byte in range(nb_bytes)]

    lda = mdl.MultiLDA(
        [nb_classes] * nb_bytes,
        [lda_dim] * nb_bytes,
        pois,
        gemm_mode
    )

    for index_train_file in tqdm(range(nb_files_train)):
        traces_train = np.load(traces_path.format(index_train_file + 1), mmap_mode='r')
        metadata_train = np.load(metadata_path.format(index_train_file + 1))
        multilabel_train = multilabelize(
            metadata_train['masks'][:],
            metadata_train['keys'][:],
            metadata_train['plaintext'][:]
        )

        l = multilabel_train[target_label][:-1].astype(np.uint16)
        if not can_be_zero:
            l -= 1

        for s in tqdm(range(0, nb_traces_per_file, lda_batch_size)):
            traces = traces_train[s:min(s + lda_batch_size, nb_traces_per_file), :].astype(np.int16)
            labels = l[s:min(s + lda_batch_size, nb_traces_per_file), :]
            lda.fit_u(traces, labels)
            del traces
        del traces_train

    lda.solve(done=True)

    if save:
        if output_path is None:
            print('No filename provided. Skipping saving')
        else:
            with open(output_path, 'wb') as output_file:
                pickle.dump(lda, output_file)

    return lda

def pi(
    lda,
    target_byte,
    nb_bytes=1,
    nb_classes=256,
    can_be_zero=True,
    null_threshold=1e-20,
    nb_files_train=2,
    nb_files_test=1
):
    pis = [[] for _ in range(nb_bytes)]
    for index_test_file in tqdm(range(nb_files_train, nb_files_train + nb_files_test)):
        traces_test = np.load(traces_path.format(index_test_file + 1), mmap_mode='r')
        metadata_test = np.load(metadata_path.format(index_test_file + 1))
        multilabel_test = multilabelize(
            metadata_test['masks'][:20_000],
            metadata_test['keys'][:20_000],
            metadata_test['plaintext'][:20_000]
        )

        probas = list(lda.predict_proba(traces_test[:20_000].astype(np.int16)))
        labels = multilabel_test[target_byte][:len(probas)]

        if not can_be_zero:
            labels -= 1

        for index_byte in range(nb_bytes):
            prs = probas[index_byte]
            prs[np.where(prs < null_threshold)] = null_threshold
            pi = np.mean(np.log2(prs[np.arange(len(labels)), labels]))
            pis[index_byte].append(pi)

    for index_byte in range(nb_bytes):
        pi = np.log2(nb_classes) + np.mean(np.array(pis[index_byte]))
        print(f'For byte no {index_byte}, pi = {pi} with maximum {np.log2(nb_classes)}')
