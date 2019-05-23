import numpy as np
import scipy as sp
import pickle
import scipy.io as sio
import scipy.stats as st
import simplejson
from scipy.optimize import minimize
from enum import Enum

"""
Load embedding file and estimate noise constant values and embedding scaling
"""


class KNormalizationType(Enum):
    CONSTANT = 0
    NORMALIZED = 1
    DECAYING = 2


class NoiseModel(Enum):
    BT = 0
    NORMAL = 1


# generate user and random pairs to get Perr
def get_random_pairs(N, M):
    indices = np.random.choice(N, (int(1.5*M), 2))
    indices = [(i[0], i[1]) for i in indices if i[0] != i[1]]
    assert len(indices) >= M
    return np.asarray(indices[0:M])


def process_embedding(EMBED_FILE):
    DATASET_HOME = './make-embedding/data'
    embedding_file = DATASET_HOME + '/' + EMBED_FILE
    mat_dict = sio.loadmat(embedding_file)

    D = mat_dict['X'].shape[1]  # dimension
    N = mat_dict['X'].shape[0]  # number of embedding points
    Embedding = mat_dict['X']

    Embedding_mean = np.mean(Embedding, 0)
    Embedding_mean = Embedding_mean.reshape((1, -1))
    assert Embedding_mean.shape == (1, D)

    Embedding = Embedding - Embedding_mean  # center embedding
    Embedding_cov = np.cov(Embedding, rowvar=False)

    evals, _ = np.linalg.eig(Embedding_cov)
    nstd = 3
    embed_scale = np.sqrt(D) / (nstd * np.sqrt(np.ndarray.min(evals)))

    Embedding = embed_scale * Embedding

    print('embedding {}, {} points x {} dimens'.format(
        embedding_file, N, D))

    # get triplets
    def load_actual_triplets():
        dset = simplejson.load(open(DATASET_HOME+'/dataset.json'))

        uuid_map = {uuid: i for i, uuid in enumerate(dset['image_uuids'])}
        triplets = []
        for line in open(DATASET_HOME+'/all-triplets.txt').readlines():
            (a, b, c) = line.replace('\n', '').split(' ')
            triplets.append((uuid_map[a], uuid_map[b], uuid_map[c]))
        triplets = np.array(triplets)

        # the data set's triplets are of the form |t[0]-t[1]| < |t[0]-t[2]|
        return (len(uuid_map), triplets)

    [num_triplet_points, triplets] = load_actual_triplets()

    print('triplets {} of {} points'.format(triplets.shape[0],
                                            num_triplet_points))
    assert num_triplet_points == N
    print(triplets.shape)
    Ntriplets = triplets.shape[0]

    # compute log likelihood for a given k
    # the data set's triplets are of the form |t[0]-t[1]| < |t[0]-t[2]|
    W_sim = Embedding[triplets[:, 0], :]
    A_sim_orig = 2*(Embedding[triplets[:, 1], :] -
                    Embedding[triplets[:, 2], :])
    tau_sim_orig = (np.linalg.norm(Embedding[triplets[:, 1], :], axis=1)**2
                    - np.linalg.norm(Embedding[triplets[:, 2], :], axis=1)**2)
    anorms = np.linalg.norm(A_sim_orig, axis=1)

    print(A_sim_orig.shape)
    print(W_sim.shape)
    print(tau_sim_orig.shape)

    z = np.zeros((Ntriplets,))
    num_errors = 0
    for i in range(Ntriplets):
        z[i] = np.dot(A_sim_orig[i, :], W_sim[i, :]) - tau_sim_orig[i]
        if z[i] < 0:
            num_errors += 1

    error_frac = num_errors/Ntriplets
    print('{}% error'.format(error_frac*100))

    kopt_dict = {}

    Ngenerate = 100000
    Pairs = get_random_pairs(N, Ngenerate)
    A_sim_test = 2*(Embedding[Pairs[:, 0], :] - Embedding[Pairs[:, 1], :])
    tau_sim_test = (np.linalg.norm(Embedding[Pairs[:, 0], :], axis=1)**2
                    - np.linalg.norm(Embedding[Pairs[:, 1], :], axis=1)**2)
    anorms_test = np.linalg.norm(A_sim_test, axis=1)

    for noise_model in NoiseModel:
        kopt_dict[noise_model.name] = {}
        kd = kopt_dict[noise_model.name]

        for k_normalization in KNormalizationType:
            kd[k_normalization.name] = {}

            if k_normalization == KNormalizationType.CONSTANT:
                A_sim = A_sim_orig
                tau_sim = tau_sim_orig
            elif k_normalization == KNormalizationType.NORMALIZED:
                A_sim = A_sim_orig / np.tile(anorms, (D, 1)).T
                tau_sim = tau_sim_orig / anorms
            elif k_normalization == KNormalizationType.DECAYING:
                A_sim = A_sim_orig * np.tile(np.exp(-anorms), (D, 1)).T
                tau_sim = tau_sim_orig * np.exp(-anorms)

            z = np.zeros((Ntriplets,))
            for i in range(Ntriplets):
                z[i] = np.dot(A_sim[i, :], W_sim[i, :]) - tau_sim[i]

            def neg_log_likelihood(k):
                if noise_model == NoiseModel.BT:
                    return -np.sum(np.log(sp.special.expit(k * z)))
                elif noise_model == NoiseModel.NORMAL:
                    return -np.sum(np.log(st.norm.cdf(z, loc=0, scale=1/k)))

            x0 = 1
            converged = False
            while not converged:
                print('optimizing k for normalization: ' +
                      k_normalization.name + ',  noise model: ' +
                      noise_model.name + ' at x0 = {:.4e}'.format(x0))
                lik_model = minimize(neg_log_likelihood, x0,
                                     method='L-BFGS-B', options={'disp': 101})
                if np.isfinite(lik_model.fun):
                    converged = True
                else:
                    x0 /= 2

            print(lik_model)

            kopt = np.asscalar(lik_model.x)
            kd[k_normalization.name]['kopt'] = kopt
            kd[k_normalization.name]['neg-log-likelihood'] = neg_log_likelihood(
                kopt)

            print('testing optimized k')
            if k_normalization == KNormalizationType.CONSTANT:
                A_sim = A_sim_test
                tau_sim = tau_sim_test
            elif k_normalization == KNormalizationType.NORMALIZED:
                A_sim = A_sim_test / np.tile(anorms_test, (D, 1)).T
                tau_sim = tau_sim_test / anorms_test
            elif k_normalization == KNormalizationType.DECAYING:
                A_sim = A_sim_test * np.tile(np.exp(-anorms_test), (D, 1)).T
                tau_sim = tau_sim_test * np.exp(-anorms_test)

            num_errors = 0
            for i in range(Ngenerate):
                z = np.dot(A_sim[i, :], W_sim[i, :]) - tau_sim[i]

                if noise_model == NoiseModel.BT:
                    y_sim = int(np.random.binomial(
                        1, sp.special.expit(kopt * z)))
                elif noise_model == NoiseModel.NORMAL:
                    y_sim = int(z + (1/kopt)*np.random.randn() > 0)

                if y_sim != (z > 0):
                    num_errors += 1

            model_error_frac = num_errors/Ngenerate
            kd[k_normalization.name]['model_error_frac'] = model_error_frac

            print('{:g}% gen errors'.format(model_error_frac*100))
            print('kopt: {:20}, neg-log-likelihood: {:e}'.format(kopt,
                                                                 kd[k_normalization.name]['neg-log-likelihood']))
            print('')

        kd['best_likelihood'] = min(
            kd.items(), key=lambda x: x[1]['neg-log-likelihood'])[0]

    with open(embedding_file[:-4] + '_processed.pickle', 'wb') as handle:
        pickle.dump(
            {'kopt': kopt_dict,
             'error_frac': error_frac,
             'embed_scale': embed_scale,
             'EMBED_FILE': EMBED_FILE,
             'D': D,
             'N': N
             }, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sio.savemat(embedding_file[:-4] + '_processed',
                {'kopt': kopt_dict,
                 'error_frac': error_frac,
                 'embed_scale': embed_scale,
                 'EMBED_FILE': EMBED_FILE,
                 'D': D,
                 'N': N
                 })


if __name__ == '__main__':
    EMBED_FILES = {
        2: 'output-d220180512-001631.mat',
        3: 'output-d320180509-165802.mat',
        4: 'output-d420180509-165958.mat',
        5: 'output-d520180512-000800.mat',
        6: 'output-d620180512-001232.mat',
        7: 'output-d720180512-001205.mat',
        9: 'output-d9-20190428-020247.mat',
        12: 'output-d12-20190428-020705.mat',
        15: 'output-d15-20190428-020815.mat',
        20: 'output-d20-20190428-020534.mat'
    }

    for f in EMBED_FILES.values():
        process_embedding(f)
