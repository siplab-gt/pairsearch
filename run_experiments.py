import numpy as np
import scipy as sp
import scipy.io as sio
import pickle
import time
import sys
from enum import Enum
from actrankq import ActRankQ
from active_search import ActiveSearcher, AdaptType, KNormalizationType, pair2hyperplane

"""
Script to run paper experiments. Example usage:

python run_experiments.py 4 5 2 CONSTANT BT CONSTANT 0.0001 1 infogain actrank3

dim: 4
number of queries: 5
number of trials: 2
model normalization: CONSTANT
noise model: BT, Bradley-Terry logistic model
noise normalization: CONSTANT
pair subsampling rate (beta): 0.0001
embedding subsampling rate (if dataset subsampling is desired): 1
methods: infogain, actrank3
"""

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


class NoiseModel(Enum):
    BT = 0  # Bradley-Terry (logistic) noise
    NORMAL = 1  # Gaussian distributed noise
    NONE = 2  # noiseless


exper_types = [['random', AdaptType.RANDOM],
               ['infogain', AdaptType.INFOGAIN],
               ['mcmv', AdaptType.MCMV],
               ['epmv', AdaptType.EPMV],
               ['actrank1', AdaptType.ACTRANKQ, 1],
               ['actrank3', AdaptType.ACTRANKQ, 3],
               ['actrank5', AdaptType.ACTRANKQ, 5]
               ]

if len(sys.argv) < 10:
    print("""Must supply embedding dimension, number of iterations, 
          number of trials, model normalization, noise model,
          noise normalization, pair subsampling rate, embedding subsampling rate,
          and method numbers""")
    sys.exit(1)

dim = int(sys.argv[1]) # dimension
M = int(sys.argv[2]) # number of queries
ntrials = int(sys.argv[3]) # number of trials
k_normalization = KNormalizationType[sys.argv[4]] # model normalization
noise_model = NoiseModel[sys.argv[5]] # noise model type
noise_normalization = KNormalizationType[sys.argv[6]] # noise normalization
pair_ss_rate = float(sys.argv[7]) # pair subsampling rate
emb_ss_rate = float(sys.argv[8]) # embedding subsampling rate
methods = sys.argv[9:] # methods

status_file = ('output-data/jobstatus_timing_' + '_'.join(sys.argv[1:7]) +
               time.strftime("%Y%m%d-%H%M%S") + '_' +
               str(np.random.randint(1, 10000)) + '.txt')

print('embedding dimension:', dim)
print('M:', M)
print('ntrials:', ntrials)
print('model normalization:', k_normalization.name)
print('noise model:', noise_model.name)
print('noise normalization:', noise_normalization.name)
print('pair subsampling rate:', pair_ss_rate)
print('embedding subsampling rate:', emb_ss_rate)

for m in methods:
    print('method: {}'.format(m))

DATASET_HOME = './make-embedding/data'
EMBED_FILE = EMBED_FILES[dim]
embedding_file = DATASET_HOME + '/' + EMBED_FILE

mat_dict = sio.loadmat(embedding_file)
with open(embedding_file[:-4] + '_processed.pickle', 'rb') as handle:
    processed_dict = pickle.load(handle)

k = processed_dict['kopt']['BT'][k_normalization.name]['kopt']

if noise_model != NoiseModel.NONE:
    k_noise = processed_dict['kopt'][noise_model.name][noise_normalization.name]['kopt']
else:
    k_noise = 1e9

embed_scale = processed_dict['embed_scale']

D = mat_dict['X'].shape[1]  # dimension
assert D == dim

N = mat_dict['X'].shape[0]  # number of embedding points
Embedding = mat_dict['X']

exper_types = [e for e in exper_types if e[0] in methods]
print('methods:', [e[0] for e in exper_types])
print('embedding {}, {} points x {} dimens'.format(
    embedding_file, N, D))

bounds = np.array([-1, 1])  # preference point hypercube edges

Embedding_mean = np.mean(Embedding, 0)
Embedding_mean = Embedding_mean.reshape((1, -1))
assert Embedding_mean.shape == (1, D)

Embedding = Embedding - Embedding_mean  # center embedding
Embedding = embed_scale * Embedding
Embedding_cov = np.cov(Embedding, rowvar=False)

A = np.max(Embedding, axis=0)
B = np.min(Embedding, axis=0)

print(A)
print(B)

print(np.mean(Embedding, 0))
print(Embedding_cov)


def main():
    searcher = ActiveSearcher()  # construct searcher
    ranker = ActRankQ()  # construct ranker

    out_data = {}

    for e in exper_types:
        out_data[e[0]] = {'W_hist': [], 'W_sim': [], 'timer_vec': []}

    for trial in range(ntrials):

        for e in exper_types:

            print('trial', trial+1, '/', ntrials, 'experiment', e[0])

            f = open(status_file, 'a')
            f.write('started trial {}, method {}\n'.format(trial+1, e[0]))
            f.close()

            W_sim = np.random.uniform(bounds[0], bounds[1], (D, 1))
            print(np.squeeze(W_sim.T))

            if e[1] == AdaptType.ACTRANKQ:
                W_hist, timer_vec = run_single_experiment(
                    ranker, e[1], k, W_sim, round(N*emb_ss_rate), e[2])
            else:
                W_hist, timer_vec = run_single_experiment(searcher, e[1], k, W_sim,
                                                          item_subsample_size=round(N*emb_ss_rate))

            o = out_data[e[0]]
            o['W_hist'].append(W_hist) # M x d array of user point estimates
            o['W_sim'].append(W_sim) # ground-truth user point
            o['timer_vec'].append(timer_vec) # vector of timing data
            print('\n')

    sio.savemat('output-data/simulation-timing-output-'
                + time.strftime("%Y%m%d-%H%M%S") + '_' +
                str(np.random.randint(1, 10000)),
                {'data': out_data,
                 'D': D, 'k': k, 'ntrials': ntrials,
                 'k_normalization': k_normalization.name,
                 'noise_model': noise_model.name,
                 'noise_normalizaton': noise_normalization.name,
                 'pair_ss_rate': pair_ss_rate,
                 'emb_ss_rate': emb_ss_rate
                 })


def run_single_experiment(agent, adaptive, k, W_sim, item_subsample_size=None, nvotes=1):

    using_searcher = (adaptive != AdaptType.ACTRANKQ)

    # check for embedding downsampling
    if item_subsample_size is not None and item_subsample_size < N:
        subsample_idx = np.random.choice(
            list(range(N)), item_subsample_size, replace=False)
        sampled_Embedding = Embedding[subsample_idx, :]
    else:
        sampled_Embedding = Embedding

    print(
        'embedding subsampled to {} / {} items'.format(sampled_Embedding.shape[0], N))

    # reset search agent
    if using_searcher:
        agent.initialize(sampled_Embedding, k, k_normalization, adaptive,
                         bounds, pair_sample_rate=pair_ss_rate,
                         lambda_pen_EPMV=np.sqrt(np.trace(Embedding_cov)))
    else:
        agent.initialize(sampled_Embedding, nvotes, bounds, debug=False)

    def oracle(p):
        # if y=1, then p[0] selected

        (a, tau) = pair2hyperplane(p, sampled_Embedding, noise_normalization)
        z = np.dot(a, W_sim) - tau

        if noise_model == NoiseModel.BT:
            y = int(np.random.binomial(1, sp.special.expit(k_noise * z)))
        elif noise_model == NoiseModel.NORMAL:
            y = int(z + (1/k_noise)*np.random.randn() > 0)
        elif noise_model == NoiseModel.NONE:
            y = int(z > 0)

        return {'y': y, 'z': z, 'a': a, 'tau': tau}

    # generate pairwise measurements
    A_sim = [np.zeros((D))] * M
    tau_sim = np.zeros(M)
    y_sim = [int(0)] * M
    W_hist = np.zeros((M+1, D))
    timer_vec = np.zeros(M+1)

    tic = time.time()
    num_errors = 0
    for i in range(0, M+1):  # i measurements have been taken
        W_hist[i, :] = agent.getEstimate()
        elapsed = time.time() - tic
        timer_vec[i] = elapsed
        print(elapsed, 'elapsed')

        if i == M:
            # all M_exp measurements have been taken
            break

        print('measurement {} / {}'.format(i+1, M))

        query, oracle_out = agent.getQuery(oracle)

        if query is not None:
            y_sim[i] = oracle_out['y']
            z = oracle_out['z']
            A_sim[i] = oracle_out['a']
            tau_sim[i] = oracle_out['tau']

            if y_sim[i] != (z > 0):
                num_errors += 1
        else:
            y_sim[i] = None
            z = None
            A_sim[i] = np.empty((0, 0))
            tau_sim[i] = None

        print(A_sim[i].shape, W_sim.shape, adaptive.name)

    print('{} / {} individual errors'.format(num_errors, M))

    return W_hist, timer_vec


if __name__ == "__main__":
    main()
