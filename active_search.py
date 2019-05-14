import numpy as np
import scipy.special as sc
import scipy as sp
import pystan
import pickle
from hashlib import md5
from enum import Enum
from matplotlib import pyplot as plt

"""
Implementation of InfoGain, EPMV, MCMV methods. See main() for
usage.
"""

class ActiveSearcher():

    """
    search object for InfoGain, EPMV, and MCMV methods.

    Callable methods:
    - initialize: initializes search object
    - getQuery: actively selects next search pair
    - getEstimate: produces user point estimate
    """
    
    my_model = """
    data {
        int<lower=0> D;       // space dimension
        int<lower=0> M;       // number of measurements so far
        real k;               // logistic noise parameter (scale)
        vector[2] bounds;      // hypercube bounds [lower,upper]
        int y[M];             // measurement outcomes
        vector[D] A[M];       // hyperplane directions
        vector[M] tau;        // hyperplane offsets
    }
    parameters {
        vector<lower=bounds[1],upper=bounds[2]>[D] W;          // the user point
    }
    transformed parameters {
        vector[M] z;
        for (i in 1:M)
            z[i] = dot_product(A[i], W) - tau[i];
    }
    model {
        // prior
        W ~ uniform(bounds[1],bounds[2]);
    
        // linking observations
        y ~ bernoulli_logit(k * z);
    }
    """
    
    
    def __init__(self):
        
        # make Stan model
        self.sm = pystan.StanModel(model_code=self.my_model)
        
    def initialize(self, embedding, k, normalization, method, bounds=np.array([-1,1]), 
                   Nchains=4, Nsamples=4000, pair_sample_rate=10**-3, 
                   plotting=False, plot_pause=0.5, scale_to_embedding=False, 
                   ref=None,lambda_pen_MCMV=1, lambda_pen_EPMV=None):
        """
        arguments:
            embedding: np.array - an N x d embedding of points
            k: noise constant value
            normalization: model normalization scheme
            method: pair selection method

        optional arguments:
            bounds: hypercube lower and upper bounds [lb, ub]
            Nchains: number of sampling chains
            Nsamples: number of posterior samples
            pair_sample_rate: downsample rate for pair selection

            plotting settings:
                plotting: plotting flag (bool)
                plot_pause: pause time between plots, in seconds
                scale_to_embedding: if True, scale plot to embedding
                ref: np.array - d x 1 user point vector
                lambda_pen_MCMV: lambda penalty for MCMV method
                lambda_pen_EPMV: lambda penalty for EPMV method
        """
        
        self.embedding = embedding
        self.k = k
        self.method = method
        self.normalization = normalization
        self.bounds = bounds
        self.Nchains = Nchains
        self.Nsamples = Nsamples
        
        Niter = int(2*Nsamples/Nchains)
        assert Niter >= 1000
        self.Niter = Niter
        self.N = embedding.shape[0]
        self.Npairs = int(pair_sample_rate * sp.special.comb(self.N,2))
        
        self.D = embedding.shape[1]
        self.oracle_queries_made = []
        self.mu_W = np.zeros(self.D)
        
        self.A = []
        self.tau = []
        self.y_vec = []
        
        self.plotting = plotting
        self.plot_pause = plot_pause
        self.scale_to_embedding = scale_to_embedding
        self.ref = ref
        self.lambda_pen_MCMV = lambda_pen_MCMV
        
        if lambda_pen_EPMV is None:
            self.lambda_pen_EPMV = np.sqrt(self.D)
        else:
            self.lambda_pen_EPMV = lambda_pen_EPMV
        
        
    def getQuery(self,oracle):
        """
        selects pair for searching

        arguments:
            oracle: function accepting two indices i,j and returning
                sorted pair
                
                arguments:
                    p: tuple (i,j) of query pair
                output: dict with key 'y' where y=1 selects p[0], y=0 selects
                    p[1]
        outputs:
            'query': (i,j)
            'oracle_output': output of oracle function
        """
        
        # given measurements 0..i, get posterior samples
        if not self.A:
            W_samples = np.random.uniform(self.bounds[0], self.bounds[1], (self.Nsamples, self.D))
        else:
            data_gen = {'D': self.D, 'k': self.k, 'M': len(self.A),
                        'A': self.A,
                        'tau': self.tau,
                        'y': self.y_vec,
                        'bounds':self.bounds}
            
            # get posterior samples
            # num_samples = iter * chains / 2, unless warmup is changed
            fit = self.sm.sampling(data=data_gen, iter=self.Niter, chains=self.Nchains,
                        init=0)  # , n_jobs=1)
            W_samples = fit.extract()['W']
        
        if W_samples.ndim < 2:
            W_samples = W_samples[:,np.newaxis]
        
        assert W_samples.shape == (self.Nsamples, self.D)
        self.mu_W = np.mean(W_samples, 0)
        
        # generate and evaluate a batch of proposal pairs
        Pairs = self.get_random_pairs(self.N, self.Npairs)
            
        if self.method == AdaptType.INFOGAIN:
            value = np.zeros((self.Npairs,))
            for j in range(self.Npairs):
                p = Pairs[j]
                (A_emb, tau_emb) = pair2hyperplane(p, self.embedding, self.normalization)
                value[j] = self.evaluate_pair(A_emb, tau_emb, W_samples, self.k)

            p = Pairs[np.argmax(value)]
     

        elif self.method == AdaptType.MCMV:
            Wcov = np.cov(W_samples, rowvar=False)
            value = np.zeros((self.Npairs,))
            
            for j in range(self.Npairs):
                p = Pairs[j]
                (A_emb, tau_emb) = pair2hyperplane(p, self.embedding, self.normalization)

                varest = np.dot(A_emb, Wcov).dot(A_emb)
                
                distmu = np.abs(
                        (np.dot(A_emb, self.mu_W) - tau_emb)
                            / np.linalg.norm(A_emb)
                    )

                # choose highest variance, but smallest distance to mean
                value[j] = self.k * np.sqrt(varest) - self.lambda_pen_MCMV * distmu
                
            p = Pairs[np.argmax(value)]

        elif self.method == AdaptType.EPMV:
            Wcov = np.cov(W_samples, rowvar=False)
            value = np.zeros((self.Npairs,))
            for j in range(self.Npairs):
                p = Pairs[j]
                (A_emb, tau_emb) = pair2hyperplane(p, self.embedding, self.normalization)

                assert np.dot(A_emb, W_samples.T).size == self.Nsamples
                
                varest = np.dot(A_emb, Wcov).dot(A_emb)
                p1 = np.mean(sp.special.expit(
                    self.k*(np.dot(A_emb, W_samples.T) - tau_emb)
                    ))

                assert p1.size == 1

                value[j] = (
                        self.k * np.sqrt(varest)
                        - self.lambda_pen_EPMV * np.abs(p1 - 0.5))

            p = Pairs[np.argmax(value)]
        else:   # random pair method
            p = Pairs[0]
        
        (A_sel, tau_sel) = pair2hyperplane(p, self.embedding, self.normalization)    
        self.A.append(A_sel)
        self.tau = np.append(self.tau,tau_sel)
        
        oracle_out = oracle(p)
        y = oracle_out['y']
        self.y_vec.append(y)
        
        self.oracle_queries_made.append(p)

        # for plotting during experiment
        if self.plotting:
            # diagnostic
            Nsplit = 0
            Isplit = []
            for j in range(1, W_samples.shape[0]):
                z = np.dot(A_sel, W_samples[j,:]) - tau_sel
                if z > 0:
                    Isplit.append(j)
                    Nsplit += 1

            plt.figure(189)
            plt.clf()

            if self.scale_to_embedding:
                ax_min = np.min(self.embedding[:,0])
                ax_max = np.max(self.embedding[:,0])
            else:
                ax_min = self.bounds[0]
                ax_max = self.bounds[1]
                
            if self.D == 1:
                y_samples = np.zeros(self.Nsamples)
                y_p0 = 0
                y_p1 = 0
                ay_min = -1
                ay_max = 1
                y_ref = 0
            else:
                y_samples = W_samples[:,1]
                y_p0 = self.embedding[p[0],1]
                y_p1 = self.embedding[p[1],1]
                
                if self.scale_to_embedding:
                    ay_min = np.min(self.embedding[:,1])
                    ay_max = np.max(self.embedding[:,1])
                else:
                    ay_min = self.bounds[0]
                    ay_max = self.bounds[1]
                
                if self.ref is not None:
                    y_ref = self.ref[1]

            plt.axis([ax_min,ax_max,ay_min,ay_max])
            plt.plot(W_samples[:,0], y_samples, 'y.')
            plt.plot(W_samples[Isplit,0], y_samples[Isplit], 'r.')
            
            if self.ref is not None:
                plt.plot(self.ref[0], y_ref, 'go')
                
            plt.plot(self.embedding[p[0],0], y_p0, 'bo')
            plt.plot(self.embedding[p[1],0], y_p1, 'bo')
            plt.ion()
            plt.pause(self.plot_pause) # for observation
        
        return p,oracle_out
        
    
    def getEstimate(self):
        """
        returns estimate of user point as d x 1 np.array
        """
        return self.mu_W
    

    def evaluate_pair(self, a, tau, W_samples, k):
        # mutual information heuristic, larger is better
        # NOTE: each row of W_samples is a sample
        Lik = self.likelihood_vec(a, tau, W_samples, k)
        Ftilde = np.mean(Lik)
    
        mutual_info = self.binary_entropy(Ftilde) - np.mean(
                self.binary_entropy(Lik))
    
        return mutual_info


    def likelihood_vec(self, a, tau, W, k):
        # a: (3,) tau: (1,) W: (1000, 3)
        z = np.dot(W, a) - tau  # broadcasting
        return sp.special.expit(k * z)


    def get_random_pairs(self, N, M):
        indices = np.random.choice(N, (int(1.5*M), 2))
        indices = [(i[0], i[1]) for i in indices if i[0] != i[1]]
        assert len(indices) >= M
        return indices[0:M]

    
    def binary_entropy(self, x):
        return -(sc.xlogy(x, x) + sc.xlog1py(1 - x, -x))/np.log(2)


class AdaptType(Enum):
    RANDOM = 0
    INFOGAIN = 1
    MCMV = 2
    EPMV  = 3
    ACTRANKQ = 4
    
    
class KNormalizationType(Enum):
    CONSTANT    = 0
    NORMALIZED  = 1
    DECAYING    = 2
    
class NoiseModel(Enum):
    BT = 0
    NONE = 1
    
def pair2hyperplane(p, embedding, normalization, slice_point = None):
        A_emb = 2*(embedding[p[0],:] - embedding[p[1],:])
    
        if slice_point is None:
            tau_emb = (np.linalg.norm(embedding[p[0],:])**2
                    - np.linalg.norm(embedding[p[1],:])**2)
        else:
            tau_emb = np.dot(A_emb, slice_point)
    
        if normalization == KNormalizationType.CONSTANT:
            pass
        elif normalization == KNormalizationType.NORMALIZED:
            A_mag = np.linalg.norm(A_emb)
            A_emb = A_emb / A_mag
            tau_emb = tau_emb / A_mag
        elif normalization == KNormalizationType.DECAYING:
            A_mag = np.linalg.norm(A_emb)
            A_emb = A_emb * np.exp(-A_mag)
            tau_emb = tau_emb * np.exp(-A_mag)
        return (A_emb, tau_emb)


def main():
    """
    Example usage of ActiveSearcher:

    - generates random embedding of items
    - defines search parameters
    - defines oracle
    - initalize search object
    - get paired comparison queries
    """

    N = 100
    d = 2
    max_query = 100
    
    embedding = np.random.randn(N, d)
    k = 10
    k_normalization = KNormalizationType.CONSTANT
    noise_model = NoiseModel.NONE
    method = AdaptType.INFOGAIN
    
    def oracle(p):
        # if y=1, then p[0] selected
        
        (a, tau) = pair2hyperplane(p, embedding, k_normalization)
        z = np.dot(a, ref) - tau
            
        if noise_model == NoiseModel.BT:
            y = int(np.random.binomial(1, sp.special.expit(k * z)))
        else:
            y = int(z > 0)
        
        return {'y':y,'z':z,'a':a,'tau':tau}
    
    print("Search points: ")
    print(embedding)

    bounds = [-1,1]
    ref = np.random.uniform(bounds[0], bounds[1], (d,1))

    print("Reference point: ")
    print(ref)

    searcher = ActiveSearcher()
    searcher.initialize(embedding,k,k_normalization,method,
                        pair_sample_rate=10**-3, plotting = True, ref=ref,
                        scale_to_embedding=True)
    
    queries_made = 0
    while queries_made < max_query:
        query,response = searcher.getQuery(oracle)
        
        if query is None:
            break
        
        queries_made += 1
        print('# queries made: {} / {}'.format(queries_made,max_query))
              

if __name__ == '__main__':
    main()
