import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cvxopt as cvx
cvx.solvers.options['show_progress'] = False
import scipy as sp
from enum import Enum
from active_search import KNormalizationType, pair2hyperplane
import time

"""
Active pairwise ranking, as in Jamieson & Nowak 2011 (see README)
Based on class skeleton by Stefano Fenu
"""


class ActRankQ():
    """
    ranker object for ActRankQ

    Callable methods:
    - initialize: initializes search object
    - getQuery: actively selects next search pair
    - getEstimate: produces user point estimate
    - getStats: returns stats on ActRankQ result
    """

    def initialize(self, embedding, nvotes=1, bounds=np.array([-1,1]), ref=None,
                   plot=False, debug=False, pause_len=2):
        """
        arguments:
            embedding: np.array - an N x d embedding of points

        optional arguments:
            nvotes: number of votes for query committee
            bounds: hypercube lower and upper bounds [lb, ub]
            ref: np.array - d x 1 user point vector
            plot: plotting flag (bool), only available in d=2
            debug: debugging flag (bool)
            pause_len: pause length between plots, in seconds
        """

        lb = bounds[0]
        ub = bounds[1]

        self.embedding = embedding
        self.nvotes = nvotes
        self.debug = debug
        self.indices = np.random.choice(list(range(len(embedding))),
            len(embedding),replace=False)
        d = embedding.shape[1]
        A = np.empty((0,d))
        b = np.empty((0,))

        for i in range(d):
            a0 = np.zeros((1,d))
            a0[0,i] = -1

            A = np.vstack((A,a0))
            b = np.append(b,-lb)

            a0 = np.zeros((1,d))
            a0[0,i] = 1

            A = np.vstack((A,a0))
            b = np.append(b,ub)

        if plot:
            self.feasible_region = FeasibleRegion(A=A,b=b,figure_handle=243,
                ref=ref,xrange=(lb,ub),yrange=(lb,ub),pause_len=pause_len)
        else:
            self.feasible_region = FeasibleRegion(A=A,b=b)

        self.oracle_queries_made = []
        self.y_vec = []
        self.ambiguous_q = 0
        self.total_q = 0
        self.total_ranking = [self.indices[0]]
        self.idx = 1
        self.L = 0
        self.R = len(self.total_ranking)-1
        self.ins = None
        self.acquired_votes = 0
        self.buffer_estimate = self.feasible_region.chebyshev_center()[0]
        self.t = time.time()

    def getQuery(self,oracle):
        """
        selects pair for ranking

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

        do_return = False

        while self.idx < len(self.indices):
            i = self.indices[self.idx]

            while self.L <= self.R:

                self.total_q += 1
                m = (self.L+self.R)//2
                j = self.total_ranking[m]

                if self.acquired_votes == 0:
                    ambiguity_test = self.feasible_region.ambiguous(
                        self.embedding,i,j)
                else:
                    ambiguity_test = 0

                if ambiguity_test == 0:
                    self.ambiguous_q += 1
                    p = (i,j)
                    oracle_out = oracle(p)
                    y = oracle_out['y']
                    self.y_vec.append(y)

                    if self.debug:
                        print('pair:{}, y:{}'.format(p,y))

                    closest,farthest = p[1-y],p[y]
                    self.oracle_queries_made.append(p)

                    self.acquired_votes = ((self.acquired_votes + 1) %
                        self.nvotes)

                    if self.acquired_votes > 0:
                        return p,oracle_out

                    y_total = sum(self.y_vec[-self.nvotes:])
                    if y_total == self.nvotes / 2:
                        y = np.random.binomial(1,0.5)
                    else:
                        y = int(y_total > self.nvotes / 2)
                    closest,farthest = p[1-y],p[y]

                    self.feasible_region.append(self.embedding, closest,
                        farthest)
                    self.buffer_estimate = (
                        self.feasible_region.chebyshev_center()[0])

                    comp = -1 + 2*(closest == j)

                    do_return = True
                else:
                    comp = ambiguity_test

                if comp > 0:
                    self.L = m+1
                    self.ins = m+1
                else:
                    self.R = m-1
                    self.ins = m

                if do_return:
                    return p,oracle_out

            self.total_ranking.insert(self.ins,i)
            self.L = 0
            self.R = len(self.total_ranking)-1

            if self.debug:
                if (self.idx+1) % 50 == 0:
                    print("    Inserted index {} / {} in {:.4f} s".
                    format(self.idx+1,len(self.indices),time.time()-self.t))

            self.idx += 1

        return None,None

    def getStats(self):
        """
        Returns stats on ActRankQ result
        """
        return (self.oracle_queries_made, self.total_ranking,
        self.ambiguous_q, self.total_q)

    def getEstimate(self):
        """
        returns estimate of user point as d x 1 np.array
        """
        return self.buffer_estimate


class FeasibleRegion():
    """
    Support function for ActRankQ
    feasible region defined by linear inequality constraints in the form of
    Ax <= b.
    """

    def __init__(self,A=None,b=None,figure_handle=None,ref=None,
                 xrange=(0.0,1.0),yrange=(0.0,1.0),pause_len = 2):
        self.figure_handle = figure_handle
        self.ref = ref
        self.xrange = xrange
        self.yrange = yrange
        self.res = 10
        self.t = np.linspace(self.xrange[0],self.xrange[1],self.res)
        self.pause_len = pause_len
        self.A = A
        self.b = b

    def __len__(self):
        return self.A.shape[0]

    def chebyshev_center(self):
        """
        Calculates the Chebyshev center of the current feasible region polytope.
        See S. Boyd, L. Vandenberghe, Convex Optimization. Code inspired from:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.
        HalfspaceIntersection.html

        solves the following linear program (norm indicates Euclidean norm):

        maximize_{x,R}    R
        s.t.    a_i^T x + R||a_i|| <= b_i    i = 1...m
                R >= 0

        Letting nv[i] = ||a_i||, we can write this as

        minimize_{x,R}    <[0,...0,-1],[x,R]>
        s.t.    [A,nv] [x] <= [b]
                [0,-1] [R]    [0]

        returns:
            xc: Chebyshev center
            R: depth of xc
        """
        norm_vector = np.reshape(np.linalg.norm(self.A,axis=1),
            (self.A.shape[0],1))
        A_ub = np.vstack((
                np.hstack((self.A,norm_vector)),
                np.hstack((np.zeros(self.A.shape[1]),-1))
                ))
        b_ub = np.append(self.b,0)

        c = np.zeros((self.A.shape[1]+1,))
        c[-1] = -1

        z,_ = cvxlp(c,A_ub,b_ub)
        #res = so.linprog(c, A_ub = A_ub, b_ub = b_ub)

        if z is not None:
            xc = z[:-1]
            R = z[-1]
        else:
            xc = None
            R = None

        return xc,R

    def ambiguous(self, embedding, i, j):
        """
        A query is only informative if it intersects the feasible region.

        embedding: np.array - an N x d embedding of points
        i, j: int - indices of embedding for two points in a binary ranking
            query
        return: int
            0: query is ambiguous
            -1: i < j in total order
            1: i > j in total order
        """

        if self.A is None:
            return 0

        normal, bias, _ = self.separator(embedding[i], embedding[j])
        iflag = self.intersects(self.A,self.b,normal,bias)

        (xc,R) = self.chebyshev_center()

        if self.figure_handle is not None and self.A.shape[1] == 2:

            color = 'g' if iflag==0 else 'r'

            cseq = ['k']*embedding.shape[0]
            cseq[i] = color
            cseq[j] = color

            plt.figure(self.figure_handle)
            plt.clf()

            plt.scatter(embedding[:,0],embedding[:,1],s=25,c=cseq)

            if self.ref is not None:
                plt.scatter(self.ref[0],self.ref[1],s=50,c='b',marker='*')

            for ii in range(self.A.shape[0]):
                a = self.A[ii,:]
                b = self.b[ii]
                plt.plot(self.t,(b-a[0]*self.t)/(a[1]+1e-10),'b')

            plt.plot(self.t,(bias-normal[0]*self.t)/(normal[1]+1e-10),color)
            plt.scatter(xc[0],xc[1],s=50,c='c',marker='x')

            circle = Circle(xc,radius=R,alpha=0.3,color='c')

            ax = plt.gca()
            ax.add_patch(circle)

            plt.axis('equal')
            plt.grid(True)
            plt.xlim(self.xrange)
            plt.ylim(self.yrange)
            plt.pause(self.pause_len)

        return iflag

    def intersects(self,A,b,a_test,b_test):
        """
        given polytope defined by Ax <= b for m x n numpy array A and length-n
        numpy vector b (denoted by set P), returns 0 if a_test^T x = b_test
        intersects this set for length-n numpy vector a_test and scalar b_test,
        -1 if a_test.dot(x) <= b_test for any x in P, and 1 otherwise.
        This intersection will occur if and only if both [A;a_test^T]x <=
        [b;b_test] and [A;-a_test^T]x <= [b;-b_test] are both non-empty sets,
        checked with a linear program. Note that Ax <= b is assumed to be a
        bounded, non-empty set
        """

        _,stat1 = cvxlp(c=np.zeros(A.shape[1]),A_ub=np.vstack((A,a_test)),
            b_ub=np.append(b,b_test))
        _,stat2 = cvxlp(c=np.zeros(A.shape[1]),A_ub=np.vstack((A,-a_test)),
            b_ub=np.append(b,-b_test))

        suc1 = (stat1 == 'optimal')
        suc2 = (stat2 == 'optimal')

        if suc1:
            if suc2:
                return 0
            else:
                return -1
        elif suc2:
            return 1
        else:
            print("""Constraint set appears to be empty!
                Declaring query as ambiguous""")
            return 0

    def separator(self, a, b):
        """
        Produces hyperplane that bisects the line from a to b is
            orthogonal to it
        a, b: np.array - 1 x d, the two points being separated.
        return: (np.array, np.array, np.array)- normal and bias of hyperplane,
            and midpoint of query
        suh that normal.dot(midpoint) = bias
        """
        normal = (b - a) / np.sqrt(sum((b-a)**2))
        assert(abs(np.linalg.norm(normal) - 1) < 1e-3)
        midpoint = a + (b-a)/2.
        bias = midpoint.dot(normal)

        return normal, bias, midpoint

    def append(self, embedding, i, j):
        """
        Adds a binary constraint to the feasible region
        embedding: np.array - an N x d embedding of points
        i, j: int - sorted indices for two points in a binary ranking query,
            where rank_i < rank_j
        return: self for posterity
        """
        normal, bias, midpoint = self.separator(embedding[j],embedding[i])

        # assert that added constraints be consistent with existing ones
        assert np.sign(np.dot(embedding[i] - midpoint, normal)) == np.sign(1)

        if self.A is None:
            self.A = np.empty((0,len(normal)))
            self.b = np.empty((0,))

        self.A = np.vstack((self.A,-normal))
        self.b = np.append(self.b,-bias)


def cvxlp(c=None,A_ub=None,b_ub=None):
    """
    solves the following linear program
    min_x c^T x
    s.t.
    A_ub*x <= b_ub

    inputs: c, A_ub, b_ub
    output: cvxopt solution as numpy vector, cvxopt status
    """

    c_cvx = cvx.matrix(c)
    A_cvx = cvx.matrix(A_ub.astype(np.double))
    b_cvx = cvx.matrix(b_ub.astype(np.double).reshape((b_ub.size,1)))

    sol = cvx.solvers.lp(c_cvx,A_cvx,b_cvx)
    if sol['x'] is not None:
        x_opt = np.array(sol['x'])
        x_opt = x_opt.flatten()
    else:
        x_opt = None

    return x_opt, sol['status']


class NoiseModel(Enum):
    # enumerate noise model types
    BT = 0
    NONE = 1

def main():
    """
    Example usage of ActRankQ:

    - generates random embedding of items
    - defines oracle
    - initalize ranking object
    - get paired comparison queries
    """

    N = 100 # number of items
    d = 2 # embedding dimension
    max_query = 100 # number of queries to ask
    nvotes = 1 # number of votes in ActRankQ

    embedding = np.random.randn(N, d) # generate embedding

    print("Search points: ")
    print(embedding)

    bounds = np.array([-1,1]) # define user point prior
    ref = np.random.uniform(bounds[0], bounds[1], (d,1))

    print("Reference point: ")
    print(ref)

    k = 10 # specify noise constant value
    k_normalization = KNormalizationType.CONSTANT # specify noise constant type
    noise_model = NoiseModel.NONE # specify noise model

    def oracle(p):
        # if y=1, then p[0] selected

        (a, tau) = pair2hyperplane(p, embedding, k_normalization)
        z = np.dot(a, ref) - tau

        if noise_model == NoiseModel.BT:
            y = int(np.random.binomial(1, sp.special.expit(k * z)))
        else:
            y = int(z > 0)

        return {'y':y,'z':z,'a':a,'tau':tau}

    correct_ranking = sorted(range(len(embedding)), key=lambda x:
        np.linalg.norm(embedding[x] - ref.T))

    ranker = ActRankQ() # construct ranker
    ranker.initialize(embedding, nvotes, bounds=bounds,
        ref=ref, plot=False, debug=True, pause_len=0.01) # initialize ranker

    queries_made = 0
    while queries_made < max_query:
        query,response = ranker.getQuery(oracle) # get query, pass oracle

        if query is None:
            break

        queries_made += 1
        print('# queries made: {} / {}'.format(queries_made,max_query))

    # get stats from ranker, including learned ranking of items
    oracle_queries, full_ranking, ambiguous_q, total_q = ranker.getStats()

    print("learned ranking: ", full_ranking)
    print("correct ranking: ", correct_ranking)
    print("Oracle was asked {} out of {} considered queries.".format(
        len(oracle_queries), total_q))
    print("Ordering correctness: ",
        np.array_equal(correct_ranking, full_ranking))


if __name__ == '__main__':
    main()
