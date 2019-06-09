# Active Embedding Search via Noisy Paired Comparisons

Project code for "Active Embedding Search via Noisy Paired Comparisons"
([ICML 2019](https://arxiv.org/abs/1905.04363)) by Gregory H. Canal,
Andrew K. Massimino, Mark A. Davenport, Christopher J. Rozell.

## Paper abstract
Suppose that we wish to estimate a user's preference vector w from paired comparisons of the form "does user w prefer item p or item q?," where both the user and items are embedded in a low-dimensional Euclidean space with distances that reflect user and item similarities. Such observations arise in numerous settings, including psychometrics and psychology experiments, search tasks, advertising, and recommender systems. In such tasks, queries can be extremely costly and subject to varying levels of response noise; thus, we aim to actively choose pairs that are most informative given the results of previous comparisons. We provide new theoretical insights into the benefits and challenges of greedy information maximization in this setting, and develop two novel strategies that maximize lower bounds on information gain and are simpler to analyze and compute respectively. We use simulated responses from a real-world dataset to validate our strategies through their similar performance to greedy information maximization, and their superior preference estimation over state-of-the-art selection methods as well as random queries.

## Requirements

### Packages (python)
- [Python 3](https://www.python.org/downloads/)
- [NumPy](https://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyStan](https://pystan.readthedocs.io/en/latest/)
- [CVXOPT](https://cvxopt.org/)

### Data

`run_experiments.py` uses our pre-processed embedding files contained in
the sub-directory `make-embedding/data/`.  It is not necessary to download
the original dataset in this case.

To build an embedding using the scripts in `make-embedding/` or to
pre-process an embedding using `process_embedding.py` it is necessary to
have the Food-10k dataset of triplets.  This dataset may be obtained from
the [SE(3) Computer Vision Group at Cornell Tech](https://vision.cornell.edu/se3/concept-embeddings/).

### Packages (MATLAB only for Enusvm/GaussCloud method baseline)
- [CVX](cvxr.com/cvx/)

## Code
- `active_search.py`: module for proposed InfoGain, EPMV, and MCMV search methods.
- `actrankq.py`: module for ActRankQ, our implementation of an [active ranking baseline](https://papers.nips.cc/paper/4427-active-ranking-using-pairwise-comparisons.pdf).
- `run_experiments.py`: script to run paper experiments.
- `process_embedding.py`: script to estimate noise constant and embedding scaling from embedding file
- `make-embedding/*.py`: scripts to generate embedding from human intelligence task sourced triplets.
- `enusvm/*.m`: implementation and simulation for the "GaussCloud" [baseline method](https://arxiv.org/abs/1802.10489).

Please send correspondence to Greg Canal (gregory.canal@gatech.edu) and Andrew Massimino (massimino@gatech.edu)
