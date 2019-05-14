# Active Embedding Search via Noisy Paired Comparisons

Project code for "Active Embedding Search via Noisy Paired Comparisons" ([ICML 2019](https://arxiv.org/abs/1905.04363))

## Paper abstract
Suppose that we wish to estimate a user's preference vector w from paired comparisons of the form "does user w prefer item p or item q?," where both the user and items are embedded in a low-dimensional Euclidean space with distances that reflect user and item similarities. Such observations arise in numerous settings, including psychometrics and psychology experiments, search tasks, advertising, and recommender systems. In such tasks, queries can be extremely costly and subject to varying levels of response noise; thus, we aim to actively choose pairs that are most informative given the results of previous comparisons. We provide new theoretical insights into the benefits and challenges of greedy information maximization in this setting, and develop two novel strategies that maximize lower bounds on information gain and are simpler to analyze and compute respectively. We use simulated responses from a real-world dataset to validate our strategies through their similar performance to greedy information maximization, and their superior preference estimation over state-of-the-art selection methods as well as random queries.

## Requirements
- [Python 3](https://www.python.org/downloads/)
- [NumPy](https://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [PyStan](https://pystan.readthedocs.io/en/latest/)
- [CVXOPT](https://cvxopt.org/)

## Code
- `active_search.py`: module for proposed InfoGain, EPMV, and MCMV search methods.
- `actrankq.py`: module for ActRankQ, our implementation of an [active ranking baseline](https://papers.nips.cc/paper/4427-active-ranking-using-pairwise-comparisons.pdf).
