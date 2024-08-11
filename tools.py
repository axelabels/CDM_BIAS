import sys
from scipy import stats
import math
import re
import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

from itertools import chain, combinations

def greedy_choice(a, axis=None):
    """
    Selects actions greedily based on their scores, creating a probability distribution that favors the maximum values.

    Args:
        a (numpy.ndarray): Array of scores from which to select actions.
        axis (int, optional): Axis along which to perform the operation.

    Returns:
        numpy.ndarray: A probability distribution where the highest scores receive the highest probabilities.
    """
    max_values = np.amax(a, axis=axis, keepdims=True)
    choices = (a == max_values).astype(np.float)
    return choices / np.sum(choices, axis=axis, keepdims=True)


def get_noisy_advice(truth, dist):
    """
    Generates noisy advice based on the truth values and a distortion factor, see Section 5.1.1. Used primarily for regression problems.

    Args:
        truth (numpy.ndarray): True values from which to generate advice.
        dist (float): Distortion factor that affects how much noise to add.

    Returns:
        numpy.ndarray: Noisy advice derived from the truth values.
    """
    eta = np.random.uniform(0, 1, size=truth.shape)

    w_1 = max(0, (1-2*dist))
    w_2 = max(0, (2*dist-1))
    w_3 = 1-w_1-w_2
    return truth * w_1 + (1-truth)*w_2 + eta*w_3


eps = 1e-15
def KL(x, y):
    """
    Computes the Kullback-Leibler divergence between two probabilities, x and y.

    Args:
        x (float): The probability of success for the first distribution.
        y (float): The probability of success for the second distribution.

    Returns:
        float: The KL divergence between the two distributions.
    """
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def generate_noisy_advice(bandit, agents, experts, n_trials, seed,  expertise_distribution,  maximum_error, within_cluster_correlation):
    """
    Generates noisy advice for a set of experts based on predefined distributions and errors. Used in simulations to test agents' performance under different noise conditions.

    Args:
        bandit (ArtificialBandit): Bandit environment providing contexts and expected rewards.
        agents (list): List of agents participating in the simulation.
        experts (list): List of experts whose advice is being simulated.
        n_trials (int): Number of trials for which to generate advice.
        seed (int): Seed for random number generation to ensure reproducibility.
        expertise_distribution (str): Type of distribution for expertise levels among experts.
        maximum_error (float): Maximum error in the expert advice.
        within_cluster_correlation (float): Correlation within clusters of experts.

    Effects:
        Modifies internal states of bandit and agents by setting up conditions and providing noisy advice.
    """

    np.random.seed(seed)
    n_experts = len(experts)
    # reset everything
    bandit.reset()
    for actor in agents+experts:
        actor.reset()

    # initialize bandit
    bandit.cache_contexts(n_trials, seed)
    expected_rewards = bandit.cached_values.flatten()

    # see 5.1.2
    if expertise_distribution == "heterogeneous":
        expert_errors = np.linspace(0, maximum_error, n_experts)
    else:  # homogeneous
        expert_errors = np.zeros(n_experts)+maximum_error

    if bandit.problem == 'classification':
        expert_errors = (expert_errors*.8+.2)
    else:
        expert_errors = (expert_errors*.6+.2)

    advice = np.repeat(bandit.cached_values[None], n_experts, axis=0)

    # Generate noisy advice for each cluster of experts
    for cluster_idx in range(2):
        c = 0.25  # proportion of experts contained by the first cluster, see Section 5.1.2
        if cluster_idx == 0:
            # experts within the cluster
            cluster_expert_idxs = range(0, int(np.round(n_experts*c)))
            common_var = expert_errors[0]
        else:
            cluster_expert_idxs = range(int(np.round(n_experts*c)), n_experts)
            common_var = expert_errors[-1]

        # cfr. section 5.1.2
        if bandit.problem == 'classification':
            common_error_idxs = np.random.choice(len(bandit.cached_values), size=int(
                len(bandit.cached_values)*common_var), replace=False)
            shared_advice = bandit.cached_values + 0
            shared_advice[common_error_idxs] = bandit.generate_random_values(
                bandit.cached_values.shape)[common_error_idxs]
        else:
            shared_advice = get_noisy_advice(  # see Section 5.1.1
                expected_rewards, common_var).reshape(bandit.cached_values.shape)

        # generate noisy advice for each expert within the cluster
        for n in cluster_expert_idxs:

            # noisy advice procedure, see Section 5.1.1
            if bandit.problem == 'classification':

                # indices for which the expert predicts the wrong class
                error_idxs = np.random.choice(len(bandit.cached_values), size=int(
                    len(bandit.cached_values)*expert_errors[n]), replace=False)

                advice[n, error_idxs] = bandit.generate_random_values(
                    bandit.cached_values.shape)[error_idxs]

                # indices for which the expert agrees with its cluster's common beliefs (cfr. section 5.1.2)
                herding_idxs = np.random.choice(n_trials,
                                                size=int(len(bandit.cached_values)*(within_cluster_correlation)), replace=False)
                advice[n, herding_idxs] = shared_advice[herding_idxs]+0
            else:
                advice[n] = get_noisy_advice(  # see Section 5.1.1
                    expected_rewards, expert_errors[n]).reshape(bandit.cached_values.shape)

                # indices for which the expert agrees with its cluster's common beliefs (cfr. section 5.1.2)
                is_herded = np.random.choice([0, 1], p=[
                                             1-within_cluster_correlation, within_cluster_correlation], size=advice[n].shape)
                advice[n] = advice[n] * \
                    (1-is_herded) + shared_advice*is_herded

    for i, e in (enumerate(experts)):
        e.cache_predictions(bandit, n_trials)
        e.cached_predictions[:] = advice[i]


def df_to_sarray(df):
    """
    Converts a pandas DataFrame to a numpy structured array, encoding string columns to bytes.

    Args:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        numpy.ndarray: A structured array representation of the DataFrame.
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int)
                    col_type = 'S%s' % maxlen  # ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values
    types = df.dtypes
    numpy_struct_types = [make_col_type(
        types[col], df.loc[:, col]) for col in df.columns]
    # print(numpy_struct_types)
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype


    """
    Converts a numpy array into a one-line string for easier logging or printing.

    Args:
        array (numpy.ndarray): Array to be converted.

    Returns:
        str: A string representation of the array, compressed to one line.
    """
    return re.sub(r'\s+', ' ',
                  str(array).replace('\r', '').replace('\n', '').replace(
                      "array", "").replace("\t", " "))


def safe_logit(p, eps=1e-6):
    """
    Computes the logit (log-odds) of a probability, ensuring no division by zero or log of zero occurs.

    Args:
        p (float): Probability value.
        eps (float): Small constant to ensure stability near probability bounds of 0 and 1.

    Returns:
        float: The logit of the probability.
    """
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))


def SMInv(Ainv, u, v,):
    """
    Updates the inverse of a matrix A using the Sherman-Morrison formula

    Args:
        Ainv (numpy.ndarray): Current inverse of matrix A.
        u (numpy.ndarray): Column vector u in the rank-1 update.
        v (numpy.ndarray): Row vector v in the rank-1 update.

    Returns:
        numpy.ndarray: Updated inverse of matrix A after applying the rank-1 update.
    """
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))
    
    return Ainv - np.dot(Ainv, np.dot(np.dot(u, v.T), Ainv)) / (1 + np.dot(v.T, np.dot(Ainv, u)))


def randargmax(b, **kw):
    """
    Returns the index of the maximum value in an array. If multiple maxima, returns one randomly.

    Args:
        b (numpy.ndarray): Array from which to find the index of the maximum value.
        **kw: Additional keyword arguments for numpy's argmax function.

    Returns:
        int: Index of the maximum value, chosen randomly if multiple maxima exist.
    """
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)



def powerset(iterable):
    """
    Generates the power set of a given iterable, which is a set of all possible subsets, including the empty set and the set itself.

    Args:
        iterable (iterable): An iterable (e.g., list, set, tuple) from which to generate the power set.

    Returns:
        itertools.chain: An iterator over all subsets of the iterable. Each subset is returned as a tuple, which is an immutable ordered list of elements.

    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
