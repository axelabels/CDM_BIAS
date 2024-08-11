
import numpy as np
from tools import *


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def shape_rewards(a, x):
    return a**x


class ArtificialBandit():
    """
    Simulates an artificial bandit problem for testing bandit algorithms. This class can simulate both classification
    and regression problems with options for rewards to be generated from a Bernoulli distribution.

    Attributes:
        k (int): Number of arms in the bandit problem.
        problem (str): Type of problem ('classification' or 'regression').
        bernoulli (bool): Determines if rewards are generated from a Bernoulli distribution.
        dims (int): Number of dimensions for context vectors.
    """

    def __init__(self, n_arms=1, problem='classification',  bernoulli=True):
        """
        Initializes the artificial bandit with specified parameters.

        Args:
            n_arms (int): Number of arms for the bandit.
            problem (str): Specifies whether the bandit problem is a classification or regression problem.
            bernoulli (bool): If True, rewards are sampled from a Bernoulli distribution based on the probabilities.
        """
        self.k = n_arms
        self.problem = problem
        self.bernoulli = bernoulli
        self.cached_contexts = None
        self.dims = 1

    @property
    def expected_reward(self):
        """
        Calculates the expected reward for the bandit based on the problem type.

        Returns:
            float: The expected reward value.
        """
        return .5 if self.problem == 'regression' else 1/self.k

    def reset(self):
        self.cached_contexts = None

    def generate_random_values(self, shape):
        """
        Generates random values for contexts or action values based on the problem type.

        Args:
            shape (tuple): The shape of the array to generate.

        Returns:
            numpy.ndarray: Randomly generated values for contexts or actions.
        """
        if self.problem == 'regression':
            values = np.repeat(np.arange(self.k)[None], shape[0], axis=0)
            values = shuffle_along_axis(values, axis=1)
            values = values/(self.k-1) * .8 + .1
            return values
        else:
            values = np.repeat(np.arange(self.k)[None], shape[0], axis=0)
            values = shuffle_along_axis(values, axis=1)
            values = softmax(np.log(10)*values, axis=1)
            return values

    def cache_contexts(self, t, cache_id):
        """
        Caches generated contexts for the bandit problem for efficient retrieval during simulations.

        Args:
            t (int): The number of contexts to generate and cache.
            cache_id (any): An identifier used for the cache.

        Returns:
            numpy.ndarray: The cached contexts.
        """
        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.random.uniform(
                0, 1, size=(t,  self.dims))

            self.cached_values = self.generate_random_values((t, self.k))
            if self.problem == 'classification':

                self.cached_values = np.repeat(
                    np.arange(self.k)[None], t, axis=0)

                self.cached_values = shuffle_along_axis(
                    self.cached_values, axis=1)

                self.cached_values = softmax(
                    np.log(10)*self.cached_values, axis=1)

            assert np.shape(self.cached_values) == (
                t, self.k), (np.shape(self.cached_values), (t, self.k))
            self.cached_rewards = self.sample(self.cached_values)

            assert np.shape(self.cached_rewards) == (t, self.k)
            self.cache_id = cache_id

        return self.cached_contexts

    def observe_contexts(self, cache_index):
        """
        Retrieves a specific context from the cache based on the index.

        Args:
            cache_index (int, optional): The index of the context to retrieve.

        Returns:
            numpy.ndarray: The specific context requested.
        """

        self.contexts = self.cached_contexts[cache_index]
        self.action_values = self.cached_values[cache_index]

        self.optimal_value = np.max(self.action_values)

        return self.contexts

    def sample(self, values=None, cache_index=None):
        """
        Samples rewards based on the specified values or cached values at a particular index.

        Args:
            values (numpy.ndarray, optional): Values to sample rewards from.
            cache_index (int, optional): Index to retrieve cached rewards.

        Returns:
            numpy.ndarray: Sampled rewards.
        """
        if cache_index is not None:
            return self.cached_rewards[cache_index]

        if values is None:
            values = self.action_values
        if self.bernoulli:

            return np.random.uniform(size=np.shape(values)) < values
        else:
            return values
