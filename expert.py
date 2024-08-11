from __future__ import print_function

from math import ceil
# from bandit import OffsetBandit
from scipy.stats import gmean

from sklearn.gaussian_process.kernels import RationalQuadratic, PairwiseKernel


from policy import *
from tools import *


EXPECTED_STD_REWARD = np.float64(.5)


MAX_SLICE_SIZE = 100
RANDOM_CONFIDENCE = .5


class Agent(object):
    """
    Represents a generic agent interacting with a bandit problem, utilizing a specified policy for decision making.

    Attributes:
        bandit (Bandit): The bandit problem instance the agent will interact with.
        policy (Policy): The decision-making policy based on aggregated advice or predictions.
        name (str, optional): A name for the agent to identify it in logs or output.
    """

    def __init__(self, bandit, policy, name=None):
        self.bandit = bandit
        self.policy = policy

        self.prior_bandit = None

        self.t = 0
        self.reward_history = []
        self.context_history = []
        self.action_history = []
        self.cache_id = None

        self.name = name

    def get_advice(self, contexts=None, cache_index=None, return_std=False, arm=None, batch=True):
        """
        Retrieves advice for the current contexts or based on a cached index.

        Args:
            contexts (numpy.ndarray, optional): Contexts for which to get advice.
            cache_index (int, optional): Index to retrieve cached predictions.
            return_std (bool, optional): Flag indicating whether to return standard deviations.
            arm (int, optional): Specific arm for which to retrieve advice.
            batch (bool): Whether the advice is for batch processing.

        Returns:
            numpy.ndarray: Advice or predictions for the contexts.
        """
        if cache_index is not None:
            assert self.is_prepared(
            ), "When an int is given as context the expert should be prepared in advance"
            self.mu = np.copy(self.cached_predictions[cache_index])
        else:
            self.mu = self.predict_normalized(
                contexts, arm=arm, batch=batch)
   
        return self.mu

    def is_prepared(self, cache_id=None):
        """
        Checks if the agent's cached data is prepared and matches the specified cache identifier.

        Args:
            cache_id (any, optional): The cache identifier to check against the agent's current cache id.

        Returns:
            bool: True if prepared, False otherwise.
        """
        if cache_id is None:
            return self.cache_id == self.bandit.cache_id
        return self.cache_id == cache_id


    def reset(self):
        self.t = 0
        self.reward_history = []
        self.context_history = []
        self.action_history = []
        self.cache_id = None

    def predict_normalized(self, contexts, slice_size=None, arm=None, batch=False):
        """
        Generates normalized predictions for given contexts.

        Args:
            contexts (numpy.ndarray): Contexts to predict.
            slice_size (int, optional): Size of context slices to process (unused placeholder).
            arm (int, optional): Specific arm to predict.
            batch (bool, optional): If True, processes predictions in batches.

        Returns:
            numpy.ndarray: Normalized predictions for the contexts.
        """
        assert np.shape(contexts)[1:] == (self.bandit.dims,)
        
        mu = self.bandit.cached_values 
        if arm is not None:
            mu = mu[..., arm]
        return mu

    def cache_predictions(self, bandit, trials):
        """
        Caches predictions for a specified number of trials.

        Args:
            bandit (ArtificialBandit): The bandit for which predictions are cached.
            trials (int): Number of trials for which to cache predictions.

        Returns:
            bool: True if predictions were recomputed, False otherwise.
        """
        if not self.is_prepared(bandit.cache_id) or len(self.cached_predictions) < trials:
            self.cached_predictions = np.array(
                self.predict_normalized(bandit.cached_contexts,batch=True))

            assert np.shape(self.cached_predictions) == (
                trials, bandit.k), np.shape(self.cached_predictions)

            self.cache_id = bandit.cache_id
            recomputed = True
       
        return recomputed

Expert = Agent