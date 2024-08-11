
import numpy as np

from tools import greedy_choice, softmax


class Policy(object):
    """
    Base class for policies. This class provides methods for determining
    the action probabilities and choosing an action based on those probabilities.

    Methods:
        probabilities(agent, contexts): Computes the probabilities for selecting each arm based on the given contexts.
        choose(agent, contexts, greedy): Chooses an arm based on the probabilities and a potential greedy choice.
    """

    def probabilities(self, agent, contexts):
        """
        Calculates uniform probabilities across all arms.

        Args:
            agent (Agent): The agent applying the policy.
            contexts (any): The current contexts based on which actions are evaluated.

        Returns:
            numpy.ndarray: A uniform probability distribution over the arms.
        """
        self.pi = np.ones(agent.bandit.k)/agent.bandit.k
        return self.pi

    def choose(self, agent, contexts, greedy=False):
        """
        Chooses an arm based on computed probabilities. Optionally makes a greedy choice.

        Args:
            agent (Agent): The agent applying the policy.
            contexts (any): The current contexts based on which actions are evaluated.
            greedy (bool): If True, forces a greedy choice from the probabilities.

        Returns:
            int: The index of the chosen arm.
        """
        self.pi = self.probabilities(agent, contexts)

        if greedy:
            self.pi = greedy_choice(self.pi)
        np.testing.assert_allclose(np.sum(
            self.pi), 1, atol=1e-5, err_msg=str(agent)+" "+str(np.sum(self.pi))+" "+str(self.pi))
        action = np.searchsorted(np.cumsum(self.pi), np.random.rand(1))[0]

        return action


class EpsilonGreedyPolicy(Policy):
    """
    Introduces exploration through a probability epsilon of choosing
    a random action

    Attributes:
        epsilon (float): Probability of taking a random action.
    """


    def __init__(self, epsilon):
        self.epsilon = epsilon

    def probabilities(self, agent, contexts):
        """
        Computes the probabilities for each arm, predominantly selecting the best arm with a probability of (1 - epsilon)
        and all others with a probability of epsilon divided by the number of arms.

        Args:
            agent (Agent): The agent applying the policy.
            contexts (any): The current contexts used to estimate rewards.

        Returns:
            numpy.ndarray: Adjusted probabilities incorporating epsilon-greedy strategy.
        """
        v = agent.estimate_rewards(contexts)
        self.pi = greedy_choice(v)
        self.pi *= (1-self.epsilon)
        self.pi += self.epsilon/agent.bandit.k

        return self.pi


class GreedyPolicy(EpsilonGreedyPolicy):

    def __init__(self):
        super().__init__(0)


class Exp3Policy(Policy):
    """
    EXP3 Policy for adversarial bandit problems with a modification parameter eps to mix exploration uniformly.

    Attributes:
        eps (float): Exploration parameter to ensure non-zero probabilities for all arms.
    """
    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'

    def probabilities(self, agent, contexts):
        """
        Computes probabilities by mixing the estimated values with a uniform distribution scaled by eps.

        Args:
            agent (Agent): The agent applying the policy.
            contexts (any): The current contexts used to estimate probabilities.

        Returns:
            numpy.ndarray: Probabilities adjusted for exploration.
        """
        self.pi = agent.probabilities(contexts)

        self.pi = self.pi * (1 - self.eps) + self.eps / len(self.pi)
        return self.pi


class SCBPolicy(Policy):
    """
    SCB (Square Confidence Bound) Policy, which implements the probability selection scheme as described by
    Abe and Long (1999). The policy computes probabilities inversely proportional to the squared gap
    between each action's score and the lowest score, promoting a balance between exploration and exploitation.

    Attributes:
        gamma (float): Tuning parameter that affects the weight of the confidence bounds in the probability calculation.
    """

    def __init__(self, gamma=0):
        self.gamma = gamma
        self.key = 'probability'

    def probabilities(self, agent, contexts):
        """
        Computes the probabilities for selecting each arm using a method where each action’s probability
        is inversely proportional to the square of the gap between the action’s score and the lowest score.
        This approach ensures more exploration for actions with scores close to the minimum.

        Args:
            agent (Agent): The agent applying the policy.
            contexts (any): The current contexts used to estimate rewards.

        Returns:
            numpy.ndarray: Probabilities for each arm, adjusted to account for the relative scores of actions.
        """

        values = agent.estimate_rewards(contexts)
        best_arm = np.argmax(values)
        self.pi = np.zeros_like(values)
        self.pi[:] = 1 / \
            (agent.bandit.k+self.gamma*(values[best_arm]-values))
        self.pi[best_arm] += (1-(np.sum(self.pi)))

        return self.pi
