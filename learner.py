
from copy import deepcopy
from numbers import Number
from sklearn.linear_model import Ridge
from expert import *

EXPECTED_AVG_REWARD = .5


class Average(Agent):
    """
    Implements a basic agent that averages expert advice to make decisions on bandit problems.

    Attributes:
        bandit (Bandit): Instance of a Bandit problem.
        policy (Policy): Policy used to determine action based on aggregated advice.
        n_experts (int): Number of experts providing advice.
        name (str, optional): Human-readable name for the agent.
    """

    def __init__(self, bandit, policy, n_experts,   name=None,
                 ):
        """
        Initializes the Average agent with necessary parameters and defaults.

        Args:
            bandit (Bandit): The bandit problem instance this agent will interact with.
            policy (Policy): The decision-making policy based on aggregated advice.
            n_experts (int): The number of experts that provide advice.
            name (str, optional): Overrides human-readable name
        """
        super(Average, self).__init__(bandit, policy)
        self.k = self.bandit.k
        self.n = self.n_experts = n_experts
        self.name = name
        self.advice = np.zeros((self.n, self.bandit.k))
        self._reward_estimates = np.zeros(self.k)
        self._probabilities = np.zeros(self.k)

        self.confidences = np.ones((self.n, self.bandit.k))

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize learner's parameters.
        """
        pass

    def short_name(self):
        return "Average"

    def observe(self, reward, arm):
        """
        Update the learner's state based on the observed reward and chosen arm.

        Args:
            reward (float): The reward received from the bandit after playing an arm.
            arm (int): The index of the arm that was played.
        """
        self.t += 1

    def get_weights(self, contexts):
        """
        Return the weights assigned to each expert

        Args:
            contexts (dict): Contextual information about the current state

        Returns:
            np.ndarray: A matrix of weights for expert advice.
        """
        return np.ones((self.n, self.bandit.k))/self.n

    def __str__(self):
        if self.name is not None:
            return (self.name)
        return (self.short_name())

    def set_name(self, name):
        self.name = name

    def choose(self, advice):
        """
        Select an arm to play based on given advice

        Args:
            advice (np.ndarray): Expert advice provided for each arm.

        Returns:
            int: The index of the chosen arm.
        """
        return self.policy.choose(self, advice)

    def estimate_rewards(self, contexts):
        """
        Estimate the expected reward based on current context (de facto the advice)

        Args:
            contexts (dict): Contextual information including current expert advice.

        Returns:
            np.ndarray: An array of estimated expected reward for each arm.
        """
        self.advice = np.copy(contexts['advice'])

        self._reward_estimates = np.sum(
            self.get_weights(contexts) * (self.advice - self.bandit.expected_reward), axis=0)

        return self._reward_estimates

    def reset(self):
        """
        Reset the internal state of the agent, use at the start of a new simulation.
        """
        super().reset()
        self.initialize_parameters()


class Exp4(Average):
    """
    Implements the EXP4 algorithm, an extension of the EXP3 algorithm, designed for adversarial bandit problems
    with expert advice. It uses the exponential weighting strategy to aggregate advice and make decisions.

    Attributes:
        gamma (float): Exploration parameter, controlling the trade-off between exploration and exploitation.
    """

    def __init__(self, bandit, policy, n_experts, gamma, name=None,):
        """
        Initializes the EXP4 agent with the bandit problem, policy, number of experts, and exploration parameter.

        Args:
            bandit (Bandit): The bandit problem instance this agent will interact with.
            policy (Policy): The decision-making policy based on aggregated advice.
            n_experts (int): The number of experts that provide advice.
            gamma (float): Exploration parameter (and learning rate).
            name (str, optional): Optional name for the agent.
        """
        super(Exp4, self).__init__(bandit, policy,
                                   n_experts,  name=name, )
        self.gamma = gamma
        self.e_sum = 1
        self.w = np.zeros(self.n)

    def probabilities(self, contexts):
        """
        Computes the probabilities of choosing each arm based on expert advice and current weights.

        Args:
            contexts (dict): Contextual information including expert advice for each arm.

        Returns:
            np.ndarray: Probabilities of choosing each arm.
        """
        self.advice = np.copy(contexts['advice'])
        assert self.advice.shape == (self.n_experts, self.bandit.k,)
        if not np.allclose(np.sum(self.advice, axis=1), 1):
            self.advice = greedy_choice(self.advice, axis=1)

        w = self.get_weights(contexts)
        self._probabilities = np.sum(w * self.advice, axis=0)

        return self._probabilities

    def short_name(self):
        return "EXP4-IX"

    def initialize_parameters(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.zeros(self.n)

    def reset(self):
        super(Exp4, self).reset()

    def get_weights(self, contexts):
        """
        Computes weights for each expert based on their past performance and current contexts.

        Args:
            contexts (dict): Contextual information that may affect weight calculation.

        Returns:
            np.ndarray: A matrix of weights for each expert across each arm.
        """
        w = np.repeat(self.w[:, np.newaxis], self.bandit.k, axis=1)
        w = softmax(w, axis=0)

        return w

    def observe(self, reward, arm):
        """
        Updates weights based on the observed reward and the arm that was played.

        Args:
            reward (float): The reward received from the bandit after playing an arm.
            arm (int): The index of the arm that was played.
        """
        assert np.allclose(np.sum(self.advice, axis=1),
                           1), "expected probability advice"

        x_t = self.advice[:, arm] * (1-reward)
        y_t = x_t / (self.policy.pi[arm]+self.gamma)

        self.e_sum += np.sum(np.max(self.advice, axis=0))

        numerator = self.e_sum
        lr = np.sqrt(np.log(self.n) / numerator)
        self.w -= (lr) * y_t

        self.t += 1


class SafeFalcon(Average):
    """
    Implements the SafeFalcon algorithm, which is a variant of the FALCON method with safety checks.
    This algorithm incorporates mechanisms to ensure decisions are made based on reliable estimates and
    maintains safety margins based on statistical confidence levels.

    Attributes:
        n_trials (int): Total number of trials or epochs the algorithm is expected to run.
        beta (float): Parameter used in calculating the gamma value for exploration/exploitation balance.
        alpha (float): Regularization strength of the Ridge regression model used for predictions.
        n_experts (int): Number of experts providing advice.
    """

    def __init__(self, bandit, policy, n_experts, n_trials, name=None,
                 alpha=1,  beta=20,):
        """
        Initializes the SafeFalcon agent with specified parameters and dependencies.

        Args:
            bandit (Bandit): The bandit problem instance this agent will interact with.
            policy (Policy): The decision-making policy based on aggregated advice.
            n_experts (int): The number of experts providing advice.
            n_trials (int): Total number of trials or epochs for the agent.
            name (str, optional): Optional name for the agent.
            alpha (float): Regularization parameter for the Ridge regression.
            beta (float): Parameter for calculating exploration balance.
        """
        self.e_sum = 1
        self.n_trials = n_trials
        self.beta = beta
        self.alpha = alpha
        self.n_experts = n_experts
        super().__init__(bandit, policy,
                         n_experts,  name=name, )

    def short_name(self):
        return f"Meta-CMAB (FALCON)"

    def initialize_parameters(self):
        """
        Initializes or resets parameters and statistical models necessary for the SafeFalcon algorithm.
        """
        self.e_sum = 1
        self.context_history = []
        self.reward_history = []
        self.model = Ridge(alpha=self.alpha, fit_intercept=False)
        self.epoch_ends = 2**(np.arange(1, np.log(self.n_trials) /
                              np.log(2))).astype(int)
        self.t = 0
        self.delta = np.log(self.n_trials) / self.n_trials

        self.delta_prime = self.delta/13

        self.prev_l = 0
        self.hat_m = 0
        self.model_history = []
        self.cum_reward = 0
        self.safe = True

    def choose_safe(self, m, reward_history, l):
        """
        Determines whether the next step is safe based on historical reward data and the previous safe level.

        Args:
            m (int): Current epoch or time period number.
            reward_history (list): List of past rewards observed.
            l (float): Previous level of safety or reward threshold.

        Returns:
            tuple: Updated safe level and model index if safety condition is adjusted.
        """
        sub = np.sqrt(1/(2*len(reward_history))*np.log(m**2/self.delta))

        l_prime = np.mean(reward_history-sub+self.bandit.expected_reward)
        lm = max(l, l_prime)
        if lm != l:
            self.hat_m = m

        return lm, self.hat_m

    def e_rate(self, T, e):
        er = (self.n_experts+np.log(1/e))/T  # (d+log(1/eps))/T
        return er

    def get_epoch(self, t):
        return np.searchsorted(self.epoch_ends, t, side='right')+1

    def get_epoch_end(self, m):
        return self.epoch_ends[m-1]

    def get_epoch_start(self, m):
        if m == 1:
            return 0
        return self.epoch_ends[m-2]

    def epoch_length(self, epoch):
        return self.get_epoch_end(epoch)-self.get_epoch_start(epoch)

    def check_is_safe(self, m, t, l, crwd):
        """
        Checks whether the current strategy is safe based on cumulative rewards and other safety metrics. 
        In essence, the current strategy is safe if performance is not degrading

        Args:
            m (int): Current epoch number.
            t (int): Current time step.
            l (float): Last known safe reward level.
            crwd (float): Cumulative reward.

        Returns:
            bool: True if the current strategy is considered safe, False otherwise.
        """
        if (np.log(t+1-self.get_epoch_end(m-1))/np.log(2) % 1) == 0 or (t <= self.epoch_ends[-1] and t == self.get_epoch_end(m)):
            tau1 = self.get_epoch_end(1)
            tau2 = self.get_epoch_end(2)

            L = t*l - tau1 - \
                np.sqrt(
                    2*t*np.log((np.ceil(m + np.log(tau1)/np.log(2))**3)/self.delta_prime))
            assert not np.isnan(L)

            roots = np.sqrt([self.e_rate(self.epoch_length(self.get_epoch(
                i)-1), self.delta_prime/(self.get_epoch(i)**2)) for i in np.arange(tau2, t+1)])

            L2 = (20.3*np.sqrt(self.bandit.k) *
                  np.sum(roots))

            L -= L2

            self.safe = crwd >= L

            assert L < 0, (L)
            assert not np.isnan(L)
        return self.safe

    def observe(self, reward, arm):
        """
        Observes and processes the reward from the chosen arm, updating historical data and safety checks.

        Args:
            reward (float): The observed reward from playing an arm.
            arm (int): The index of the arm that was played.
        """
        self.context_history.append(
            self.meta_contexts[arm]/(self.n_experts**.5))
        self.reward_history.append(reward-self.bandit.expected_reward)
        if self.safe:
            self.cum_reward += reward
            if self.epoch >= 2:
                self.safe = self.check_is_safe(
                    self.epoch, self.t+1, self.prev_l, self.cum_reward)

        self.t += 1

        if self.safe and self.t == self.get_epoch_start(self.get_epoch(self.t)):
            # train model on past epoch
            prev_epoch_start, prev_epoch_end = self.get_epoch_start(
                self.epoch-1), self.get_epoch_end(self.epoch-1)

            prev_epoch_contexts, prev_epoch_rewards = self.context_history[
                prev_epoch_start:prev_epoch_end], self.reward_history[prev_epoch_start:prev_epoch_end]

            assert len(prev_epoch_contexts) == (
                prev_epoch_end-prev_epoch_start), (prev_epoch_start, prev_epoch_end, len(prev_epoch_contexts))

            self.prev_l, self.hat_model = self.choose_safe(self.epoch-1,
                                                           prev_epoch_rewards,
                                                           self.prev_l)

            self.model.fit(prev_epoch_contexts, prev_epoch_rewards)
            self.model_history.append(deepcopy(self.model))

    @property
    def epoch(self):
        return self.get_epoch(self.t)

    def get_gamma(self):

        m = self.epoch if self.safe else self.hat_m

        if m == 1:
            return 1

        delta_m = self.delta_prime/(m**2)
        e_rate = self.e_rate(self.epoch_length(self.epoch-1), delta_m)

        return np.exp(self.beta)*np.sqrt(1/8)*np.sqrt(self.bandit.k/(e_rate))

    def choose(self, advice):
        """
        Chooses an arm to play based on the given advice and the current gamma value for exploration.

        Args:
            advice (np.ndarray): Expert advice on which arm to choose.

        Returns:
            int: The index of the chosen arm.
        """
        self.policy.gamma = self.get_gamma()

        return self.policy.choose(self, advice)

    def estimate_rewards(self, contexts):
        """
        Estimates rewards for each arm based on current contexts and expert advice.

        Args:
            contexts (dict): Current contextual information including expert advice.

        Returns:
            np.ndarray: An array of estimated rewards for each arm.
        """
        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice) - self.bandit.expected_reward

        self.meta_contexts = centered_advice.T
        if self.epoch == 1:
            self._reward_estimates = np.zeros(len(self.meta_contexts))
        else:

            if self.safe:
                self._reward_estimates = self.model.predict(
                    self.meta_contexts/(self.n_experts**.5))
            else:
                self._reward_estimates = self.model_history[self.hat_m].predict(
                    self.meta_contexts/(self.n_experts**.5))

        return self._reward_estimates


class OnlineCover(Average):
    """
    Online Cover algorithm for contextual multi-armed bandit problems. This class extends `Average` and implements
    an algorithm that leverages online ridge regression models within a cover method to balance exploration
    and exploitation effectively. Here it is used as an approximation of ILTCB and applied to solve the Meta-CMAB

    Attributes:
        alpha (float): Regularization parameter for the ridge regression models.
        epsilon (float): Exploration factor to ensure all arms are periodically tested.
        experts (int): Number of experts or dimension of the context.
        psi (float): Adjustment factor for reward calculation.
        cover_size (int): Number of models in the cover, defining the ensemble size.
    """

    def __init__(self, bandit, policy, experts, name=None,
                 alpha=1,  cover_size=2,  psi=0.01):
        """
        Initializes an instance of the OnlineCover class.

        Args:
            bandit (Bandit): The bandit problem instance this agent will interact with.
            policy (Policy): The decision-making policy based on aggregated advice.
            experts (int): Number of experts or context dimensionality.
            name (str, optional): Optional name for the agent.
            alpha (float): Regularization strength of the Ridge regression.
            cover_size (int): Number of models to maintain in the ensemble.
            psi (float): Scaling factor for the cost-sensitive reward.
        """

        super().__init__(bandit, policy,
                         experts, name=name, )

        self.alpha = alpha

        self.experts = experts
        self.policy.eps = 0
        self._models = None
        self.context_dimension = experts
        self.psi = psi
        self.cover_size = cover_size

    def short_name(self):

        return f"Meta-CMAB (ILTCB)"

    @property
    def models(self):
        """
        Lazy initialization for the models if they are not already created.

        Returns:
            list: A list of initialized online ridge regression models.
        """
        if self._models is None:

            self._models = self._init_model()

        return self._models

    def _init_model(self):
        """
        Initializes the ensemble of ridge regression models.

        Returns:
            list: A list of OnlineRidge instances.
        """
        return [OnlineRidge(
            self.alpha) for _ in range(self.cover_size)]

    def initialize_parameters(self):
        self._models = None

        self.all_context_history = []
        self.context_history = []
        self.reward_history = []
        self.all_reward_history = []
        self.action_history = []

    def probabilities(self, contexts):
        """
        Calculates the action probabilities based on the current context and the predictions from the models.

        Args:
            contexts (dict): Current contextual information including expert advice.

        Returns:
            np.ndarray: Probabilities of choosing each arm.
        """
        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice) - self.bandit.expected_reward

        self.meta_contexts = centered_advice.T

        self.greedy_actions = np.array(
            [greedy_choice(m.predict(self.meta_contexts)) for m in self.models])

        self.models[0].uncertainties(self.meta_contexts)

        return self.greedy_actions.mean(axis=0)

    def reset(self):
        super().reset()

    def observe(self, reward, arm):
        """
        Updates the agent's state based on the observed reward and the arm that was played.

        Args:
            reward (float): The reward received from the bandit after playing an arm.
            arm (int): The index of the arm that was played.
        """
        self.context_history.append(self.meta_contexts[arm])
        self.action_history.append(arm)

        new_contexts = self.meta_contexts[arm:arm+1]
        self.all_context_history.extend(new_contexts)

        self.reward_history.append(reward)

        greedy_actions = self.greedy_actions

        # see Appendix F of "Taming the Monster: A Fast and Simple Algorithm for Contextual Bandits"
        self.epsilon = 0.05 * \
            min(1/self.bandit.k, 1/(np.sqrt((self.t+1)*self.bandit.k)))*self.bandit.k

        mu = self.epsilon/self.bandit.k

        smoothed_policy = self.epsilon + (1-self.epsilon)*self.policy.pi

        r_v = np.zeros(self.bandit.k)
        r_v[arm] = reward-self.bandit.expected_reward
        chosen = np.zeros(self.bandit.k)
        chosen[arm] = 1
        for i, model in enumerate(self.models):
            if i == 0:
                model.partial_fit(
                    self.meta_contexts[arm:arm+1], [reward-self.bandit.expected_reward])

                continue

            greedy_actions[i-1] = greedy_choice(
                self.models[-1].predict(self.meta_contexts))

            Q_i = greedy_actions[:i].mean(axis=0)
            p_i = self.epsilon + (1-self.epsilon)*Q_i
            cost_sensitive_reward = chosen * \
                (r_v) / (smoothed_policy) + mu/p_i * self.psi

            round_rewards = cost_sensitive_reward[arm:arm+1]
            self.all_reward_history.extend(round_rewards)

            model.partial_fit(self.meta_contexts, cost_sensitive_reward)

        self.t += 1


def solve_q(rhs, p_a):
    """
    Solves for the upper confidence bound using the KL divergence.

    Args:
        rhs (float): The right-hand side of the KL-UCB equation, typically involving logarithmic terms for exploration.
        p_a (float): The observed empirical mean of arm pulls.

    Returns:
        float: The calculated upper confidence bound.
    """
    if p_a == 1:
        return 1
    q = np.arange(p_a, 1, 0.01)
    lhs = []
    for el in q:
        lhs.append(KL(p_a, el))
    lhs_array = np.array(lhs)
    lhs_rhs = lhs_array - rhs
    lhs_rhs[lhs_rhs <= 0] = np.inf
    min_index = lhs_rhs.argmin()
    return q[min_index]


def klucb_func(pulls, arm_rewards, time_steps, num_bandits):
    """
    Calculates the upper confidence bounds for each arm using the KL-UCB algorithm.

    Args:
        pulls (np.ndarray): Array containing the number of times each arm has been pulled.
        arm_rewards (np.ndarray): Array containing the total reward received from each arm.
        time_steps (int): Current time step in the bandit process.
        num_bandits (int): Total number of arms in the bandit.

    Returns:
        np.ndarray: An array of upper confidence bounds for each arm.
    """
    ucb_arms = np.zeros(num_bandits, dtype=float)
    for x in range(0, num_bandits):
        p_a = arm_rewards[x]/pulls[x]
        rhs = (np.log(time_steps) + 3*np.log(np.log(time_steps)))/pulls[x]
        ucb_arms[x] = solve_q(rhs, p_a)
    return ucb_arms


def ucb_func(pulls, arm_rewards, time_steps, num_bandits,):
    """
    Computes the upper confidence bounds for each arm using the UCB1 algorithm.

    Args:
        pulls (np.ndarray): Number of times each arm has been pulled.
        arm_rewards (np.ndarray): Total reward received from each arm.
        time_steps (int): Current time step in the bandit process.
        num_bandits (int): Number of arms in the bandit.

    Returns:
        np.ndarray: An array of upper confidence bounds for each arm.
    """

    means = np.array(arm_rewards / pulls)
    ucb_terms = means + np.sqrt(2 * np.log(time_steps)
                                * np.divide(np.ones([1, num_bandits]), pulls))

    return ucb_terms


class MAB(Average):
    """
    Multi-Armed Bandit (MAB) reduction implementation that can switch between different base algorithms
    like Thompson Sampling (TS) and Upper Confidence Bound (UCB) strategies, including a KL-UCB variant.

    Attributes:
        base (str): The base algorithm to use ("TS", "UCB", "KL-UCB").
    """

    def __init__(self, bandit, policy, experts,
                 name=None,   base="KL-UCB"):
        """
        Initializes the MAB agent.

        Args:
            bandit (Bandit): The bandit problem instance this agent will interact with.
            policy (Policy): Decision-making policy.
            experts (int): Number of experts or arms.
            name (str, optional): Optional name for the agent.
            base (str): The base strategy for the agent ("TS", "UCB", or "KL-UCB").
        """
        self.base = base
        super().__init__(bandit, policy,
                         experts,  name=name)

    def short_name(self):
        return f"Meta-MAB({self.base})"

    def initialize_parameters(self):
        self.reward_history = []
        self.context_history = []
        self.effective_n = self.n
        if self.base == "TS":
            self.betas = np.ones(self.effective_n)
            self.alphas = np.ones(self.effective_n)
        elif "UCB" in self.base:
            self.pulls = np.zeros(self.effective_n)
            self.scores = np.zeros(self.effective_n)
        self.chosen_expert = np.random.randint(self.effective_n)
        self.w = np.ones(self.effective_n)

    def get_weights(self, contexts):
        """
        Calculates weights for each expert based on the chosen base strategy.

        Args:
            contexts (dict): Contextual information that may influence the weighting decision.

        Returns:
            np.ndarray: A weight matrix where selected expert's weight is set to 1.
        """

        w = np.zeros((self.n, self.bandit.k))
        if self.base == "TS":
            expert_values = np.random.beta(self.alphas, self.betas)
            self.chosen_expert = randargmax(expert_values)

        else:
            if self.t < self.effective_n:
                self.chosen_expert = np.random.choice(
                    np.asarray(self.pulls == 0).nonzero()[0])
            else:
                if self.base == "KL-UCB":
                    expert_values = klucb_func(
                        self.pulls, self.scores, self.t+1, self.effective_n)
                else:
                    expert_values = ucb_func(
                        self.pulls, self.scores, self.t+1, self.effective_n)

                self.chosen_expert = randargmax(expert_values)

        w[self.chosen_expert, :] = 1

        return w

    def observe(self, reward, arm):
        """
        Updates the internal state of the agent based on the observed reward and selected arm.

        Args:
            reward (float): Observed reward from the bandit.
            arm (int): Index of the arm that was played.
        """
        assert 0 <= reward <= 1, reward
        if self.base == "TS":
            self.alphas[self.chosen_expert] += reward
            self.betas[self.chosen_expert] += 1 - reward
        else:
            self.pulls[self.chosen_expert] += 1
            self.scores[self.chosen_expert] += reward

        self.t += 1


class OnlineRidge():
    """
    Implements an online version of the Ridge regression algorithm. This class manages an iterative update
    of the Ridge regression parameters, suitable for online learning scenarios where data arrives in a stream.

    Attributes:
        alpha (float): Regularization parameter that influences the strength of the regularization.
    """

    def __init__(self, alpha, ):
        """
        Initializes the OnlineRidge instance with specified regularization, prior strength, and logging option.

        Args:
            alpha (float): Regularization strength.
        """
        self._model = None
        self.alpha = alpha

    @property
    def model(self):
        """
        Lazily initializes the model parameters if they are not already created.

        Returns:
            dict: A dictionary containing parameters of the Ridge model.
        """
        if self._model is None:

            self._model = self._init_model({})

        return self._model

    def _init_model(self, model):
        """
        Initializes the matrix parameters for the Ridge regression.

        Args:
            model (dict): A dictionary to store and initialize model parameters.

        Returns:
            dict: The initialized model parameters including A, A_inv, b, and theta.
        """
        # model['A'] = np.identity(self.context_dimension) * self.alpha
        model['A_inv'] = np.identity(self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))

        return model

    def partial_fit(self, features, outcomes):
        """
        Updates the model parameters based on new observed features and outcomes.

        Args:
            features (np.ndarray): The feature vectors of the new data.
            outcomes (np.ndarray): The corresponding outcomes or target values.
        """

        self.context_dimension = np.shape(features)[1]
        for x, y in zip(features, outcomes):

            x = x[..., None]/((1*len(x))**0.5)

            self.model['A_inv'] = SMInv(self.model['A_inv'], x, x)

            self.model['b'] += y * x

        self.model['theta'] = (
            self.model['A_inv'].dot(self.model['b']))

    def uncertainties(self, features, sample=None):
        """
        Calculates the uncertainties associated with predictions for given features.

        Args:
            features (np.ndarray): The features for which to calculate uncertainties.
            sample (float, optional): Scaling factor to apply to the noise model.

        Returns:
            np.ndarray: An array of uncertainties for each feature vector.
        """

        features = features/(features.shape[1]**.5)

        values = np.sqrt(
            ((features[:, :, np.newaxis]*self.model['A_inv'][None, ]).sum(axis=1)*features).sum(-1))
        thompson_noise = np.random.normal(
            np.zeros_like(values), values, size=values.shape)
        if sample is not None:
            values = sample*thompson_noise

        return np.asarray(values)

    def predict(self, features):
        """
        Predicts outcomes based on the feature vectors using the learned model parameters.

        Args:
            features (np.ndarray): The feature vectors.

        Returns:
            np.ndarray: Predicted outcomes.
        """
        self.context_dimension = features.shape[1]
        theta = self.model['theta'][None, ]
        features = features/((features.shape[1])**0.5)
        return (features*theta[:, :, 0]).sum(-1)


class LinUCB(Average):
    """
    LinUCB algorithm implementation for contextual multi-armed bandit problems.
    Uses linear regression to estimate the rewards of each arm based on the advice provided by experts.

    Attributes:
        beta (float): Exploration parameter controlling the confidence bounds.
        alpha (float): Regularization parameter for the online ridge regression.
        trials (int): Number of trials the LinUCB agent is expected to run.
        fixed (bool): Determines whether the model should be updated after each trial.
        mode (str): Operational mode of the algorithm, could be 'UCB' for Upper Confidence Bound or 'TS' for Thompson Sampling.
    """

    def __init__(self, bandit, policy, experts, beta=1, name=None, trials=1,
                 alpha=1,  fixed=False,  mode='UCB'):
        """
        Args:
            bandit (Bandit): The bandit problem instance this agent will interact with.
            policy (Policy): The decision-making policy based on aggregated advice.
            experts (int): Number of features or experts in the context.
            beta (float): Exploration parameter.
            name (str, optional): Optional name for the agent.
            trials (int): Total number of trials for the agent.
            alpha (float): Regularization parameter for the online regression model.
            fixed (bool): If True, prevents the model from being updated after each trial.
            mode (str): Specifies the mode of operation, either 'UCB' or 'TS'.
        """

        super().__init__(bandit, policy,
                         experts, name=name)

        self.beta = beta
        self.alpha = alpha
        self.trials = trials
        self._model = None
        self.fixed = fixed
        self.mode = mode
        self.context_dimension = experts

    def short_name(self):
        return f"Meta-CMAB (LinUCB)"

    @property
    def model(self):
        if self._model is None:
            self.counts = {}
            self._model = OnlineRidge(self.alpha)

        return self._model

    def get_values(self, contexts, return_std=True):
        """
        Computes the estimated rewards and optionally the uncertainties for given contexts.

        Args:
            contexts (np.ndarray): Contextual information for each arm.
            return_std (bool, optional): If True, also returns the standard deviations of the reward estimates.

        Returns:
            tuple or np.ndarray: Estimated rewards and optionally uncertainties and zero array.
        """

        estimated_rewards = self.model.predict(
            contexts)
        if return_std:
            uncertainties = self.model.uncertainties(
                contexts, sample=1 if (self.mode == 'TS') else None)

            return estimated_rewards, uncertainties, 0
        else:
            return estimated_rewards

    def initialize_parameters(self):
        self._model = None

        self.context_history = []
        self.full_context_history = []
        self.reward_history = []
        self.action_history = []

        self.selection_history = []

    def estimate_rewards(self, contexts, mu_only=False):
        """
        Estimates rewards for each context using the linear model.

        Args:
            contexts (dict): Contextual information including expert advice.
            mu_only (bool, optional): If True, returns only the estimated rewards (mu).

        Returns:
            np.ndarray: Estimated rewards potentially adjusted by the exploration parameter (beta).
        """
        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice) - self.bandit.expected_reward

        self.meta_contexts = centered_advice.T

        mu, sigma, eps_sigma = self.get_values(self.meta_contexts)
        if mu_only:
            return mu

        assert self.context_dimension > 0

        return mu + sigma*self.beta+eps_sigma

    def reset(self):
        super().reset()

    def observe(self, reward, arm):
        """
        Updates the agent's state based on the observed reward and the arm that was played.

        Args:
            reward (float): The reward received from the bandit after playing an arm.
            arm (int): The index of the arm that was played.
        """
        self.full_context_history.append(self.meta_contexts)
        self.context_history.append(self.meta_contexts[arm])
        selection = np.zeros(self.bandit.k)
        selection[arm] = 1
        self.selection_history.append(selection)
        self.action_history.append(arm)

        action_context = self.meta_contexts[arm]
        self.reward_history.append(reward)
        if not self.fixed:
            self.model.partial_fit(
                [action_context], [(reward-self.bandit.expected_reward)])

        self.t += 1


class SupLinUCBVar(Average):
    """
    Supervised LinUCB with Variations (SupLinUCBVar) extends the LinUCB algorithm to handle multiple 
    exploration strategies through an ensemble of EXPL3 models, adjusting exploration based on a variable threshold.

    This class is designed to adaptively select among different levels of exploration based on the performance
    and confidence of various models in the ensemble.

    Attributes:
        beta (float): Exploration parameter that controls the confidence bounds.
        alpha (float): Regularization parameter for the online ridge regression.
        ensemble (list): List of EXPL3 models each configured with a different exploration threshold.
    """

    def __init__(self, bandit, policy, experts, beta=.1, name=None,
                 alpha=1):

        self.bandit = bandit
        self._model = None
        self.context_dimension = experts
        self.alpha = alpha
        self.beta = beta
        self.name = None
        self.ensemble = [EXPL3(bandit, policy, experts, beta, name,  alpha=alpha, thresh=(2**-l), )
                         for l in np.arange(1, max(1, (np.log(np.sqrt(self.bandit.k))/np.log(2)))+1e-10)]
        assert len(self.ensemble) >= 1

    def short_name(self):
        return f"Meta-CMAB (UCB)"

    def reset(self):
        self.t = 0
        [e.reset() for e in self.ensemble]

    def choose(self, advice):
        """
        Chooses an arm to play based on aggregated advice and the current state of the ensemble models.

        Args:
            advice (dict): Expert advice for each arm.

        Returns:
            int: The index of the chosen arm.
        """
        mask = np.ones(self.bandit.k)

        for e in self.ensemble:
            self.active_expl = e
            values = e.estimate_rewards(advice, mask=mask)

            if e.exploratory:
                break

            mask = np.logical_and(
                mask, ((values) >= np.max(values)-e.thresh*self.beta))

            assert np.sum(mask) > 0, (np.max(values), e.thresh)

        return np.random.choice(self.bandit.k, p=greedy_choice(values))

    def observe(self, reward, arm):
        """
        Observes and processes the reward from the chosen arm, updating all relevant models in the ensemble
        up to the actively selected exploration model.

        Args:
            reward (float): The reward received from the bandit after playing an arm.
            arm (int): The index of the arm that was played.
        """

        for e in self.ensemble:
            e.meta_contexts = self.active_expl.meta_contexts
            e.offset = self.active_expl.offset
            e.observe(reward, arm)
            if e == self.active_expl:
                break
        self.t += 1


class EXPL3(Average):
    """
    EXPL3 is similar to LinUCB but instead of adding the exploration bonus to the estimated reward for each arm, 
    it explores when the highest exploration bonus (i.e., confidence interval) is above some threshold

    Attributes:
        beta (float): Coefficient for adjusting exploration based on the uncertainty.
        alpha (float): Regularization parameter for the online ridge regression model.
        thresh (float): Threshold for determining when to switch from exploration to exploitation.

        context_dimension (int): Number of features or experts in the context plus one for a bias term.
    """

    def __init__(self, bandit, policy, experts, beta=1, name=None,  thresh=0,
                 alpha=1, ):

        super().__init__(bandit, policy,
                         experts, name=name)

        self.beta = beta
        self.alpha = alpha
        self.thresh = thresh
        self._model = None
        self.context_dimension = experts+1

    @property
    def model(self):
        if self._model is None:
            self._model = OnlineRidge(self.alpha)

        return self._model

    def get_values(self, contexts, return_std=True):

        estimated_rewards = self.model.predict(contexts)

        if return_std:
            uncertainties = self.model.uncertainties(contexts)
            return estimated_rewards, uncertainties
        else:
            return estimated_rewards

    def initialize_parameters(self):
        self._model = None

        self.context_history = []
        self.reward_history = []
        self.action_history = []

    def estimate_rewards(self, contexts, mask=None):
        """
        Estimates rewards for each context using the linear model and determines the mode of operation
        (explorative or exploitative) based on the uncertainty threshold.

        Args:
            contexts (dict): Contextual information including expert advice.
            mask (np.ndarray, optional): A mask to apply over the rewards to filter out certain actions.

        Returns:
            np.ndarray: Modified reward estimates adjusted for exploration or exploitation.
        """

        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice)

        self.offset = np.reshape(self.bandit.expected_reward, (1, 1))

        self.meta_contexts = centered_advice.T - self.offset
        self.meta_contexts = np.concatenate(
            (self.meta_contexts, np.ones((self.bandit.k, 1))), axis=1)
        mu, sigma = self.get_values(self.meta_contexts)

        mu += self.offset[..., 0]
        mu -= np.min(mu)

        if mask is not None:
            mu *= mask
            sigma *= mask

        self.exploratory = np.max(sigma) > self.thresh

        if self.exploratory:

            return sigma  # explore
        else:
            return mu  # exploit

    def reset(self):
        super().reset()

    def observe(self, reward, arm):
        """
        Observes and processes the reward from the chosen arm, updating the model based on the context


        Args:
            reward (float): The reward received from the bandit after playing an arm.
            arm (int): The index of the arm that was played.
        """
        self.context_history.append(self.meta_contexts[arm])
        self.action_history.append(arm)

        action_context = self.meta_contexts[arm]
        offset = self.offset

        self.model.partial_fit(
            [action_context], [reward-offset])
        self.t += 1
