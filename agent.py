
from expert import *

EXPECTED_AVG_REWARD = .5


def get_distance(i, spread_type, max_distance, n_experts):

    window_width = max(0, min(1-max_distance, max_distance)-.1)
    if spread_type in ('diverse', 'heterogeneous'):
        cluster_id = i / (n_experts-1)*2
    elif spread_type == 'polarized':
        cluster_id = 0 if i < n_experts//2 else 2
    else:
        cluster_id = 1
    desired_distance = max_distance + (cluster_id-1) * window_width
    return desired_distance


class Collective(Agent):
    def __init__(self, bandit, policy, n_experts,  gamma=None,  name=None,   alpha=1, beta=1, expert_spread='normal',
                 ):

        super(Collective, self).__init__(bandit, policy)
        self.expert_spread = expert_spread

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma


        self.k = self.bandit.k
        self.n = n_experts
        self.name = name
        self.advice = np.zeros((self.n, self.bandit.k))
        self._value_estimates = np.zeros(self.k)
        self._probabilities = np.zeros(self.k)

        self.confidences = np.ones((self.n, self.bandit.k))

        self.initialize_w()

    def initialize_w(self):
        pass

    def short_name(self):
        return "Average"

    @property
    def info_str(self):
        info_str = ""
        return info_str


    def observe(self, reward, arm):

        self.t += 1

    def get_weights(self, contexts):

        self.confidences = np.ones((self.n, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']
        w = safe_logit(self.confidences)

        return w

    def __str__(self):
        if self.name is not None:
            return self.name
        return self.short_name

    def set_name(self, name):
        self.name = name

    @staticmethod
    def prior_play(experts, bias_steps, expert_spread, base_bandit, average_expert_distance=0, spread=1):
        options = KernelUCB.KERNELS
        np.random.shuffle(options)
        for e in experts:
            e.kernel = (np.random.choice(options))

        for i, e in (list(enumerate(experts))):
            e.index = i

            desired_distance = get_distance(i, expert_spread, average_expert_distance, len(
                experts))

            if expert_spread == 'polarized':
                if i == 0 or i == len(experts)//2:
                    cluster_bandit = base_bandit.from_bandit(
                        desired_distance=desired_distance)

                prior_bandit = cluster_bandit.from_bandit(
                    desired_distance=0.05)
            else:
                prior_bandit = base_bandit.from_bandit(
                    desired_distance=desired_distance)

            e.prior_play(steps=bias_steps, bandit=prior_bandit, spread=spread)

    def choose(self, advice, greedy=False):
        return self.policy.choose(self, advice, greedy=greedy)

    def probabilities(self, contexts):
        self.advice = np.copy(contexts['advice'])
        if isinstance(self, Exp4) and not np.allclose(np.sum(self.advice, axis=1), 1):
            advice = np.zeros_like(self.advice)
            advice[self.advice == np.max(self.advice, axis=1)[:, None]] = 1
            self.advice = advice/np.sum(advice, axis=1)[:, None]

        w = self.get_weights(contexts)
        self._probabilities = np.sum(w * self.advice, axis=0)

        assert len(self._probabilities) == self.bandit.k
        return self._probabilities

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice - self.bandit.expected_reward), axis=0)

        return self._value_estimates

    def reset(self):
        super().reset()

        self.initialize_w()


class Exp4(Collective):
    def __init__(self, bandit, policy, n_experts, gamma, name=None,  expert_spread='normal',
                 confidence_weight=100,
                 crop=1, prefix='W',):
        super(Exp4, self).__init__(bandit, policy,
                                   n_experts,  name=name, gamma=gamma,   expert_spread=expert_spread)
        self.crop = crop
        self.prefix = prefix
        self.e_sum = 1
        self.w = np.zeros(self.n)
        self.confidence_weight = confidence_weight

    def copy(self):
        return Exp4(self.bandit, self.policy, self.n, self.gamma,
                    crop=self.crop, prefix=self.prefix)

    def short_name(self):
        return "EXP4-IX"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.zeros(self.n)

    def reset(self):
        super(Exp4, self).reset()

    def get_weights(self, contexts):

        w = np.copy(self.w)
        N = self.n

        w = np.repeat(w[:, np.newaxis], self.bandit.k, axis=1)
        self.confidences = np.zeros((N, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']

        c = self.confidences*self.gamma*self.confidence_weight

        w = softmax(c + w, axis=0)

        return w

    def observe(self, reward, arm):

        self.effective_n = self.n

        assert np.allclose(np.sum(self.advice, axis=1),
                           1), "expected probability advice"
        x_t = self.advice[:, arm] * (1-reward)
        y_t = x_t / (self.policy.pi[arm]+self.gamma)

        self.e_sum += np.sum(np.max(self.advice, axis=0))

        numerator = self.e_sum
        lr = np.sqrt(np.log(self.effective_n) /
                     numerator)
        self.w -= (lr) * y_t

        self.t += 1


class SquareCB(Collective):
    def __init__(self, bandit, policy, n_experts, name=None,  expert_spread='normal',
                 confidence_weight=100,
                 crop=1, prefix='W',):
        super(SquareCB, self).__init__(bandit, policy,
                                       n_experts,  name=name,   expert_spread=expert_spread)
        self.crop = crop
        self.prefix = prefix
        self.e_sum = 1
        self.w = np.zeros(self.n)
        self.confidence_weight = confidence_weight

    def copy(self, ):
        return SquareCB(self.bandit, self.policy, self.n,
                        crop=self.crop, prefix=self.prefix,)

    def short_name(self):
        return "SquareCB"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.zeros(self.n)

    def reset(self):
        super(SquareCB, self).reset()

    def get_weights(self, contexts):

        w = np.copy(self.w)
        w = np.repeat(w[:, np.newaxis], self.bandit.k, axis=1)
        self.confidences = np.zeros((self.n, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']

        c = self.confidences*self.confidence_weight

        w = softmax(c + w, axis=0)

        return w

    def observe(self, reward, arm):

        x_t = (self.advice[:, arm] - reward)**2
        y_t = x_t / (self.policy.pi[arm])

        self.w -= 2 * y_t

        self.t += 1

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice), axis=0)

        return self._value_estimates



class MAB(Collective):

    def __init__(self, bandit, policy, experts,  include_time=False, include_ctx=True, expert_spread='normal',
                 name=None,  gamma=None):

        super().__init__(bandit, policy,
                         experts, gamma=gamma,  name=name,  expert_spread=expert_spread)

        self.include_ctx = include_ctx
        self.include_time = include_time

    def short_name(self):
        return "Meta-MAB"

    def initialize_w(self):
        self.reward_history = []
        self.context_history = []
        self.effective_n = self.n
        self.betas = np.ones(self.effective_n)
        self.alphas = np.ones(self.effective_n)
        self.chosen_expert = np.random.randint(self.effective_n)
        self.w = np.ones(self.effective_n)

    def get_weights(self, contexts):

        self.confidences = np.zeros_like(contexts['advice']) if contexts.get(
            'confidence', None) is None else contexts['confidence']

        c = np.clip(self.confidences, 1e-6, 1 - 1e-6)
        w = np.zeros_like(c)
        conf_weight = 0 if contexts.get('confidence', None) is None else 100
        ac = c*conf_weight + self.alphas[:, None]
        bc = (1 - c)*conf_weight + self.betas[:, None]

        expert_values = np.random.beta(ac, bc)

        expert_values[:, :] = expert_values[:, 0:1]

        self.chosen_expert = randargmax(expert_values, axis=0)[0]

        w[self.chosen_expert, :] = 1
        return w

    def observe(self, reward, arm):


        self.alphas[self.chosen_expert] += reward
        self.betas[self.chosen_expert] += 1 - reward

        self.t += 1



class LinUCB(Collective):
    def __init__(self, bandit, policy, experts, beta=1,name=None, expert_spread='normal',
                 alpha=10, ):

        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, beta=beta, expert_spread=expert_spread)

        self._model = None
        self.context_dimension = experts+1

    def copy(self):
        return LinUCB(self.bandit, self.policy, self.n, beta=self.beta,
                      alpha=self.alpha                      )

    def short_name(self):
        return "Meta-CMAB"

    @property
    def model(self):
        if self._model is None:

            self._model = self._init_model({})

        return self._model

    def predict(self, advice):
        pr = np.sum(self.get_weights(None, full=True) *
                    (advice - self.bandit.expected_reward), axis=0)
        return pr

    def _init_model(self, model):
        model['A'] = np.identity(self.context_dimension) * self.alpha
        model['A_inv'] = np.identity(self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))

        return model

    def get_values(self, contexts, return_std=True):

        theta = self.model['theta'][None, ]

        estimated_rewards = (contexts*theta[:, :, 0]).sum(-1)

        if return_std:
            uncertainties = np.sqrt(
                ((contexts[:, :, np.newaxis]*self.model['A_inv'][None, ]).sum(axis=1)*contexts).sum(-1))

            return estimated_rewards, uncertainties
        else:
            return estimated_rewards

    def initialize_w(self):
        self._model = None

        self.context_history = []
        self.reward_history = []
        self.action_history = []




    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        centered_advice = self.advice - self.bandit.expected_reward

        self.meta_contexts = np.concatenate(
            (centered_advice, np.ones((1, self.bandit.k))), axis=0).T

        mu, sigma = self.get_values(self.meta_contexts)

        return mu + sigma*self.beta

    def reset(self):
        super(LinUCB, self).reset()

    def observe(self, reward, arm):
        self.context_history.append(self.meta_contexts[arm])
        self.action_history.append(arm)

        action_context = self.meta_contexts[arm][..., None]
        self.reward_history.append(reward)

        self.model['A_inv'] = SMInv(
            self.model['A_inv'], action_context, action_context, 1)

        self.model['b'] += (reward -
                            self.bandit.expected_reward) * action_context

        self.model['theta'] = (
            self.model['A_inv'].dot(self.model['b']))

        self.t += 1

    def get_weights(self):
        return self.model['theta']
