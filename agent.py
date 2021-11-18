
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
    def __init__(self, bandit, policy, n_experts, choice_only=False, gamma=None,  name=None, value_mode=True,  alpha=1, beta=1, expert_spread='normal',
                 top=False,  add_random=False, ):

        super(Collective, self).__init__(bandit, policy)
        self.expert_spread = expert_spread

        self.add_random = add_random

        self.beta = beta
        self.alpha = alpha
        self.top = top
        self.gamma = gamma

        self.choice_only = choice_only
        self.value_mode = value_mode

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

    @property
    def base_str(self):
        return ("Av" if self.value_mode else 'Mj')

    def observe(self, reward, arm, contexts={}):
        self.last_action = arm

        self.t += 1

    def get_weights(self, contexts):

        self.confidences = np.ones((self.n, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']
        w = safe_logit(self.confidences)
        if not self.value_mode:  # scale weights to sum to 1
            # set arm confidence to 1 if all experts have the same confidence
            w[:, np.std(w, axis=0) == 0] = 1
            w /= np.sum(np.abs(w), axis=0)

        return w

    def __str__(self):
        if self.name is not None:
            return self.name
        return self.base_str + self.info_str

    def set_name(self, name):
        self.name = name

    @staticmethod
    def prior_play( experts, bias_steps,expert_spread, base_bandit, average_expert_distance=0, spread=1):
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

        if self.add_random and len(self.advice) == self.n:
            random_advice = np.random.uniform(size=self.bandit.k)
            if isinstance(self, Exp4):
                random_advice = np.ones(self.bandit.k)/self.bandit.k
            random_advice /= np.sum(random_advice)
            self.advice = np.vstack([self.advice, random_advice])
        w = self.get_weights(contexts)
        self._probabilities = np.sum(w * self.advice, axis=0)

        if type(self)==SquareCB:
            assert self.value_mode
            values = self._probabilities+0
            best_arm = np.argmax(values)
            self._probabilities=np.zeros_like(values)
            self.misspecification = 0#self.trials/2
            # printt(self.t,"gamma",self.gamma)
            # printt("values",values)
            self._probabilities[:] = 1/(self.bandit.k+self.gamma*(values[best_arm]-values))
            # printt("dist probs",self._probabilities)
            # printt(type(self._probabilities[best_arm]),)
            self._probabilities[best_arm]+=(1-(np.sum(self._probabilities)))
            # printt("probs",self._probabilities)

            assert np.isclose(np.sum(self._probabilities),1) ,np.sum(self._probabilities)
            assert (self._probabilities>=0).all()
        assert len(self._probabilities) == self.bandit.k
        return self._probabilities

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        if self.add_random:
            random_advice = np.random.uniform(size=self.bandit.k)/2+.25
            if isinstance(self, Exp4):
                random_advice = np.ones(self.bandit.k)/self.bandit.k
            self.advice = np.vstack([self.advice, random_advice])

        self._value_estimates = np.sum(
            self.get_weights(contexts) * (self.advice - EXPECTED_AVG_REWARD), axis=0)

        return self._value_estimates

    def reset(self):
        super().reset()

        self.initialize_w()


class Exp4(Collective):
    def __init__(self, bandit, policy, n_experts, gamma, choice_only=False, name=None, sum_adapt=False, expert_spread='normal',
                 gamma_2=0, value_mode=True,  top=False, add_random=False, confidence_weight=100,
                  crop=1, prefix='W', flip=None, flip_strat=None, adaptive=False, ratio=1):
        super(Exp4, self).__init__(bandit, policy,
                                   n_experts, choice_only, name=name, gamma=gamma,   expert_spread=expert_spread)
        self.crop = crop
        self.sum_adapt = sum_adapt
        self.prefix = prefix
        self.adaptive = adaptive
        self.add_random = add_random
        self.e_sum = 1
        self.w = np.zeros(self.n+self.add_random)
        self.confidence_weight = confidence_weight
        self.gamma_2 = gamma_2
        self.value_mode = value_mode

    def copy(self, ratio=None):
        return Exp4(self.bandit, self.policy, self.n, self.gamma, choice_only=self.choice_only, sum_adapt=self.sum_adapt, gamma_2=self.gamma_2,
                    value_mode=self.value_mode, crop=self.crop, prefix=self.prefix, adaptive=self.adaptive,
                    top=self.top, add_random=self.add_random,
                    flip=self.flip, flip_strat=self.flip_strat)

    def short_name(self):
        return "EXP4-IX"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.zeros(self.n+self.add_random)

    def reset(self):
        super(Exp4, self).reset()

    def get_weights(self, contexts, full=True):

        w = np.copy(self.w)
        N = self.n+self.add_random
   
        w = np.repeat(w[:, np.newaxis], self.bandit.k, axis=1)
        self.confidences = np.zeros((N, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']
        if self.add_random and len(self.confidences) == self.n:
            random_confidence = np.zeros(self.bandit.k)
            if contexts.get('confidence', None) is not None:
                random_confidence += RANDOM_CONFIDENCE
            self.confidences = np.vstack((self.confidences, random_confidence))
        c = self.confidences*self.gamma*self.confidence_weight

        w = softmax(c + w, axis=0)

        return w

    def observe(self, reward, arm, contexts={}):
        self.last_action = arm

        self.effective_n = self.n + self.add_random

        advice = self.advice if contexts.get('advice',None) is None else contexts['advice']
        assert np.allclose(np.sum(advice,axis=1),1),"expected probability advice"
        x_t = advice[:, arm] * (1-reward)
        y_t = x_t / (self.policy.pi[arm]+self.gamma)

        self.e_sum += np.sum(np.max(advice, axis=0))

        numerator = self.e_sum
        lr = np.sqrt(np.log(self.effective_n) /
                        numerator)
        # lr = 2*self.gamma
        self.w -= (lr) * y_t

        self.t += 1

    @property
    def base_str(self):
        prefix = self.prefix
        s_string = ("{}Sok".format(self.adaptive) if self.adaptive >
                    0 else "ok")+('s' if self.sum_adapt else '')

        return prefix+"LEx4q{}{}{}{}{}".format(s_string, self.gamma, "randddd" if self.add_random else "", 0, "" if ((not self.value_mode and self.crop == 1) or (self.value_mode and self.crop == self.bandit.k)) else self.crop)


class SquareCB(Collective):
    def __init__(self, bandit, policy, n_experts, gamma, choice_only=False, name=None, sum_adapt=False, expert_spread='normal',
                 gamma_2=0, value_mode=True,  top=False, add_random=False, confidence_weight=100,
                  crop=1, prefix='W', flip=None, flip_strat=None, adaptive=False, ratio=1):
        super(SquareCB, self).__init__(bandit, policy,
                                   n_experts, choice_only, name=name, gamma=gamma,   expert_spread=expert_spread)
        self.crop = crop
        self.sum_adapt = sum_adapt
        self.prefix = prefix
        self.adaptive = adaptive
        self.add_random = add_random
        self.e_sum = 1
        self.w = np.zeros(self.n+self.add_random)
        self.confidence_weight = confidence_weight
        self.gamma_2 = gamma_2
        self.value_mode = value_mode

    def copy(self, ratio=None):
        return SquareCB(self.bandit, self.policy, self.n, self.gamma, choice_only=self.choice_only, sum_adapt=self.sum_adapt, gamma_2=self.gamma_2,
                    value_mode=self.value_mode, crop=self.crop, prefix=self.prefix, adaptive=self.adaptive,
                    top=self.top, add_random=self.add_random,
                    flip=self.flip, flip_strat=self.flip_strat)

    def short_name(self):
        return "SquareCB"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.zeros(self.n+self.add_random)

    def reset(self):
        super(SquareCB, self).reset()

    def get_weights(self, contexts, full=True):

        w = np.copy(self.w)
        N = self.n+self.add_random
   
        w = np.repeat(w[:, np.newaxis], self.bandit.k, axis=1)
        self.confidences = np.zeros((N, self.bandit.k)) if contexts.get(
            'confidence', None) is None else contexts['confidence']
        if self.add_random and len(self.confidences) == self.n:
            random_confidence = np.zeros(self.bandit.k)
            if contexts.get('confidence', None) is not None:
                random_confidence += RANDOM_CONFIDENCE
            self.confidences = np.vstack((self.confidences, random_confidence))
        c = self.confidences*self.gamma*self.confidence_weight

        w = softmax(c + w, axis=0)

        return w

    def observe(self, reward, arm, contexts={}):
        self.last_action = arm

        advice = self.advice if contexts.get('advice',None) is None else contexts['advice']
        x_t = (advice[:, arm] -reward)**2
        y_t = x_t / (self.policy.pi[arm])

        lr = 2
        self.w -= (lr) * y_t

        self.t += 1

    @property
    def base_str(self):
        prefix = self.prefix
        s_string = ("{}Sok".format(self.adaptive) if self.adaptive >
                    0 else "ok")+('s' if self.sum_adapt else '')

        return prefix+"LEx4q{}{}{}{}{}".format(s_string, self.gamma, "randddd" if self.add_random else "", 0, "" if ((not self.value_mode and self.crop == 1) or (self.value_mode and self.crop == self.bandit.k)) else self.crop)


class MAB(Collective):

    def __init__(self, bandit, policy, experts, choice_only=False,  include_time=False, include_ctx=True, value_mode=True, expert_spread='normal',
                 name=None,  top=False, add_random=False, flip=None, flip_strat=None, gamma=None, mode='MetaMAB'):

        super().__init__(bandit, policy,
                         experts, gamma=gamma, choice_only=choice_only, name=name,  value_mode=value_mode, expert_spread=expert_spread,
                         add_random=add_random)

        self.include_ctx = include_ctx
        self.include_time = include_time

    def short_name(self):
        return "Meta-MAB"

    def initialize_w(self):
        self.reward_history = []
        self.context_history = []
        self.effective_n = self.n+self.add_random
        self.betas = np.ones(self.effective_n)
        self.alphas = np.ones(self.effective_n)
        self.chosen_expert = np.random.randint(self.effective_n)
        self.w = np.ones(self.effective_n)

    def get_weights(self, contexts):

        self.confidences = np.zeros_like(contexts['advice']) if contexts.get(
            'confidence', None) is None else contexts['confidence']

        if self.add_random and len(self.confidences) == self.n:
            random_confidence = np.zeros(self.bandit.k)
            if contexts.get('confidence', None) is not None:
                random_confidence += RANDOM_CONFIDENCE
            self.confidences = np.vstack((self.confidences, random_confidence))

        # print(self.confidences)
        c = np.clip(self.confidences, 1e-6, 1 - 1e-6)
        w = np.zeros_like(c)
        # print(c)
        conf_weight = 0 if contexts.get('confidence', None) is None else 100 
        ac = c*conf_weight + self.alphas[:, None]
        bc = (1 - c)*conf_weight + self.betas[:, None]
        # print(self.t,"\nalphas",self.alphas,"\nbetas",self.betas,"\nac",ac[:, 0:1],"\nbc",bc[:, 0:1])
        expert_values = np.random.beta(ac, bc)
        # print("sampled",expert_values[:, 0:1])
        expert_values[:, :] = expert_values[:, 0:1]

        self.chosen_expert = randargmax(expert_values, axis=0)[0]

        w[self.chosen_expert, :] = 1
        return w

    def observe(self, reward, arm, contexts={}):

        self.last_action = arm

        self.alphas[self.chosen_expert] += reward
        self.betas[self.chosen_expert] += 1 - reward

        self.t += 1

    @property
    def base_str(self):
        return str(self.policy) + ("V") + 'TS' + str(2) + ("randddd" if self.add_random else "") + ('time' if self.include_time else '') + ('only' if not self.include_ctx else '')


class LinUCB(Collective):
    def __init__(self, bandit, policy, experts, choice_only=False, beta=1, delta=0.5, name=None, expert_spread='normal',
                 value_mode=True, add_b=True,   expand=False, top=False,
                 alpha=1, early=False, shared_model=False, shared_meta=False, flip=None, flip_strat=None, log_advice=False,
                 d_delta=1, gamma=1, S=1, adaptive=0, sigma=0, L=0, residual=False, proximity=False, oracular=False):

        self.oracular = oracular
        self.d_delta = d_delta
        self.residual = residual
        self.add_b = add_b
        self.proximity = proximity
        super().__init__(bandit, policy,
                         experts, choice_only=choice_only, name=name, alpha=alpha, beta=beta, expert_spread=expert_spread)
        self.gamma = gamma
        self.S = S
        self.L = L
        self.sigma = sigma
        self.adaptive = adaptive
        self.log_advice = log_advice
        self.value_mode = value_mode
        self.early = early
        self._model = None
        self.meta_contexts = None

        self.context_history = [[] for _ in range(self.k)]
        self.reward_history = [[] for _ in range(self.k)]
        self.delta = delta
        self.shared_model = shared_model
        self.shared_meta = shared_meta

        self.k = 1 if self.shared_model else self.bandit.k

    def copy(self, ratio):
        return LinUCB(self.bandit, self.policy, self.n, choice_only=self.choice_only, beta=self.beta, delta=self.delta, value_mode=self.value_mode, add_b=self.add_b,
                      top=self.top, alpha=self.alpha, early=self.early, shared_model=self.shared_model, shared_meta=self.shared_meta,
                      flip=self.flip, flip_strat=self.flip_strat, log_advice=self.log_advice, d_delta=self.d_delta, gamma=self.gamma, S=self.S, adaptive=self.adaptive, sigma=self.sigma, L=self.L, residual=self.residual,
                      proximity=self.proximity, oracular=self.oracular)

    def short_name(self):
        return "Meta-CMAB"

    @property
    def model(self):
        if self._model is None:
            self.context_dimension = np.size(self.meta_contexts[0])

            self._model = self._init_model({})

        return self._model

    def predict(self, advice):
        pr = np.sum(self.get_weights(None, full=True) *
                    (advice - EXPECTED_AVG_REWARD), axis=0)
        return pr

    def _init_model(self, model):
        model['A'] = np.identity(self.context_dimension) * self.alpha

        model['A_inv'] = np.identity(
            self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))

        model['theta'] = np.zeros((self.context_dimension, 1))

        return model

    def get_values(self, contexts, return_std=True):

        uncertainties = np.sqrt(
            ((contexts[:, :, np.newaxis]*self.model['A_inv'][None, ]).sum(axis=1)*contexts).sum(-1))

        theta = self.model['theta'][None, ]

        estimated_rewards = (contexts*theta[:, :, 0]).sum(-1)

        if return_std:
            return estimated_rewards, uncertainties
        else:
            return estimated_rewards

    def initialize_w(self):
        self._model = None

        self.context_history = [[] for _ in range(1)]
        self.reward_history = [[] for _ in range(1)]
        self.action_history = []
        self.distance_history = []

    @property
    def base_str(self):
        s = ('LinUCB' + (str(self.delta) if self.delta != -1 else '') + ("_alph{}".format(self.alpha) if self.alpha != 1 else '')
             + ("_beta{}".format(self.beta) if self.beta != 1 else '')+('_earl'+str(self.early)
                                                                        if self.early is not False else '')+('shared' if self.shared_model else '')
             + ('monocontext' if self.shared_meta else '')+('log' if self.log_advice else ''))
        if self.adaptive > 0:
            s += f"d{self.d_delta}s{self.sigma}g{self.gamma}S{self.S}L{self.adaptive}"
        if self.proximity:
            s += "prox"+(self.proximity if self.proximity != "c" else "")
        if self.oracular:
            s += "OO"
        return s

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        centered_advice = self.advice - \
            (.5 if self.value_mode else 1/self.bandit.k)

        self.meta_contexts = np.concatenate(
            (centered_advice, np.ones((1, self.bandit.k))), axis=0).T

        mu, sigma = self.get_values(self.meta_contexts)

        return mu + sigma*self.beta

    def reset(self):
        super(LinUCB, self).reset()

    def observe(self, reward, arm, contexts={}, meta_context=None, meta_contexts=None, dummy=False, invert=True,  pi=None):
        if meta_contexts is None:
            meta_contexts = self.meta_contexts
        elif self.meta_contexts is None:
            self.meta_contexts = meta_contexts

        if meta_context is None:
            meta_context = meta_contexts[arm]
            self.context_history.append(meta_context)
            self.action_history.append(arm)

        action_context = meta_context[..., None]
        self.reward_history.append(reward)

        self.model['A_inv'] = SMInv(
            self.model['A_inv'], action_context, action_context, 1)

        self.model['b'] += (reward -
                            EXPECTED_AVG_REWARD) * action_context

        self.model['theta'] = (
            self.model['A_inv'].dot(self.model['b']))

        self.t += 1

    def get_weights(self, context, full=False):
        return self.model['theta']
