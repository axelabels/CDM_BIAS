
from copy import deepcopy
from numbers import Number
from sklearn.linear_model import Ridge
from expert import *

EXPECTED_AVG_REWARD = .5


class Collective(Agent):
    def __init__(self, bandit, policy, n_experts,  gamma=None,  name=None,   alpha=1, beta=1, expert_spread='normal',
                 ):

        super(Collective, self).__init__(bandit, policy)
        self.expert_spread = expert_spread

        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

        self.k = self.bandit.k
        self.n = self.n_experts = n_experts
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
        return np.ones((self.n, self.bandit.k))/self.n

    def __str__(self):
        if self.name is not None:
            return (self.name)
        return (self.short_name())

    def set_name(self, name):
        self.name = name

    def choose(self, advice, greedy=False):
        return self.policy.choose(self, advice, greedy=greedy)

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

    def probabilities(self, contexts):
        self.advice = np.copy(contexts['advice'])
        assert self.advice.shape == (self.n_experts, self.bandit.k,)
        if not np.allclose(np.sum(self.advice, axis=1), 1):
            self.advice = greedy_choice(self.advice, axis=1)

        w = self.get_weights(contexts)
        self._probabilities = np.sum(w * self.advice, axis=0)

        return self._probabilities

    def short_name(self):
        return "EXP4-IX"

    def initialize_w(self):
        self.e_sum = 1
        self.context_history = []
        self.w = np.zeros(self.n)

    def reset(self):
        super(Exp4, self).reset()

    def get_weights(self, contexts):

        w = np.repeat(self.w[:, np.newaxis], self.bandit.k, axis=1)
        w = softmax(w, axis=0)

        return w

    def observe(self, reward, arm):
        assert np.allclose(np.sum(self.advice, axis=1),
                           1), "expected probability advice"

        x_t = self.advice[:, arm] * (1-reward)
        y_t = x_t / (self.policy.pi[arm]+self.gamma)

        self.e_sum += np.sum(np.max(self.advice, axis=0))

        numerator = self.e_sum
        lr = np.sqrt(np.log(self.n) / numerator)
        self.w -= (lr) * y_t

        self.t += 1


class SafeFalcon(Collective):
    def __init__(self, bandit, policy, n_experts, n_trials, name=None,  expert_spread='normal', mis=None,
                 confidence_weight=100, alpha=1, full=False, beta=0.85542707*20,
                 crop=1, prefix='W',):
        self.crop = crop
        self.prefix = prefix
        self.e_sum = 1
        self.full = full
        self.n_trials = n_trials
        self.beta = beta
        self.mis = mis
        self.n_experts = n_experts
        self.confidence_weight = confidence_weight
        super().__init__(bandit, policy,
                         n_experts,  name=name,  beta=beta, alpha=alpha, expert_spread=expert_spread)

    def short_name(self):
        return f"Meta-CMAB (FALCON)"

    def initialize_w(self):
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

    def reset(self):
        super().reset()

    def choose_safe(self, m, reward_history, l):

        sub = np.sqrt(1/(2*len(reward_history))*np.log(m**2/self.delta))

        l_prime = np.mean(reward_history-sub+self.bandit.expected_reward)
        lm = max(l, l_prime)
        if lm != l:
            self.hat_m = m

        return lm, self.hat_m

    def e_rate(self, T, e):
        # er = self.n_experts*np.log(1/e)/T # (dlog(1/eps))/T
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

        self.policy.gamma = self.get_gamma()

        return self.policy.choose(self, advice)

    def value_estimates(self, contexts):
        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice) - self.bandit.expected_reward

        self.meta_contexts = centered_advice.T
        if self.epoch == 1:
            self._value_estimates = np.zeros(len(self.meta_contexts))
        else:

            if self.safe:
                self._value_estimates = self.model.predict(
                    self.meta_contexts/(self.n_experts**.5))
            else:
                self._value_estimates = self.model_history[self.hat_m].predict(
                    self.meta_contexts/(self.n_experts**.5))

        return self._value_estimates


class OnlineCover(Collective):
    def __init__(self, bandit, policy, experts, name=None, expert_spread='normal',
                 alpha=1,  cover_size=2, simple_first=True, epsilon=1e-10, nu=True, psi=0.01, cover=False, c_mu=0.05):

        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, expert_spread=expert_spread)

        self.simple_first = simple_first
        self.epsilon = epsilon
        self.nu = nu
        self.c_mu = c_mu
        self.experts = experts
        self.cover = cover
        self.policy.eps = 0 if self.nu else self.epsilon
        self._models = None
        self.context_dimension = experts+1
        self.psi = psi
        self.cover_size = cover_size

    def short_name(self):

        return f"Meta-CMAB (ILTCB)"

    @property
    def models(self):
        if self._models is None:

            self._models = self._init_model()

        return self._models

    def _init_model(self):
        self.online_models = [MultiOnlineRidge(
            self.alpha) for _ in range(self.cover_size)]
        return self.online_models

    def initialize_w(self):
        self._models = None

        self.all_context_history = []
        self.context_history = []
        self.reward_history = []
        self.all_reward_history = []
        self.action_history = []

    def probabilities(self, contexts):
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
        self.context_history.append(self.meta_contexts[arm])
        self.action_history.append(arm)

        new_contexts = self.meta_contexts[arm:arm+1]
        self.all_context_history.extend(new_contexts)

        self.reward_history.append(reward)

        greedy_actions = self.greedy_actions

        mu = self.epsilon/self.bandit.k

        smoothed_policy = self.epsilon + \
            (1-self.epsilon)*self.policy.pi if self.nu else self.policy.pi

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
            cost_sensitive_reward = chosen * (r_v ) / (smoothed_policy) + mu/p_i * self.psi
            

            round_rewards = cost_sensitive_reward[arm:arm+1]
            self.all_reward_history.extend(round_rewards)

            model.partial_fit(self.meta_contexts, cost_sensitive_reward)

        self.t += 1


class MAB(Collective):

    def __init__(self, bandit, policy, experts,  include_time=False, include_ctx=True, expert_spread='normal',
                 name=None,  gamma=None):

        super().__init__(bandit, policy,
                         experts, gamma=gamma,  name=name,  expert_spread=expert_spread)

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

        w = np.zeros((self.n, self.bandit.k))
        expert_values = np.random.beta(self.alphas, self.betas)
        self.chosen_expert = randargmax(expert_values)

        w[self.chosen_expert, :] = 1
        return w

    def observe(self, reward, arm):

        assert 0 <= reward <= 1, reward
        self.alphas[self.chosen_expert] += reward
        self.betas[self.chosen_expert] += 1 - reward

        self.t += 1


class MultiOnlineRidge():
    def __init__(self, alpha, prior_strength=0, log_A=False):
        self._model = None
        self.alpha = alpha
        self.log_A = log_A
        self.prior_strength = prior_strength

    @property
    def model(self):
        if self._model is None:

            self._model = self._init_model({})

        return self._model

    def _init_model(self, model):
        model['A'] = np.identity(self.context_dimension) * self.alpha
        model['A_inv'] = np.identity(self.context_dimension)/self.alpha
        model['b'] = np.zeros((self.context_dimension, 1))
        model['theta'] = np.zeros((self.context_dimension, 1))

        if self.prior_strength > 0:
            model["A_inv"] = np.identity(
                self.context_dimension)/self.alpha / self.prior_strength
            model['b'] += self.prior_strength/self.context_dimension
            model['theta'] = (model['A_inv'].dot(model['b']))
        self.A_history = [model['A']]

        return model

    def partial_fit(self, features, outcomes):

        self.context_dimension = np.shape(features)[1]
        for x, y in zip(features, outcomes):

            x = x[..., None]/((1*len(x))**0.5)

            if self.log_A:
                self.model['A'] += x.dot(x.T)

            self.model['A_inv'] = SMInv(
                self.model['A_inv'], x, x, 1)

            self.model['b'] += y * x

        if self.log_A:
            self.A_history.append(np.copy(self.model['A']))

        self.model['theta'] = (
            self.model['A_inv'].dot(self.model['b']))

    def uncertainties(self, features, sample=None):

        features = features/(features.shape[1]**.5)

        values = np.sqrt(
            ((features[:, :, np.newaxis]*self.model['A_inv'][None, ]).sum(axis=1)*features).sum(-1))
        thompson_noise = np.random.normal(
            np.zeros_like(values), values, size=values.shape)
        if sample is not None:
            values = sample*thompson_noise

        return np.asarray(values)

    def predict(self, features):
        self.context_dimension = features.shape[1]
        theta = self.model['theta'][None, ]
        features = features/((features.shape[1])**0.5)
        return (features*theta[:, :, 0]).sum(-1)


class LinUCB(Collective):
    def __init__(self, bandit, policy, experts, beta=1, name=None, trials=1, expert_spread='normal', epsilon=0,
                 alpha=10, min_one=False, fixed=False, residual=False, mode='UCB', weighted_update=False):

        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, beta=beta, expert_spread=expert_spread)
        self.trials = trials
        self.epsilon = epsilon
        self._model = None
        self.min_one = min_one
        self.fixed = fixed
        self.mode = mode
        self.weighted_update = weighted_update
        self.residual = residual
        self.context_dimension = experts+1

    def short_name(self):
        return f"Meta-CMAB (LinUCB)"
        
    @property
    def model(self):
        if self._model is None:
            self.counts = {}
            self._model = MultiOnlineRidge(self.alpha, log_A=self.epsilon > 0)

        return self._model

    def get_values(self, contexts, return_std=True):

        estimated_rewards = self.model.predict(
            contexts)
        if return_std:
            uncertainties = self.model.uncertainties(
                contexts, sample=1 if (self.mode == 'TS') else None)

            return estimated_rewards, uncertainties, 0
        else:
            return estimated_rewards

    def initialize_w(self):
        self._model = None

        self.context_history = []
        self.full_context_history = []
        self.reward_history = []
        self.action_history = []

        self.selection_history = []

    def value_estimates(self, contexts, mu_only=False):
        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice) - self.bandit.expected_reward

        self.meta_contexts = centered_advice.T

        mu, sigma, eps_sigma = self.get_values(self.meta_contexts)
        if mu_only:
            return mu

        assert self.context_dimension > 0

        if self.mode == 'TS' and not isinstance(self.beta, Number):

            beta1 = np.sqrt(.5*np.log(2*self.trials*self.bandit.k*self.trials))
            beta3 = 0 if self.t == 0 else np.sqrt(
                max(0, 2*np.log(self.trials)+np.log(1/np.linalg.det(self.model.model["A_inv"]))))
            beta4 = 1

            adapt_beta = np.array([0.00276422,  0.00448784, 0.04283073]).dot(
                [beta1, beta3, beta4])
            return mu + sigma*adapt_beta

        else:

            return mu + sigma*self.beta+eps_sigma

    def reset(self):
        super().reset()

    def observe(self, reward, arm):
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


class SupLinUCBVar(Collective):
    def __init__(self, bandit, policy, experts, beta=.1, name=None, expert_spread='normal', p=1,
                 alpha=1, ):

        self.bandit = bandit

        self._model = None
        self.context_dimension = experts
        self.alpha = alpha
        self.p = p
        self.beta = beta
        self.name = None
        self.ensemble = [EXPL3(bandit, policy, experts, beta, name, expert_spread, alpha=alpha, thresh=(2**-l))
                         for l in np.arange(1, max(1, (np.log(np.sqrt(self.bandit.k))/np.log(2))**p)+1e-10)]
        assert len(self.ensemble) >= 1

    def short_name(self):
        return f"Meta-CMAB (UCB)" 

    def reset(self):
        self.t = 0
        [e.reset() for e in self.ensemble]

    def choose(self, advice, greedy=False):
        mask = np.ones(self.bandit.k)

        for e in self.ensemble:
            self.active_expl = e
            values = e.value_estimates(advice, mask=mask)

            if e.exploratory:
                break

            mask = np.logical_and(
                mask, ((values) >= np.max(values)-e.thresh*self.beta))

            assert np.sum(mask) > 0, (np.max(values), e.thresh)

        return np.random.choice(self.bandit.k, p=greedy_choice(values))

    def observe(self, reward, arm):

        for e in self.ensemble[::1]:
            e.meta_contexts = self.active_expl.meta_contexts
            e.observe(reward, arm, force=True)
            if e == self.active_expl:
                break
        self.t += 1


class EXPL3(Collective):
    def __init__(self, bandit, policy, experts, beta=1, name=None, expert_spread='normal', thresh=0,
                 alpha=1, ):

        super().__init__(bandit, policy,
                         experts, name=name, alpha=alpha, beta=beta, expert_spread=expert_spread)
        self.thresh = thresh
        self._model = None
        self.context_dimension = experts+1

    @property
    def model(self):
        if self._model is None:
            self._model = MultiOnlineRidge(self.alpha)

        return self._model

    def get_values(self, contexts, return_std=True):

        estimated_rewards = self.model.predict(contexts)

        if return_std:
            uncertainties = self.model.uncertainties(contexts)
            return estimated_rewards, uncertainties
        else:
            return estimated_rewards

    def initialize_w(self):
        self._model = None

        self.context_history = []
        self.reward_history = []
        self.action_history = []

    def value_estimates(self, contexts, mask=None):

        self.advice = np.copy(contexts['advice'])

        centered_advice = np.array(self.advice) - self.bandit.expected_reward

        self.meta_contexts = centered_advice.T

        mu, sigma = self.get_values(self.meta_contexts)
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

    def observe(self, reward, arm, force=False):
        self.context_history.append(self.meta_contexts[arm])
        self.action_history.append(arm)

        mu, sigma = self.get_values(self.meta_contexts)
        if np.max(sigma) > self.thresh or force:
            action_context = self.meta_contexts[arm]

            self.model.partial_fit(
                [action_context], [reward-self.bandit.expected_reward])

        self.t += 1
