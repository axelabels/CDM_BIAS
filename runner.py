
import random
import h5py
from collections import defaultdict
from itertools import product
import shutil
import sys
import numpy as np
import pandas as pd
from agent import *
from bandit import *
from policy import *
import os
from tqdm import tqdm


def generate_expert(bandit, i):
    experts = [OracleExpert(bandit, GreedyPolicy())]
    return experts[i % len(experts)]


def generate_experts(n, bandit):
    return [generate_expert(bandit, i) for i in range(n)]


def initialize_experiment(bandit, agents, experts, seed,  average_expert_distance):

    np.random.seed(seed)

    bandit.reset()
    [e.reset() for e in experts]
    for agent in agents:
        agent.reset()

    for i, e in (list(enumerate(experts))):
        e.prior_bandit = bandit

    bandit.cache_contexts(n_trials, seed)

    truth = bandit.cached_values.flatten()
    spread_variance, n_clusters, desired_var, desired_covar = average_expert_distance

    if spread_variance:
        expert_var = np.linspace(0, desired_var, n_experts)
    else:
        expert_var = np.zeros(n_experts)+desired_var

    if bandit.problem == 'classification':
        expert_var = np.array(sorted(expert_var*.8+.2))
    else:
        expert_var = np.array(sorted(expert_var*.6+.2))

    advice = np.zeros((n_experts,)+truth.shape)
    advice_reshaped = advice.reshape(
        (n_experts,)+bandit.cached_values.shape)

    if np.mean(expert_var) == 0:
        advice[:] = truth+0

    else:
        for i in range(n_clusters):

            if i == 0:
                lo = 0
                hi = int(np.round(n_experts*1/4))
            else:
                lo = int(np.round(n_experts*1/4))
                hi = n_experts
            if i == 0:
                common_var = expert_var[0]
            else:
                common_var = expert_var[-1]

            if bandit.problem == 'classification':
                r_idx = np.random.choice(len(bandit.cached_values), size=int(
                    len(bandit.cached_values)*common_var), replace=False)
                common_errors = bandit.cached_values + 0
                common_errors[r_idx] = bandit.generate_random_values(
                    bandit.cached_values.shape)[r_idx]
            else:
                common_errors = get_err_advice(
                    truth, common_var).reshape(bandit.cached_values.shape)

            for n in range(lo, hi):
                advice_reshaped[n] = bandit.cached_values+0
                e_idx = np.random.choice(len(bandit.cached_values), size=int(
                    len(bandit.cached_values)*expert_var[n]), replace=False)

                expert_erroneous_advice = get_err_advice(
                    truth, expert_var[n]).reshape(bandit.cached_values.shape)

                if bandit.problem == 'classification':
                    advice_reshaped[n, e_idx] = bandit.generate_random_values(
                        bandit.cached_values.shape)[e_idx]

                    covar_idx = np.random.choice(len(bandit.cached_values),
                                                 size=int(len(bandit.cached_values)*(desired_covar)), replace=False)

                    advice_reshaped[n, covar_idx] = common_errors[covar_idx]+0
                else:
                    advice_reshaped[n] = expert_erroneous_advice

                    weights = np.random.choice(
                        [0, 1], p=[1-desired_covar, desired_covar], size=advice[n].shape)
                    advice[n] = advice[n] * \
                        (1-weights) + common_errors.flatten()*weights

    for i, e in (enumerate(experts)):
        e.cache_predictions(bandit, n_trials)
        e.cached_predictions[:, 0] = advice[i].reshape(
            bandit.cached_values.shape)


seed = int(sys.argv[1])
np.random.seed(seed)
random.seed(seed)
n_experiments = 1
experiment_seeds = np.random.randint(
    0, np.iinfo(np.int32).max, size=n_experiments)
experiment_seeds[0] = seed

problems = ['classification', 'regression']


K_N_configurations = ((16, 16), (16, 32), (16, 64), (32, 16), (64, 16),
                      (128, 16), (16, 128),
                      (16, 8), (8, 16), (16, 4), (4, 16))
trials = [10, 100, 1000, 10000]
alphas = [1]
clusters = [2]
desired_variances = [0.1, 0.3, 0.5, 0.7, .9]
desired_covariances = [0.1, 0.5, .9]
bernoulli_rewards = [True]
spread_variances = [True, False]  # heterogeneous, homogeneous

all_configurations = list(product(K_N_configurations, spread_variances, clusters, desired_variances, desired_covariances,
                          problems,  trials, bernoulli_rewards, alphas))

print(len(all_configurations))
all_configurations = sorted(all_configurations)
np.random.shuffle(all_configurations)
total_trials = np.sum([a[-3] for a in all_configurations])
print(total_trials)
overall_bar = tqdm(total=n_experiments*total_trials,
                   smoothing=0, desc='Overal progress')

output_folder = os.path.join(os.getenv('VSC_SCRATCH') or "", "results/")
os.makedirs(output_folder, exist_ok=True)

for experiment in range(n_experiments):
    h5f_filename_tmp = output_folder + f'{seed}_{experiment}_tmp.hdf5'
    h5f_filename = output_folder + f'{seed}_{experiment}.hdf5'
    h5f = h5py.File(h5f_filename_tmp, 'a')

    for ((n_arms, n_experts), spread_variance, n_clusters, desired_var, desired_covar, problem, n_trials, is_bernoulli, alpha) in all_configurations:

        key_str = "_".join(map(str, ((n_arms, n_experts), spread_variance, n_clusters, desired_var,
                                     desired_covar, problem, n_trials, is_bernoulli, alpha, experiment, seed)))

        if key_str in h5f:
            overall_bar.update(n_trials)
            continue

        data = []
        np.random.seed(experiment_seeds[experiment])
        random.seed(experiment_seeds[experiment])

        bandit = ArtificialBandit(
            n_arms=n_arms, problem=problem, bernoulli=is_bernoulli)

        overall_bar.set_description("generating experts")
        experts = generate_experts(n_experts, bandit)

        overall_bar.set_description("generating models")
        # set up agents
        agents = []
        agents += [Collective(bandit, GreedyPolicy(),
                              n_experts, )]

        # experts doubled for inversion method
        agents += [MAB(bandit, GreedyPolicy(), n_experts * 2, )]
        agents += [Exp4(bandit, Exp3Policy(), n_experts*2,
                        gamma=.5*(2*np.log(n_experts + 1)/(n_arms * n_trials)) ** (1 / 2))]

        agents += [SafeFalcon(bandit, SCBPolicy(None), n_experts,
                              n_trials=n_trials, alpha=alpha, )]

        agents += [OnlineCover(bandit, Exp3Policy(eps=None), n_experts,
                               epsilon=0.05*min(1/n_arms, 1/(np.sqrt(1*n_arms)))*n_arms)]

        agents += [SupLinUCBVar(bandit, GreedyPolicy(),
                                n_experts, alpha=alpha)]

        agents += [LinUCB(bandit, GreedyPolicy(), n_experts, beta=0,
                          alpha=alpha, fixed=True)]

        overall_bar.set_description(
            f"Initializing experiment {experiment} S:{problem} N:{n_experts}, K:{n_arms}, Δ:{spread_variance,n_clusters,desired_var,desired_covar},  experts")

        # set up experiment (initializes bandits and experts)
        initialize_experiment(bandit, agents, experts, experiment_seeds[experiment], (
            spread_variance, n_clusters, desired_var, desired_covar))

        cached_advice = np.array([e.cached_predictions for e in experts])[
            :, :, 0].reshape((n_experts, -1))

        optimal_linear_agent = agents[-1]
        optimal_linear_agent.name = 'Linear Optimum'
        optimal_linear_agent.model.context_dimension = n_experts
        X = ((cached_advice.T-bandit.expected_reward)/(n_experts**.5))
        Y = bandit.cached_values.flatten()-bandit.expected_reward

        unconstrained_ridge = Ridge(alpha=alpha, fit_intercept=False).fit(X, Y)

        misspecification = np.abs((unconstrained_ridge.predict(X)-Y)).mean()

        optimal_linear_agent.model.model['theta'] = unconstrained_ridge.coef_.reshape(
            (-1, 1))

        choices = np.random.choice(
            n_trials*n_arms, size=n_trials//2, replace=False)
        cov_matrix = (
            np.cov(cached_advice[:, choices]-bandit.cached_rewards.flatten()[choices]))

        variance = np.diag(cov_matrix).mean()
        covariance = cov_matrix[~np.eye(
            cov_matrix.shape[0], dtype=bool)].mean()

        overall_bar.set_description(
            f"Simulating experiment {experiment} T:{n_trials} B:{is_bernoulli} S:{problem} N:{n_experts}, K:{n_arms}, Δ:{spread_variance,n_clusters,desired_var,desired_covar}, experts")

        # run experiment
        results = np.zeros((n_experts+len(agents)+2, n_trials))
        advice_history = []
        value_history = []
        step_seeds = np.random.randint(
            0, np.iinfo(np.int32).max, size=n_trials)
        for t in range(n_trials):

            # Get current context and expert advice
            np.random.seed(step_seeds[t])
            bandit.observe_contexts(cache_index=t)
            sampled_rewards = bandit.sample(cache_index=t)
            advice = np.array(
                [e.value_estimates(cache_index=t) for e in experts])
            advice_history.extend(advice.T)
            value_history.extend(bandit.action_values)

            # Choose action, log reward, and update each agent
            meta_context = {'advice': advice}
            expanded_meta_context = {'advice': np.vstack([advice, 1-advice])}

            # Play one step for all aggregation algorithms on current advice
            for n, agent in enumerate(agents):
                # ensure random state is identical
                np.random.seed(step_seeds[t])

                if type(agent) == OnlineCover:
                    agent.epsilon = agent.c_mu * \
                        min(1/n_arms, 1/(np.sqrt((t+1)*n_arms)))*n_arms

                if type(agent) in (MAB, Exp4):
                    action = agent.choose(expanded_meta_context)
                else:
                    action = agent.choose(meta_context)
                reward = sampled_rewards[action]
                results[n_experts+n, t] = reward
                agent.observe(reward, action)

            # Log expert performance
            for e in range(n_experts):
                np.random.seed(step_seeds[t])
                results[e, t] = bandit.action_values.dot(
                    greedy_choice(advice[e]))

            # Best expected reward
            results[-1, t] = np.max(bandit.action_values)
            results[-2, t] = np.mean(bandit.action_values)  # Random policy

            overall_bar.update()

        # log results, averaged into chunks
        chunks = 1

        # experts
        for sort_n, n in enumerate(sorted(range(n_experts), key=lambda n: np.mean(results[n][:]), reverse=True)):
            for c in range(chunks):
                data.append([seed, is_bernoulli, c/chunks*n_trials, f"expert {sort_n}", experiment, np.mean(np.array_split(results[n], chunks)[c]), "value", n_arms, n_experts, (
                    n_arms, n_experts),  spread_variance, n_clusters, desired_var, desired_covar, problem, covariance, variance, n_trials, alpha])

                if sort_n == 0 or sort_n == n_experts-1:
                    data.append([seed, is_bernoulli, c/chunks*n_trials, "best" if sort_n == 0 else "worst", experiment, np.mean(np.array_split(results[n], chunks)[c]), "value", n_arms,
                                n_experts, (n_arms, n_experts),  spread_variance, n_clusters, desired_var, desired_covar, problem, covariance, variance, n_trials, alpha])

        # aggregation algorithms
        for c in range(chunks):
            for n, agent in enumerate(agents):
                agent_score = np.mean(np.array_split(
                    results[n_experts+n], chunks)[c])
                data.append([seed, is_bernoulli, c/chunks*n_trials, str(agent), experiment, agent_score, "value", n_arms, n_experts, (n_arms, n_experts),
                             spread_variance, n_clusters, desired_var, desired_covar, problem, covariance, variance, n_trials, alpha])

            data.append([seed, is_bernoulli, c/chunks*n_trials, f"random", experiment, np.mean(np.array_split(results[-2], chunks)[c]), "value", n_arms, n_experts,
                        (n_arms, n_experts),  spread_variance, n_clusters, desired_var, desired_covar, problem, covariance, variance, n_trials, alpha])
            data.append([seed, is_bernoulli, c/chunks*n_trials, f"optimal", experiment, np.mean(np.array_split(results[-1], chunks)[c]), "value", n_arms, n_experts,
                        (n_arms, n_experts),  spread_variance, n_clusters, desired_var, desired_covar, problem, covariance, variance, n_trials, alpha])

        df = pd.DataFrame(data, columns=["seed", "bernoulli", 't', "algorithm", "experiment", "average reward", "advice type", "K", "N", "(K,N)",
                                         "spread_variance", "n_clusters", "desired_var", "desired_covar", "problem", "covariance", "variance", "n_trials", "alpha"])
        df = df.sort_values(by=["seed", "bernoulli", 't', 'algorithm', 'experiment', 'advice type', 'K', 'N',
                                '(K,N)',  "spread_variance", "n_clusters", "desired_var", "desired_covar", "problem", "covariance", "n_trials", "alpha"])

        reshaped_values = df.loc[(df.n_trials == n_trials),
                                 'average reward'].values.reshape((chunks, -1)).T
        cum_values = np.cumsum(reshaped_values, axis=1) / \
            (np.arange(reshaped_values.shape[1])+1)[None]
        df.loc[(df.n_trials == n_trials),
               'cum reward'] = cum_values.T.flatten()

        sa, saType = df_to_sarray(df)
        h5f.create_dataset(key_str, data=sa, dtype=saType)

        # ensures outputs are flushed intermitently
        if (n_arms, n_experts) == K_N_configurations[0] and n_trials == trials[0]:
            h5f.close()
            shutil.copy(h5f_filename_tmp, h5f_filename)
            h5f = h5py.File(h5f_filename_tmp, 'a')
    h5f.close()
overall_bar.close()
