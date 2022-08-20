
import random
import h5py
from collections import defaultdict
from bandit import UPPER
from scipy.optimize import lsq_linear
from time import time
from collections import Counter
from itertools import product
from pathlib import Path
import shutil
import sys
import numpy as np
from agent import *
from bandit import *
from policy import *
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

# %%


def generate_expert(bandit, i, variance):
    experts = [OracleExpert(bandit, GreedyPolicy(), var=variance)]
    return experts[i % len(experts)]


def generate_experts(n, bandit, variance):
    return [generate_expert(bandit, i, variance) for i in range(n)]


# %%

def initialize_experiment(bandit, agents, experts, seed, n_prior_experiences, expert_spread, average_expert_distance, reset=True):

    np.random.seed(seed)
    if reset:
        bandit.reset()
        [e.reset() for e in experts]
    for agent in agents:
        agent.reset()
    if reset:

        for i, e in (list(enumerate(experts))):
            e.prior_bandit = bandit
                
        bandit.cache_contexts(n_trials, seed)

        truth = bandit.cached_values.flatten()
        variance_spread, n_clusters, desired_var, desired_covar = average_expert_distance

        cov_matr = np.zeros((n_experts, n_experts))
        for i in range(n_clusters):
            lo, hi = i*(n_experts//n_clusters), (i+1)*(n_experts//n_clusters)
            cov_matr[lo:hi, lo:hi] = desired_covar if desired_covar != 2 else np.random.uniform(
                size=cov_matr[lo:hi, lo:hi].shape)

        cov_matr[np.identity(n_experts, dtype=bool)] = 1

        cov_matr = np.tril(cov_matr) + np.tril(cov_matr, -1).T
        if variance_spread:
            expert_var = np.linspace(0, desired_var, n_experts)
        else:
            expert_var = np.zeros(n_experts)+desired_var
        if bandit.smoothness == 10:
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
                common_errors = bandit.cached_values + 0
                assert n_clusters == 2
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
                r_idx = np.random.choice(len(bandit.cached_values), size=int(
                    len(bandit.cached_values)*common_var), replace=False)

                random_erroneous_advice = get_err_advice(
                    truth, common_var).reshape(bandit.cached_values.shape)

                if bandit.smoothness == 10:

                    common_errors[r_idx] = bandit.generate_random_values(
                        bandit.cached_values.shape)[r_idx]
                else:

                    common_errors = random_erroneous_advice

                for n in range(lo, hi):
                    advice_reshaped[n] = bandit.cached_values+0
                    e_idx = np.random.choice(len(bandit.cached_values), size=int(
                        len(bandit.cached_values)*expert_var[n]), replace=False)

                    expert_erroneous_advice = get_err_advice(
                        truth, expert_var[n]).reshape(bandit.cached_values.shape)

                    if bandit.smoothness == 10:
                        advice_reshaped[n, e_idx] = bandit.generate_random_values(
                            bandit.cached_values.shape)[e_idx]
                        if n == 0:
                            continue
                        covar_idx = np.random.choice(len(bandit.cached_values),
                                                     size=int(len(bandit.cached_values)*(np.random.uniform() if desired_covar == 2 else desired_covar)), replace=False)

                        advice_reshaped[n,
                                        covar_idx] = common_errors[covar_idx]+0
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

smooths = smooth_defaults = [2., 10.][:]


K_N_configurations = ((16, 16), (16, 32), (16, 64), (32, 16), (64, 16),
                      (128, 16), (16, 128),
                       (16, 8), (8, 16), (16, 4), (4, 16))[:]

K_N_defaults = ((16, 16),)[:]
trial_defaults = trials = [10, 100,  ][:]
alpha_defaults = alphas = [1][:1]
n_prior_experiences = 1

clusters = [2, 1][:1]
clusters = clusters[seed % len(clusters):seed % len(clusters)+1]


desired_variances = [0.1, 0.3, 0.5, 0.7, .9][:]
desired_covariances = [0.1, 0.5, .9][:]
bernoulli_rewards = [True]

variance_spreads = [True, False][:]
prev_v = {}
sum_size = defaultdict(lambda: 0)
expert_configurations = [0][:]
configurations_defaults = [0][:]
all_configurations = []

all_configurations.extend(product(K_N_configurations, variance_spreads, clusters, desired_variances, desired_covariances,
                          smooth_defaults, configurations_defaults, trial_defaults, bernoulli_rewards, alpha_defaults))
all_configurations.extend(product(K_N_defaults, variance_spreads, clusters, desired_variances,
                          desired_covariances, smooths, configurations_defaults, trial_defaults, bernoulli_rewards, alpha_defaults))
all_configurations.extend(product(K_N_defaults, variance_spreads, clusters, desired_variances, desired_covariances,
                          smooth_defaults, expert_configurations, trial_defaults, bernoulli_rewards, alpha_defaults))
all_configurations.extend(product(K_N_defaults, variance_spreads, clusters, desired_variances,
                          desired_covariances, smooth_defaults, configurations_defaults, trials, bernoulli_rewards, alpha_defaults))
all_configurations.extend(product(K_N_defaults, variance_spreads, clusters, desired_variances,
                          desired_covariances, smooth_defaults, configurations_defaults, trial_defaults, bernoulli_rewards, alphas))
all_configurations = list(set(all_configurations))
print(len(all_configurations))
all_configurations = sorted(all_configurations)
np.random.shuffle(all_configurations)
total_trials = np.sum([a[-3] for a in all_configurations])
print(total_trials)
overall_bar = tqdm(total=n_experiments*total_trials,
                   smoothing=0, desc='Overal progress')

# on_cluster = os.getenv('VSC_SCRATCH') is not None
output_folder = os.path.join(os.getenv('VSC_SCRATCH') or "", "results_tmp/")
os.makedirs(output_folder, exist_ok=True)
running_dif = []
running_dif10 = []
running_dif1000 = []

for experiment in range(n_experiments):
    h5f_filename = output_folder + f'{seed}_{experiment}.hdf5'
    h5f_filename_bak = output_folder + f'{seed}_{experiment}_bak.hdf5'
    h5f = h5py.File(h5f_filename, 'a')
    for ((n_arms, n_experts), variance_spread, n_clusters, desired_var, desired_covar, smoothness, expert_conf, n_trials, is_bernoulli, ALPHA) in all_configurations:
        data = []
        np.random.seed(experiment_seeds[experiment])
        random.seed(experiment_seeds[experiment])
        filename = output_folder + "_".join(map(str, ((n_arms, n_experts), variance_spread, n_clusters, desired_var,
                                            desired_covar, smoothness, expert_conf, n_trials, is_bernoulli, ALPHA, experiment, seed)))+".csv"

        
        bandit = ArtificialBandit(n_arms=n_arms, smoothness=smoothness, bernoulli=is_bernoulli)
        # bandit.get
        overall_bar.set_description("generating experts")
        experts = generate_experts(n_experts, bandit, variance=0.0000001)

        overall_bar.set_description("generating models")
        # set up agents
        agents = []
        agents += [Collective(bandit, GreedyPolicy(),
                              n_experts, )]

        # experts doubled for inversion method
        agents += [MAB(bandit, GreedyPolicy(), n_experts *
                       2, )]
        agents += [Exp4(bandit, Exp3Policy(), n_experts*2, 
                        gamma=.5*(2*np.log(n_experts + 1)/(n_arms * n_trials)) ** (1 / 2))]

        agents += [SafeFalcon(bandit, SCBPolicy(None), n_experts,
                              n_trials=n_trials, alpha=ALPHA, )]

        agents += [OnlineCover(bandit, Exp3Policy(eps=None), n_experts, epsilon=0.05*min(1/n_arms, 1/(np.sqrt(1*n_arms)))*n_arms)]

        agents += [SupLinUCBVar(bandit, GreedyPolicy(), n_experts,alpha=ALPHA)]

        agents += [LinUCB(bandit, GreedyPolicy(), n_experts, beta=0,
                          alpha=ALPHA, fixed=True)]
        agents += [LinUCB(bandit, GreedyPolicy(), n_experts, beta=0,
                          alpha=ALPHA,  fixed=True)]

        overall_bar.set_description(
                f"Initializing experiment {experiment} B:{is_bernoulli} S:{smoothness} N:{n_experts}, K:{n_arms}, Δ:{variance_spread,n_clusters,desired_var,desired_covar}, {expert_conf} experts")

        # set up experiment (initializes bandits and experts)
        initialize_experiment(bandit, agents, experts, experiment_seeds[experiment], n_prior_experiences, expert_conf, (
            variance_spread, n_clusters, desired_var, desired_covar), reset=True)

        cached_advice = np.array([e.cached_predictions for e in experts])[
            :, :, 0].reshape((n_experts, -1))

        optimal_linear_agent = agents[-1]
        optimal_linear_agent.model.context_dimension = n_experts
        X = ((cached_advice.T-bandit.expected_reward)/(n_experts**.5))
        Y = bandit.cached_values.flatten()-bandit.expected_reward

        unconstrained_ridge = Ridge(alpha=ALPHA, fit_intercept=False).fit(X, Y)

        misspecification = np.abs((unconstrained_ridge.predict(X)-Y)).mean()

        optimal_linear_agent.model.model['theta'] = unconstrained_ridge.coef_.reshape(
            (-1, 1))

        optimal_weighted_average_agent = agents[-2]
        
        # from constrained_linear_regression import ConstrainedLinearRegression
        # # force weighted average
        # X = np.hstack((X, -X))*(n_experts**.5)/((2*n_experts)**.5)
        # X = np.vstack(
        #     (X, (np.ones_like(X)-bandit.expected_reward)/((2*n_experts)**.5)))
        # Y = np.hstack((Y, np.ones_like(Y)-bandit.expected_reward))
        # print(X.shape, Y.shape)
        # constrained_model = ConstrainedLinearRegression(
        #     nonnegative=True, ridge=ALPHA, fit_intercept=False).fit(X, Y)  # ,max_coef=np.ones(n_experts*2))

        # optimal_weighted_average_agent.model.context_dimension = X.shape[1]

        # optimal_weighted_average_agent.model.model['theta'] = (
        #     constrained_model.coef_)[:, None]
        optimal_weighted_average_agent.name = str(
            optimal_weighted_average_agent)+'_CON'

        choices = np.random.choice(
            n_trials*n_arms, size=n_trials//2, replace=False)
        cov_matrix = (
            np.cov(cached_advice[:, choices]-bandit.cached_rewards.flatten()[choices]))

        variance = np.diag(cov_matrix).mean()
        covariance = cov_matrix[~np.eye(
            cov_matrix.shape[0], dtype=bool)].mean()

        overall_bar.set_description(
                f"Simulating experiment {experiment} T:{n_trials} B:{is_bernoulli} S:{smoothness} N:{n_experts}, K:{n_arms}, Δ:{variance_spread,n_clusters,desired_var,desired_covar}, {expert_conf} experts")

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
                [e.value_estimates(cache_index=t, return_std=True) for e in experts])[:, 0]
            advice_history.extend(advice.T)
            value_history.extend(bandit.action_values)

            # Choose action, log reward, and update each agent
            meta_context = {'advice': advice}
            expanded_meta_context = {'advice': np.vstack([advice, 1-advice])}

            # Play one step for all aggregation algorithms on current advice
            for n, agent in enumerate(agents):
                np.random.seed(step_seeds[t]) # ensure random state is identical 

                if type(agent) == OnlineCover:
                    agent.epsilon = agent.c_mu * \
                        min(1/n_arms, 1/(np.sqrt((t+1)*n_arms)))*n_arms

                if type(agent) in (MAB, Exp4) or agent == optimal_weighted_average_agent:  
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

        ## experts
        for sort_n, n in enumerate(sorted(range(n_experts), key=lambda n: np.mean(results[n][:]), reverse=True)):
            for c in range(chunks):
                data.append([seed, is_bernoulli, c/chunks*n_trials, f"expert {sort_n}", experiment, np.mean(np.array_split(results[n], chunks)[c]), "value", n_arms, n_experts, (
                    n_arms, n_experts), expert_conf, variance_spread, n_clusters, desired_var, desired_covar, smoothness, covariance, variance, n_trials, ALPHA])

                if sort_n == 0 or sort_n == n_experts-1:
                    data.append([seed, is_bernoulli, c/chunks*n_trials, "best" if sort_n == 0 else "worst", experiment, np.mean(np.array_split(results[n], chunks)[c]), "value", n_arms,
                                n_experts, (n_arms, n_experts), expert_conf, variance_spread, n_clusters, desired_var, desired_covar, smoothness, covariance, variance, n_trials, ALPHA])

        ## aggregation algorithms
        for c in range(chunks):
            for n, agent in enumerate(agents):
                agent_score = np.mean(np.array_split(
                    results[n_experts+n], chunks)[c])
                data.append([seed, is_bernoulli, c/chunks*n_trials, str(agent), experiment, agent_score, "value", n_arms, n_experts, (n_arms, n_experts),
                            expert_conf, variance_spread, n_clusters, desired_var, desired_covar, smoothness, covariance, variance, n_trials, ALPHA])

            data.append([seed, is_bernoulli, c/chunks*n_trials, f"random", experiment, np.mean(np.array_split(results[-2], chunks)[c]), "value", n_arms, n_experts,
                        (n_arms, n_experts), expert_conf, variance_spread, n_clusters, desired_var, desired_covar, smoothness, covariance, variance, n_trials, ALPHA])
            data.append([seed, is_bernoulli, c/chunks*n_trials, f"optimal", experiment, np.mean(np.array_split(results[-1], chunks)[c]), "value", n_arms, n_experts,
                        (n_arms, n_experts), expert_conf, variance_spread, n_clusters, desired_var, desired_covar, smoothness, covariance, variance, n_trials, ALPHA])

        df = pd.DataFrame(data, columns=["seed", "bernoulli", 't', "algorithm", "experiment", "average reward", "advice type", "K", "N", "(K,N)",
                          "configuration", "variance_spread", "n_clusters", "desired_var", "desired_covar", "shape", "covariance", "variance", "n_trials", "alpha"])
        df = df.sort_values(by=["seed", "bernoulli", 't', 'algorithm', 'experiment', 'advice type', 'K', 'N',
                                '(K,N)', 'configuration', "variance_spread", "n_clusters", "desired_var", "desired_covar", "shape", "covariance", "n_trials", "alpha"])

        reshaped_values = df.loc[(df.n_trials == n_trials),
                                 'average reward'].values.reshape((chunks, -1)).T
        cum_values = np.cumsum(reshaped_values, axis=1) / \
            (np.arange(reshaped_values.shape[1])+1)[None]
        df.loc[(df.n_trials == n_trials),
               'cum reward'] = cum_values.T.flatten()

        if filename in h5f:
            stored_df = pd.DataFrame(h5f[filename][:])

            sa, saType = df_to_sarray(df)
            assert len(stored_df.values) == len(sa)
            for a, b in zip(stored_df.values, sa):
                assert len(a) == len(b)
                for c, d in zip(a, b):

                    assert c == d, (c, d)

            print("saved data matches", filename)
            overall_bar.update(n_trials)
            continue

        sa, saType = df_to_sarray(df)
        h5f.create_dataset(filename, data=sa, dtype=saType)

        if n_arms == 16 and n_experts == 16 and n_trials == 100:
            h5f.close()
            shutil.copy(h5f_filename, h5f_filename_bak)
            h5f = h5py.File(h5f_filename, 'a')
    h5f.close()
overall_bar.close()
