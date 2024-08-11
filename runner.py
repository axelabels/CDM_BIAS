
import random
import h5py
from collections import defaultdict
from itertools import product
import shutil
import sys
import numpy as np
import pandas as pd
from learner import *
from bandit import *
from policy import *
import os
from tqdm import tqdm


seed = int(sys.argv[1])
np.random.seed(seed)
random.seed(seed)
n_experiments = 1
experiment_seeds = np.random.randint(
    0, np.iinfo(np.int32).max, size=n_experiments)
experiment_seeds[0] = seed

problems = ['classification', 'regression']

K_N_configurations = ((16, 16), (16, 32), (16, 64), (32, 16), (64, 16),
                      (128, 16), (16, 128), (16, 8), (8, 16), (16, 4), (4, 16))
trials = [10, 100, 1000, 10000]
alphas = [1]
maximum_errors = [0.1, 0.3, 0.5, 0.7, .9]
cluster_correlations = [0.1, 0.5, .9]
expertise_distributions = ["heterogeneous", "homogeneous"]

all_configurations = sorted((product(K_N_configurations, expertise_distributions,  maximum_errors, cluster_correlations,
                                     problems, trials, alphas)))

np.random.shuffle(all_configurations)
total_trials = np.sum([a[-2] for a in all_configurations])
print("Running", len(all_configurations),
      "configurations, containing", total_trials, "steps in total")
overall_bar = tqdm(total=n_experiments*total_trials,
                   smoothing=0, desc='Overal progress')

output_folder = os.path.join(os.getenv('VSC_SCRATCH') or "", "results/")
os.makedirs(output_folder, exist_ok=True)


if __name__ == "__main__":
    for experiment in range(n_experiments):
        h5f_filename_tmp = output_folder + f'{seed}_{experiment}_tmp.hdf5'
        h5f_filename = output_folder + f'{seed}_{experiment}.hdf5'
        h5f = h5py.File(h5f_filename_tmp, 'a')

        for ((n_arms, n_experts), expertise_distribution,  maximum_error, within_cluster_correlation, problem, n_trials, alpha) in all_configurations:

            key_str = "_".join(map(str, ((n_arms, n_experts), expertise_distribution,  maximum_error,
                                         within_cluster_correlation, problem, n_trials,  alpha, experiment, seed)))

            if key_str in h5f:  # skip experiments that have already been collected
                overall_bar.update(n_trials)
                continue

            data = []
            np.random.seed(experiment_seeds[experiment])
            random.seed(experiment_seeds[experiment])

            bandit = ArtificialBandit(
                n_arms=n_arms, problem=problem, bernoulli=True)

            overall_bar.set_description("generating experts")
            experts = [Expert(bandit, GreedyPolicy())
                       for _ in range(n_experts)]

            overall_bar.set_description("generating models")

            agents = []

            # Average
            agents += [Average(bandit, GreedyPolicy(),
                                  n_experts, )]

            # Meta-MAB (KL-UCB)
            agents += [MAB(bandit, GreedyPolicy(), n_experts * 2,base="KL-UCB" )]

            # EXP4-IX
            agents += [Exp4(bandit, Exp3Policy(), n_experts * 2,
                            gamma=.5*(2*np.log(n_experts + 1)/(n_arms * n_trials)) ** (1 / 2))]

            # Meta-CMAB (FALCON)
            agents += [SafeFalcon(bandit, SCBPolicy(None), n_experts,
                                  n_trials=n_trials, alpha=alpha, )]

            # Meta-CMAB (ILTCB)
            agents += [OnlineCover(bandit, Exp3Policy(eps=None), n_experts,
                                 )]

            # Meta-CMAB (UCB)
            agents += [SupLinUCBVar(bandit, GreedyPolicy(),
                                    n_experts, alpha=alpha)]

            # dummy instance to be used as oracle
            agents += [LinUCB(bandit, GreedyPolicy(), n_experts, beta=0,
                              alpha=alpha, fixed=True)]

            overall_bar.set_description(
                f"Initializing experiment {experiment} S:{problem} N:{n_experts}, K:{n_arms}, Δ:{expertise_distribution,maximum_error,within_cluster_correlation},  experts")

            # set up experiment (initializes bandits and experts)
            generate_noisy_advice(bandit, agents, experts, n_trials, experiment_seeds[experiment],
                                  expertise_distribution,  maximum_error, within_cluster_correlation)

            # compute optimal weights in hindsight in order to estimate the performance of an oracle
            advice_matrix = np.array([e.cached_predictions for e in experts])[
                :, :].reshape((n_experts, -1))
            optimal_linear_agent = agents[-1]
            optimal_linear_agent.name = 'Linear Optimum'
            optimal_linear_agent.model.context_dimension = n_experts
            X = ((advice_matrix.T-bandit.expected_reward)/(n_experts**.5))
            Y = bandit.cached_values.flatten()-bandit.expected_reward

            model = Ridge(alpha=alpha, fit_intercept=False).fit(X, Y)
            misspecification = np.abs((model.predict(X)-Y)).mean()
            optimal_linear_agent.model.model['theta'] = model.coef_.reshape(
                (-1, 1))

            # estimate experts' variance and covariance
            choices = np.random.choice(
                n_trials*n_arms, size=n_trials//2, replace=False)
            cov_matrix = (
                np.cov(advice_matrix[:, choices]-bandit.cached_rewards.flatten()[choices]))

            variance = np.diag(cov_matrix).mean()
            covariance = cov_matrix[~np.eye(
                cov_matrix.shape[0], dtype=bool)].mean()

            overall_bar.set_description(
                f"Simulating experiment {experiment} T:{n_trials}  S:{problem} N:{n_experts}, K:{n_arms}, Δ:{expertise_distribution,maximum_error,within_cluster_correlation}, experts")

            # run experiment
            results = np.zeros((n_experts+len(agents)+2, n_trials))
            step_seeds = np.random.randint(
                0, np.iinfo(np.int32).max, size=n_trials)
            for t in range(n_trials):

                # Get current context and expert advice
                np.random.seed(step_seeds[t])
                bandit.observe_contexts(cache_index=t)
                sampled_rewards = bandit.sample(cache_index=t)
                advice = np.array(
                    [e.get_advice(cache_index=t) for e in experts])

                meta_context = {'advice': advice}

                # inversion method (see Section 2.3.1, paragraph "For larger biases,...")
                expanded_meta_context = {
                    'advice': np.vstack([advice, 1-advice])}

                # Choose action, log reward, and update each agent
                # Play one step for all aggregation algorithms on current advice
                for n, agent in enumerate(agents):
                    # ensure random state is identical
                    np.random.seed(step_seeds[t])

                    
                    if isinstance(agent, (MAB, Exp4)):
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
                    data.append([seed,  c/chunks*n_trials, f"expert {sort_n}", experiment, np.mean(np.array_split(results[n], chunks)[c]), "value", n_arms, n_experts, (
                        n_arms, n_experts),  expertise_distribution,  maximum_error, within_cluster_correlation, problem, covariance, variance, n_trials, alpha])

                    if sort_n == 0 or sort_n == n_experts-1:
                        data.append([seed,  c/chunks*n_trials, "best" if sort_n == 0 else "worst", experiment, np.mean(np.array_split(results[n], chunks)[c]), "value", n_arms,
                                    n_experts, (n_arms, n_experts),  expertise_distribution,  maximum_error, within_cluster_correlation, problem, covariance, variance, n_trials, alpha])

            # aggregation algorithms
            for c in range(chunks):
                for n, agent in enumerate(agents):
                    agent_score = np.mean(np.array_split(
                        results[n_experts+n], chunks)[c])
                    data.append([seed,  c/chunks*n_trials, str(agent), experiment, agent_score, "value", n_arms, n_experts, (n_arms, n_experts),
                                expertise_distribution,  maximum_error, within_cluster_correlation, problem, covariance, variance, n_trials, alpha])

                data.append([seed,  c/chunks*n_trials, f"random", experiment, np.mean(np.array_split(results[-2], chunks)[c]), "value", n_arms, n_experts,
                            (n_arms, n_experts),  expertise_distribution,  maximum_error, within_cluster_correlation, problem, covariance, variance, n_trials, alpha])
                data.append([seed,  c/chunks*n_trials, f"optimal", experiment, np.mean(np.array_split(results[-1], chunks)[c]), "value", n_arms, n_experts,
                            (n_arms, n_experts),  expertise_distribution,  maximum_error, within_cluster_correlation, problem, covariance, variance, n_trials, alpha])

            df = pd.DataFrame(data, columns=["seed",  't', "algorithm", "experiment", "average reward", "advice type", "K", "N", "(K,N)",
                                             "expertise_distribution",  "maximum_error", "within_cluster_correlation", "problem", "covariance", "variance", "n_trials", "alpha"])
            df = df.sort_values(by=["seed",  't', 'algorithm', 'experiment', 'advice type', 'K', 'N',
                                    '(K,N)',  "expertise_distribution",  "maximum_error", "within_cluster_correlation", "problem", "covariance", "n_trials", "alpha"])

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
