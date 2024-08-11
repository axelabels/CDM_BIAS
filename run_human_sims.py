import os
import pandas as pd

from collections import defaultdict
from tqdm import tqdm

from learner import *

from collections import defaultdict
from tqdm import tqdm
from policy import GreedyPolicy, Exp3Policy, SCBPolicy, Policy
from expert import Agent


import numpy as np
class DummyBandit():
    def __init__(self, k):
        self.k = k
        self.cache_id = 0
        self.expected_reward = 0


def load_data(dataset):
    if dataset == "dunchenne":
        df = pd.read_csv("dunchenne/mturklabels.txt", sep=" ",
                         names=["task", "workerID", "label"])
    if dataset == "trec":
        df = pd.read_csv("trec-rf10-crowd/trec-rf10-data.txt", sep="\t")
        df['task'] = df.topicID.map(str)+'_'+df.docID.map(str)
    if dataset == "snow":
        dfs = []
        for filename in os.listdir("snow2008"):
            taskname = filename.split(".")[0]
            if taskname not in ("anger", "disgust", "fear", "joy", "sadness", "surprise",):
                continue
            dfs.append(pd.read_csv(f"snow2008/{filename}", sep="\t"))
            dfs[-1]['task'] = filename+"_"+dfs[-1]['orig_id'].map(str)
            if len(dfs) > 1 and set(dfs[-1]['!amt_worker_ids'].unique()) != set(dfs[-2]['!amt_worker_ids'].unique()):
                dfs = dfs[:-1]
                break

        df = pd.concat(dfs)
        df.columns = ["topicID", "workerID",
                      "orig_id", "label", "gold", "task"]

    df = df.drop_duplicates()
    return df


for mode in ("dunchenne", "snow", "trec"):

    df = load_data(mode)

    # compute for each subset of workers how many problem instances they have in common
    counts = defaultdict(lambda: 0)
    tasks = []
    for i, task in tqdm(list(enumerate(df.task.unique()))):
        workers = df[df.task == task].workerID.drop_duplicates().unique()
        if len(workers) > 100:
            continue
        if len(workers) < 5:
            continue
        for subset in powerset(workers):
            subset = tuple(sorted(subset))
            if len(subset) != (5 if mode != "snow" else 6):
                continue
            counts[subset] += 1

    # prioritize expert sets that have at least 100 cases in common
    expert_subsets = sorted(counts, key=lambda k: (
        counts.get(k) > 100, len(k), counts.get(k)), reverse=True)

    stdevs = []
    covs = []
    bests = []
    means = []
    all_expert_scores = []
    scores = defaultdict(list)
    SIMULATIONS = 100
    for subset in tqdm(expert_subsets[:SIMULATIONS]):

        # tasks shared by all workers, i.e., the problems all workers in subset answered
        tasks = []
        sub_df = df[(df.workerID.isin(subset))]
        for i, task in (list(enumerate(sub_df.task.unique()))):
            if len(sub_df[(sub_df.task == task)]) == len(subset):
                tasks.append(task)

        # get subset of workers' answers to the tasks they have in common then transform it into advice and gold labels (i.e., reward)
        answers = df[(df.task.isin(tasks)) & (
            df.workerID.isin(subset))].drop_duplicates()

        if mode == "snow":
            arms = set([a.split('_')[0] for a in answers.task.unique()])

            data = answers.sort_values(
                by=["orig_id", "workerID", "task"]).values
            data = data.reshape((-1, len(subset), len(arms), 6))
            gold_labels = (data[:, 0, :, 4]/100).astype(float)
            data = (data[:, :, :, 3]/100).astype(float)

            label_count = data.shape[-1]

            advice = data
        else:
            data = answers.sort_values(by=["task", "workerID"]).values
            data = data.reshape(
                (len(answers.task.unique()), len(subset), -1))
            answer_idx = {"dunchenne": 2, "trec": 4}[mode]
            label_count = {"dunchenne": 2, "trec": 4}[mode]
            data = data[:, :, answer_idx]
            data = data[data.min(axis=1) >= 0].astype(int)
            if len(data) < 20:
                continue

            advice = np.zeros(data.shape+(label_count,))

            for i in range(len(advice)):
                advice[i][np.arange(len(subset)), data[i]] = 1

        regrets = defaultdict(list)

        np.random.seed(0)

        averaged_expert_regret = np.zeros(
            (len(advice), len(subset)-(mode != "snow")))

        targets = range(len(subset)) if mode == "treck" else (0,)

        for target in (targets):
            if mode == "snow":
                truth = gold_labels
            elif mode == 'dunchenne':
                tmp_target = np.random.choice(len(subset))

                truth = advice[:, tmp_target]
                advice = advice[:, [i for i in range(
                    len(subset)) if i != tmp_target]]
            else:
                truth = advice[:, target]
                advice = advice[:, [i for i in range(
                    len(subset)) if i != target]]

            advice.shape, truth.shape
            num_experts = advice.shape[1]

            RUNS = 10 # do 10 simulations for each expert subset
            for r in (range(RUNS)):

                np.random.seed(target*RUNS+r)
                instance_order = np.random.choice(
                    len(advice), size=len(advice), replace=False)
                bandit = DummyBandit(label_count)
                bandit.expected_reward = np.mean(truth)
                bandit.cached_contexts = np.zeros((len(advice), 1, 1))

                n_arms = advice.shape[-1]
                n_trials = advice.shape[0]
                n_experts = advice.shape[1]

                avg_agent = Average(bandit, GreedyPolicy(), n_experts)
                random_agent = Average(bandit, Policy(), n_experts)

                # experts doubled for inversion trick
                mab_agent = MAB(bandit, GreedyPolicy(), n_experts*2)  # Meta-MAB
                force_exp_agent = Exp4(bandit, Exp3Policy(), # EXP4-IX
                                       n_experts*2, gamma=.5*(2*np.log(n_experts*2)/(n_arms * n_trials)) ** (1 / 2))

                cover_agent = OnlineCover(bandit, Exp3Policy(eps=0.05*min(1/n_arms, 1/(np.sqrt(1*n_arms)))*n_arms),
                                          n_experts) # Meta-CMAB (ILTCB)

                falcon_agent = SafeFalcon(bandit, SCBPolicy( # Meta-CMAB (Falcon)
                    None), n_experts, n_trials=n_trials)
                mexpl3_agent = SupLinUCBVar( # Meta-CMAB (UCB)
                    bandit, GreedyPolicy(), n_experts)

                agents = (('random', random_agent), ('exp4', force_exp_agent), ('avg', avg_agent), ('mab', mab_agent),
                          ('cover', cover_agent), ('falcon', falcon_agent), ('mexpl3', mexpl3_agent))

                for alg, agent in agents:
                    agent.reset()
                    np.random.seed(target*RUNS+r)
                    for t in range(len(advice)):
                        advice_matrix = advice[instance_order][t]

                        meta_context = {'advice': advice_matrix}
                        expanded_meta_context = {'advice': np.vstack(
                            [advice_matrix, 1-advice_matrix])}
                        reward_vector = truth[instance_order][t]
                        avg_choice = np.argmax(np.mean(advice_matrix, axis=0))

                        if type(agent) in (MAB, Exp4,):  # invert trick
                            choice = agent.choose(expanded_meta_context)
                        else:
                            choice = agent.choose(meta_context)

                        reward = reward_vector[choice]
                        agent.observe(reward, choice)

                        regrets[alg].append(np.max(reward_vector)-reward)

                expert_regret = np.zeros((len(advice), n_experts))

                for t in range(len(advice)):
                    advice_matrix = advice[instance_order][t]
                    reward_vector = truth[instance_order][t]
                    for i in range(n_experts):
                        expert_choice = np.argmax(advice_matrix[i])
                        expert_regret[t, i] = np.max(
                            reward_vector)-reward_vector[expert_choice]

                cumsummed = np.cumsum(expert_regret, axis=0)
                order = np.argsort(cumsummed[-1])

                averaged_expert_regret += cumsummed[:, order]
        averaged_expert_regret /= (RUNS*len(targets))

        stdev = (averaged_expert_regret[-1]/len(advice)).std()
        average_expertise = 1-(averaged_expert_regret[-1]/len(advice)).mean()
        best = (averaged_expert_regret[-1]/len(advice))[0]

        cov = np.mean(
            np.cov(np.swapaxes(advice, 0, 1).reshape((n_experts, -1))))
        all_expert_scores.append((averaged_expert_regret[-1]/len(advice)))

        stdevs.append(stdev)
        expertises.append(average_expertise)
        bests.append(best)
        covs.append(cov)
        for k in regrets:
            arr_regrets = np.array(regrets[k]).reshape(
                (RUNS*(target+1), -1)).mean(axis=0)
            scores[k].append(np.cumsum(arr_regrets)[-1]/len(advice))

        all_expert_scores = np.array(all_expert_scores)
    results = []
    for label in scores:
        for expertise, stdev, cov, best, regret in zip(expertises, stdevs, covs, bests, scores[label]):
            results.append((label, 1-regret, 1-expertise, stdev, cov, 1-best))
    for i, expert_scores in enumerate(all_expert_scores.T):
        if i not in (0, num_experts-1):
            continue
        for expertise, stdev, cov, best, regret in zip(expertises, stdevs, covs, bests, expert_scores):
            if i == 0:
                label = "best expert"
            else:
                label = "worst expert"
            results.append((label, 1-regret, expertise, stdev, cov, 1-best)) # regret,  and best are all regret measures, we invert them to obtain reward measures instead

    res = pd.DataFrame(results, columns=[
                       "alg", "Mean reward", "Mean expertise", "Expertise std", "Mean Covariance", "Best expert"])
    res.to_csv(mode+"k_results.csv")
