import os
import pandas as pd
from itertools import chain, combinations

from collections import defaultdict
from tqdm import tqdm

from agent import *

from collections import defaultdict
from tqdm import tqdm
from policy import GreedyPolicy,Exp3Policy,SCBPolicy,Policy
from expert import Agent
class DummyBandit():
    def __init__(self,k):
        self.k=k
        self.cache_id=0
        self.dynamic_arms=False
        self.centers=[[0]]
        self.C=1
        self.expected_reward=0
    def get_active_arms(self,i):
        return np.ones(self.k)
    def get_inactive_arms(self,i):
        return []
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

for mode in ("dunchenne","snow","trec"):
    
    if mode == "dunchenne":
        df = pd.read_csv("dunchenne/mturklabels.txt",sep=" ",names=["task","workerID","label"])
    if mode=="trec":
        df = pd.read_csv("trec-rf10-crowd/trec-rf10-data.txt",sep="\t")
        df['task']=df.topicID.map(str)+'_'+df.docID.map(str)
    if mode=="snow":
        dfs=[]
        for filename in os.listdir("snow2008"):
            taskname = filename.split(".")[0]
            if taskname not in ("anger","disgust","fear","joy","sadness","surprise",):continue
            dfs.append(pd.read_csv(f"snow2008/{filename}",sep="\t"))
            dfs[-1]['task']=filename+"_"+dfs[-1]['orig_id'].map(str)
            if len(dfs)>1 and set(dfs[-1]['!amt_worker_ids'].unique())!=set(dfs[-2]['!amt_worker_ids'].unique()):
                dfs=dfs[:-1]
                break
        
        df = pd.concat(dfs)
        df.columns = ["topicID","workerID","orig_id","label","gold","task"]

    df = df.drop_duplicates()

    chosen_key = ('t',)
    counts=defaultdict(lambda: 0)
    tasks=[]
    for i,task in tqdm(list(enumerate(df.task.unique()))):
        workers = df[df.task==task].workerID.drop_duplicates().unique()
        if len(workers)>100:
            continue
        if len(workers)<5:continue
        for subset in powerset(workers):
            subset = tuple(sorted(subset))
            if len(subset)!=(5 if mode!="snow" else 6):continue
            if subset == chosen_key:
                if len(df[(df.task==task)&(df.workerID.isin(chosen_key))].drop_duplicates())==len(chosen_key):
                    tasks.append(task)
            counts[subset]+=1
    keys = sorted(counts,key= lambda k: (counts.get(k)>100,len(k),counts.get(k)))


        
    stdevs =[]
    covs =[]
    bests =[]
    means =[]
    expert_scores=[]
    scores=defaultdict(list)
    SIMULATIONS=100
    for chosen_key in tqdm(keys[-SIMULATIONS:][::-1]):
        
        tasks=[]
        sub_df = df[(df.workerID.isin(chosen_key))]
        for i,task in (list(enumerate(sub_df.task.unique()))):
            if len(sub_df[(sub_df.task==task)])==len(chosen_key):
                    tasks.append(task)
        
        answers = df[(df.task.isin(tasks))&(df.workerID.isin(chosen_key))].drop_duplicates()
        
        if mode=="snow":
            arms = set([a.split('_')[0] for a in answers.task.unique()])
            
            data = answers.sort_values(by=["orig_id","workerID","task"]).values
            data = data.reshape((-1,len(chosen_key),len(arms),6))
            gold = (data[:,0,:,4]/100).astype(float)
            data = (data[:,:,:,3]/100).astype(float)
            
            label_count =  data.shape[-1]
            
            
            values = data
        else:
            data = answers.sort_values(by=["task","workerID"]).values
            data = data.reshape((len(answers.task.unique()),len(chosen_key),-1))
            answer_idx = {"dunchenne":2,"trec":4}[mode] 
            label_count =  {"dunchenne":2,"trec":4}[mode]
            data = data[:,:,answer_idx]
            data = data[data.min(axis=1)>=0].astype(int)
            if len(data)<20:continue
            
            import numpy as np
            values = np.zeros(data.shape+(label_count,))

            for i in range(len(values)):
                values[i][np.arange(len(chosen_key)),data[i]]=1

        regrets=defaultdict(list)
        
        np.random.seed(0)
        
        averaged_expert_regret = np.zeros((len(values),len(chosen_key)-(mode!="snow")))

        targets = range(len(chosen_key)) if mode!="snow" else (0,)
        if mode=='dunchenne':
            targets=(0,)
        for target in (targets):
            if mode=="snow":
                advice=values
                truth = gold
            else:
                if mode=='dunchenne':
                    tmp_target = np.random.choice(len(chosen_key))
                    
                    advice=values[:,[i for i in range(len(chosen_key)) if i!=tmp_target]]
                    truth = values[:,tmp_target]
                else:
            
                    advice=values[:,[i for i in range(len(chosen_key)) if i!=target]]
                    truth = values[:,target]

            advice.shape,truth.shape
            num_experts = advice.shape[1]
            regret=0
            avg_regret=0
            RUNS=10
            for r in (range(RUNS)):
                Xs=[]
                ys=[]
                np.random.seed(target*RUNS+r)
                advice_order = np.random.choice(len(advice),size=len(advice),replace=False)
                bandit = DummyBandit(label_count)
                bandit.expected_reward = np.mean(truth)
                bandit.cached_contexts = np.zeros((len(advice),1,1))
                
            
                ALPHA=1
                n_arms = advice.shape[-1]
                n_trials = advice.shape[0]
                n_experts = advice.shape[1]# len(experts)
         
                avg_agent = Collective(bandit, GreedyPolicy(),n_experts)
                random_agent = Collective(bandit,Policy(),n_experts)
                
                #experts doubled for inversion trick
                mab_agent = MAB(bandit, GreedyPolicy(),n_experts*2)
                force_exp_agent = Exp4(bandit, Exp3Policy(),n_experts*2,gamma=.5*(2*np.log(n_experts*2)/(n_arms * n_trials)) ** (1 / 2))

                cover_agent = OnlineCover(bandit, Exp3Policy(eps = 0.05*min(1/n_arms,1/(np.sqrt(1*n_arms)))*n_arms),n_experts,alpha=ALPHA)
         
                falcon_agent =SafeFalcon(bandit, SCBPolicy(None),n_experts,n_trials=n_trials,alpha=ALPHA)
                mexpl3_agent = SupLinUCBVar(bandit, GreedyPolicy(),n_experts,alpha=ALPHA)

                agents = (('random',random_agent),('exp4',force_exp_agent),('avg',avg_agent),('mab',mab_agent),
                            ('cover',cover_agent),('falcon',falcon_agent), ('mexpl3',mexpl3_agent))

                for alg,agent in agents:
                    agent.reset()
                    np.random.seed(target*RUNS+r)
                    for t in range(len(advice)):
                        advice_matrix = advice[advice_order][t]
                        
                        meta_context = {'advice':advice_matrix}
                        expanded_meta_context = {'advice':np.vstack([advice_matrix,1-advice_matrix])}
                        reward_vector = truth[advice_order][t]
                        avg_choice = np.argmax(np.mean(advice_matrix,axis=0))

                        if type(agent) == OnlineCover:
                            agent.epsilon = agent.c_mu * \
                                min(1/n_arms, 1/(np.sqrt((t+1)*n_arms)))*n_arms
                        
                        if type(agent) in (MAB,Exp4,): #invert trick
                            choice = agent.choose(expanded_meta_context)
                        else:
                            choice = agent.choose(meta_context)

                        reward = reward_vector[choice]
                        agent.observe(reward,choice)
                            
                        regrets[alg].append(np.max(reward_vector)-reward)
                            
                expert_regret = np.zeros((len(advice),n_experts))
                
                for t in range(len(advice)):
                    advice_matrix = advice[advice_order][t]
                    reward_vector = truth[advice_order][t]
                    for i in range(n_experts):
                        expert_choice = np.argmax(advice_matrix[i])
                        expert_regret[t,i] = np.max(reward_vector)-reward_vector[expert_choice]
                        
                cumsummed = np.cumsum(expert_regret,axis=0)
                order = np.argsort(cumsummed[-1])

                averaged_expert_regret+= cumsummed[:,order]
        averaged_expert_regret/=(RUNS*len(targets))

        stdev = (averaged_expert_regret[-1]/len(advice)).std()        
        mean = (averaged_expert_regret[-1]/len(advice)).mean()     
        best = (averaged_expert_regret[-1]/len(advice))[0]
        
        cov = np.mean(np.cov(np.swapaxes(advice,0,1).reshape((n_experts,-1))))
        expert_scores.append( (averaged_expert_regret[-1]/len(advice)))
        
        stdevs.append(stdev)
        means.append(mean)
        bests.append(best)
        covs.append(cov)
        for k in regrets:
            arr_regrets=np.array(regrets[k]).reshape((RUNS*(target+1),-1)).mean(axis=0)
            scores[k].append(np.cumsum(arr_regrets)[-1]/len(advice))
            
        
        exp_sco = np.array(expert_scores)
    data_lines=[]
    for k in scores:
        for m,s,c,b,e in zip(means,stdevs,covs,bests,scores[k]):
            data_lines.append((k,1-e,1-m,s,c,1-b))
    for i,l in enumerate(exp_sco.T):
        if i not in (0,num_experts-1):continue
        for m,s,c,b,e in zip(means,stdevs,covs,bests,l):
            if i==0:
                label = "best expert"
            else:
                label = "worst expert"
            data_lines.append((label,1-e,1-m,s,c,1-b))

    res = pd.DataFrame(data_lines,columns=["alg","Mean reward","Mean expertise","Expertise std","Mean Covariance","Best expert"])
    res.to_csv(mode+"k_results.csv")
