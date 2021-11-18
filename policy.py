
import numpy as np

from tools import softmax



class Policy(object):
    def __init__(self, b=1):
        self.b = b
        self.key = 'value'

    def __str__(self):
        return 'generic policy'

    def probabilities(self, agent, contexts):
        a = agent.value_estimates(contexts)
        self.pi = softmax(a*self.b)
        return self.pi
        

    def choose(self, agent, contexts, greedy=False):
        

        self.pi = self.probabilities(agent, contexts)
        
        if greedy:
            check = np.where(self.pi == self.pi.max())[0]
            self.pi[:] = 0
            self.pi[check] = 1 / len(check)

        np.testing.assert_allclose(np.sum(self.pi),1)
        action = np.searchsorted(np.cumsum(self.pi), np.random.rand(1))[0]

        return action
        


class RandomPolicy(Policy):

    def __init__(self):
        self.key = 'value'

    def __str__(self):
        return 'random'

    def probabilities(self, agent, contexts):
        self.pi = np.ones(agent.bandit.k)/agent.bandit.k
        return self.pi


class EpsilonGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.key = 'value'

    def __str__(self):
        return 'eps'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        self.pi = np.empty(agent.bandit.k)
        self.pi.fill(self.epsilon / agent.bandit.k)
        v = agent.value_estimates(contexts)
        check = np.where(v == v.max())[0]
    
        self.pi[check] += (1 - self.epsilon) / len(check)        
        
        return self.pi


class GreedyPolicy(EpsilonGreedyPolicy):

    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return 'greedy'


class ProbabilityGreedyPolicy(Policy):
    def __init__(self, epsilon=0):
        self.epsilon = epsilon
        self.datas = []
        self.key = 'probability'

    def __str__(self):
        return 'PGP'.format(self.epsilon)

    def probabilities(self, agent, contexts):
        
        self.pi = agent.probabilities(contexts)
        check = np.where(np.array(self.pi) == self.pi.max())[0]

        self.pi[:]=self.epsilon / agent.bandit.k
        self.pi[check] += (1 - self.epsilon) / len(check)

        return self.pi


class UCBPolicy(Policy):

    def __init__(self, beta=100):
        self.beta = beta

    def __str__(self):
        return 'GPUCB' 

    def probabilities(self, agent, contexts):
        self.pi = softmax(10000*agent.ucb_values(contexts))
        return self.pi

class Exp3Policy(Policy):
    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'

    def __str__(self):
        return 'E3P'

    def probabilities(self, agent, contexts):
        self.pi = agent.probabilities(contexts)
        self.pi = self.pi * (1 - self.eps) + self.eps / len(self.pi) 
        return self.pi

class SCBPolicy(Policy):

    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'

    def __str__(self):
        return 'SCB'

    def probabilities(self, agent, contexts):
        assert self.eps==0
        if self.eps == 0:
            return agent.probabilities(contexts)
        else:
            p = agent.probabilities(contexts)
            active_arms = (agent.bandit.get_active_arms(contexts))
            k = len(p) if not agent.bandit.dynamic_arms else np.sum(active_arms)
            ps = p * (1 - self.eps) + self.eps / k
            ps[~active_arms] = 0
            ps /= np.sum(ps)
            return ps
