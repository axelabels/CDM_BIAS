
import numpy as np

from tools import greedy_choice, softmax



class Policy(object):

    def probabilities(self, agent, contexts):
        self.pi = np.ones(agent.bandit.k)/agent.bandit.k
        return self.pi
        

    def choose(self, agent, contexts, greedy=False):
        

        self.pi = self.probabilities(agent, contexts)
        
        if greedy:
            self.pi = greedy_choice(self.pi)
        np.testing.assert_allclose(np.sum(self.pi),1,atol=1e-5,err_msg=str(agent)+" "+str(np.sum(self.pi))+" "+str(self.pi))
        action = np.searchsorted(np.cumsum(self.pi), np.random.rand(1))[0]

        return action
        


class EpsilonGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon


    def probabilities(self, agent, contexts):
        v = agent.value_estimates(contexts)
        self.pi = greedy_choice(v)       
        self.pi *= (1-self.epsilon)
        self.pi += self.epsilon/agent.bandit.k
        
        return self.pi


class GreedyPolicy(EpsilonGreedyPolicy):

    def __init__(self):
        super().__init__(0)




class Exp3Policy(Policy):
    def __init__(self, eps=0):
        self.eps = eps
        self.key = 'probability'


    def probabilities(self, agent, contexts):
        self.pi = agent.probabilities(contexts)
        
        self.pi = self.pi * (1 - self.eps) + self.eps / len(self.pi) 
        return self.pi

class SCBPolicy(Policy):

    def __init__(self, gamma=0):
        self.gamma = gamma
        self.key = 'probability'


    def probabilities(self, agent, contexts):

        values = agent.value_estimates(contexts)
        best_arm = np.argmax(values)
        self.pi = np.zeros_like(values)
        self.pi[:] = 1 / \
            (agent.bandit.k+self.gamma*(values[best_arm]-values))
        self.pi[best_arm] += (1-(np.sum(self.pi)))

        return self.pi

