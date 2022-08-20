from collections import Counter
from functools import lru_cache
from math import ceil
from matplotlib.pyplot import jet
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import copy
from tools import *

from scipy.stats import wrapcauchy

import pandas as pd

MAX_ATTEMPTS_BANDIT_DISTANCE = 100
BANDIT_DISTANCE_EPSILON = .05

DATASET_SUBSET_SIZE = 100000
DISTANCE_SUBSET_SIZE = 1000

INF = float("inf")


MUSHROOM_DF = None
SHUTTLE_DF = None
COVTYPE_DF = None
ADULT_DF = None
STOCKS_DF = None
JESTER_DF = None
CENSUS_DF = None

STOCK_WEIGHTS = np.array([[1.09688638e-02, -9.83605295e-02, -3.88597338e-03,
                           5.39227842e-02, -1.16686659e-02,  1.81166614e+00,
                           -1.72454761e-01,  1.51422601e-01,  1.74259336e-03,
                           -1.27679795e-01,  9.29732436e-02,  1.18714429e+00,
                           7.50326272e-03,  1.23811466e+00,  7.13277257e-04,
                           -1.53376698e-02, -1.00051674e-02, -4.73902004e-03,
                           -1.25100160e-02,  5.72865823e-02, -8.81874045e-04],
                          [6.48145483e-01,  1.55316623e-02, -5.14443628e-01,
                           1.20769516e+00,  3.53782844e-01,  9.10925270e-02,
                           -2.16996038e-01,  3.51408248e-03, -3.96718287e-01,
                           -3.27608653e-02, -2.44949619e-02,  3.79001476e-03,
                           1.00765021e+00, -7.94479343e-01, -9.18379731e-01,
                           -1.42537190e-01, -4.59942510e-01, -5.58384491e-04,
                           8.70134172e-02, -1.65091861e-01,  2.68475514e-03],
                          [-9.56863612e-01,  4.75314051e-02,  5.41547118e-01,
                           -9.13765469e-01, -3.59845189e-01,  3.94491301e-01,
                           3.86634813e-01, -4.42129599e-03, -3.67207778e-02,
                           -9.80587270e-03,  7.13447894e-02,  4.14989307e-01,
                           2.62763460e-01, -5.81185047e-02, -3.61361707e-02,
                           2.62017859e-01, -2.33170101e-03, -7.56533337e-02,
                           -2.69102506e-03,  7.97108351e-03,  2.06280071e-03],
                          [4.30070695e-04, -1.16380486e-01,  3.49083406e-01,
                           -2.95831884e-03, -5.49516429e-03, -1.70246063e-02,
                           -7.13939470e-03,  1.53904196e-02, -1.76599995e-01,
                           2.10276897e-02, -9.47098185e-01, -3.54975424e-01,
                           1.33185832e+00, -7.05464385e-01, -3.51471661e-01,
                           -1.18763579e-02,  1.38133923e+00,  1.00095182e+00,
                           -6.95295958e-03,  1.89716262e-02,  3.37650435e-03],
                          [-2.51130636e-02, -9.86046876e-02, -3.83825381e-02,
                           1.20710899e+00,  3.21042638e-03, -1.20244618e-01,
                           8.35749459e-03,  3.84475948e-02, -1.51882539e-02,
                           -3.52772376e-02,  5.13268125e-02, -1.29087319e-02,
                           -3.57808391e-01, -1.25763615e+00,  6.06335354e-02,
                           -2.60893212e-02,  4.51536532e-02, -2.12353420e-02,
                           -1.68112021e-03,  4.59584981e-01, -6.47623617e-02],
                          [3.44158518e-01,  3.58011403e-02,  4.89151100e-01,
                           2.70843399e-01,  1.00798072e-03,  2.41098895e-01,
                           8.89793807e-03, -2.83911287e-02,  5.04489951e-02,
                           -5.07119995e-02,  1.97634555e-01, -2.76854926e-01,
                           -3.31029548e-02,  1.80738061e-01, -1.10900337e-02,
                           3.01002133e-02,  9.13530546e-02, -1.69293832e-01,
                           5.30783867e-03, -1.11970232e+00,  4.09541765e-01],
                          [-2.74266167e-01,  3.53852581e-02, -3.42624305e-01,
                           -9.49722275e-01, -1.53429716e-03,  5.04026056e-03,
                           -1.42716691e-02,  6.11375460e-02, -3.15119073e-02,
                           2.22373254e-02, -4.75866680e-02,  1.72886445e-03,
                           -1.34501831e+00, -6.95498105e-01,  1.31423140e+00,
                           -1.78831546e-02,  3.98903197e-03,  3.51038926e-02,
                           2.87406220e-03,  1.22536024e-01, -2.66355360e-02],
                          [1.93359795e-02,  1.18222701e-01, -3.03092933e-01,
                           -1.03697525e+00, -7.40406376e-02, -2.22078429e+00,
                           -2.13072879e-02, -1.23835912e-02, -4.22090059e-03,
                           1.04867493e+00, -1.95800471e-02, -3.09694567e-03,
                           2.19818693e-01, -8.68593119e-02,  1.00959205e-03,
                           -1.07132009e+00,  2.50214090e-02, -2.26162807e-01,
                           7.71435160e-02, -2.12339988e-01,  2.94185012e-03]])



class MetaBandit():
    def __init__(self, dims, k):
        self.dims = dims
        self.k = k



UPPER = .8

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)
def shape_rewards(a,x):
    return a**x
class ArtificialBandit():
    def __init__(self, n_arms=1, smoothness=0,  bernoulli=True):
        self.k = n_arms
        self.smoothness = smoothness 
        self.bernoulli=bernoulli
        self.cached_contexts = None
        self.dims=1
        #np.repeat(np.arange(10)[None],10,axis=0)
    @property
    def expected_reward(self):
        return .5 if self.smoothness==2 else 1/self.k# np.mean(self.generate_random_values((1,self.k)))

    def reset(self):
        self.cached_contexts = None
    
        

    def generate_random_values(self,shape):
        if self.smoothness==2:
            values = np.repeat(np.arange(self.k)[None],shape[0],axis=0)
            values  = shuffle_along_axis(values ,axis=1)
            values = values/(self.k-1) * .8 + .1 
            return values
            # return np.random.beta(.5,.5,size=shape)#*(2*UPPER-1) + (1-UPPER)
            return np.random.uniform(size=shape)#*.8+.1#*(2*UPPER-1) + (1-UPPER)
        else:
            values = np.repeat(np.arange(self.k)[None],shape[0],axis=0)
            # print("cache",self.cached_values[0])
            # fghfg
                # self.cached_values /= np.sum(self.cached_values,axis=1)[:,None]
            # print("smooth",self.smoothness,self.cached_values[0])
            # dfgdfg
            values  = shuffle_along_axis(values ,axis=1)
            # assert (0<=self.cached_values).all() 
            # assert (1>=self.cached_values).all() 
            # print(np.max(self.cached_values),np.min(self.cached_values),np.mean(self.cached_values))
            values  = softmax(np.log(10)*values ,axis=1)
            return values
      

    

    def cache_contexts(self, t, cache_id):
        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.random.uniform(
                0, 1, size=(t,  self.dims))

            
            self.cached_values = self.generate_random_values((t,self.k)) # 
            if self.smoothness==10:
                # self.cached_values = np.repeat(np.arange(self.k)[None]/(self.k-1),t,axis=0)
                self.cached_values = np.repeat(np.arange(self.k)[None],t,axis=0)
            # print("cache",self.cached_values[0])
            # fghfg
                # self.cached_values /= np.sum(self.cached_values,axis=1)[:,None]
            # print("smooth",self.smoothness,self.cached_values[0])
            # dfgdfg
                self.cached_values  = shuffle_along_axis(self.cached_values ,axis=1)
                # assert (0<=self.cached_values).all() 
                # assert (1>=self.cached_values).all() 
                # print(np.max(self.cached_values),np.min(self.cached_values),np.mean(self.cached_values))
                self.cached_values  = softmax(np.log(10)*self.cached_values ,axis=1)
                # print(np.max(self.cached_values),np.min(self.cached_values),np.mean(self.cached_values))
                # self.cached_values = shape_rewards(self.smoothness,self.cached_values)
                assert (0<=self.cached_values).all() 
                assert (1>=self.cached_values).all() 

            assert np.shape(self.cached_values) == (
                t, self.k), (np.shape(self.cached_values), (t, self.k))
            self.cached_rewards = self.sample(self.cached_values)

            assert np.shape(self.cached_rewards) == (t, self.k)
            self.cache_id = cache_id

        return self.cached_contexts

    def observe_contexts(self, center=.5, spread=1, k=None, cache_index=None):
        
        self.contexts = self.cached_contexts[cache_index]
        self.action_values = self.cached_values[cache_index]
       
        self.optimal_value = np.max(self.action_values)

        return self.contexts
    def sample(self, values=None, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index]

        if values is None:
            values = self.action_values
        if self.bernoulli:
            
            return np.random.uniform(size=np.shape(values)) < values
        else:
            return values
