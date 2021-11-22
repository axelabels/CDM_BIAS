from functools import lru_cache
from math import ceil
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
# from utils import BasicStatistics, RemovePolicy, Propaganda, setup_city_layout
import copy
from tools import *
from tools import normalize, get_coords, sigmoid, mse, mae, corr2
from perlin import PerlinNoiseFactory
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


FIXED_COORDS = {}


class MetaBandit():
    def __init__(self, dims, k):
        self.dims = dims
        self.k = k


class PerlinBandit():
    def __init__(self, n_arms=1, complexity=2, precision=20, reset=True, invert=False,
                 reduce_to=0, family='perlin', bernoulli=True):
        self.expected_reward=.5
        self.dims = 2
        self.k = n_arms
        self.cache_id = -1
        self.bernoulli = bernoulli
        self.int_columns = True
        self.precision = precision
        self.invert = invert
        self.reduce_to = reduce_to
        self.family = family

        self.cached_contexts = None
        self.cached_values = None
        self.cached_rewards = None
        self.last_contexts = None

        self.complexity = complexity
        self._value_landscapes = None
        self.perlin_functions = [Perlin(complexity=complexity,
                                        precision=precision,
                                        bernoulli=self.bernoulli) for _ in range(n_arms)]

        if reset:
            self.reset()


    def update_value_landscapes(self):
        grid_data = get_coords(self.dims, self.complexity, self.precision)
        self._value_landscapes = np.array(
            [s.get(grid_data, override=True) for s in self.perlin_functions])

    @property
    def value_landscapes(self):
        if self._value_landscapes is None:
            self.update_value_landscapes()

        return self._value_landscapes

    def max_distance(self):
        return np.mean([s.max_distance() for s in self.perlin_functions])

    def distance(self, other):
        return np.mean([Perlin.DISTANCE_METRIC(o.value_landscape, s.value_landscape) for o, s in zip(other.perlin_functions, self.perlin_functions)])

    @property
    def grid_data(self):
        return self.value_landscapes

    def summary(self):
        return ", ".join([s.summary() for s in self.perlin_functions])

    def convert(self, family):
        [s.convert(family) for s in self.perlin_functions]

    def from_bandit(self, desired_distance, enforce_distance=True, family=None):
        prior_bandit = PerlinBandit(self.k, self.complexity, self.precision, reset=False,
                                    invert=self.invert,
                                    reduce_to=self.reduce_to, family=family, )

        assert 0 <= desired_distance <= 1, desired_distance
        for i, sub_bandit in enumerate(self.perlin_functions):
            prior_bandit.perlin_functions[i] = sub_bandit.from_bandit(
                desired_distance=desired_distance, enforce_distance=enforce_distance)
        prior_bandit.update_value_landscapes()
        return prior_bandit

    def reset(self):
        for s in self.perlin_functions:
            s.reset()

        self.cached_contexts = None
        self.update_value_landscapes()

    def get(self, contexts, override=True, force_repeat=False):

        assert np.shape(contexts)[1:] == (self.dims,), str(
            (np.shape(contexts), " and ", (self.dims,)))

        return np.array([s.get(contexts[:], override) for i, s in enumerate(self.perlin_functions)]).T

    def observe_contexts(self, center=.5, spread=1, k=None, cache_index=None):
        if cache_index is not None:
            self.contexts = self.cached_contexts[cache_index]
            self.action_values = self.cached_values[cache_index]
        else:
            if k is None:
                k = self.k
            self.contexts = np.random.uniform(
                center - spread / 2, center + spread / 2, size=(self.dims))

            self.contexts = self.contexts % 1
            self.action_values = self.get(self.contexts[None, :])[0]
        self.optimal_value = np.max(self.action_values)

        return self.contexts

    def cache_contexts(self, t, cache_id, activity_p=1):

        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.random.uniform(
                0, 1, size=(t,  self.dims))

            self.cached_values = self.get(self.cached_contexts)
            assert np.shape(self.cached_values) == (
                t, self.k), (np.shape(self.cached_values), (t, self.k))
            self.cached_rewards = self.sample(self.cached_values)

            assert np.shape(self.cached_rewards) == (t, self.k)
            self.cache_id = cache_id

        return self.cached_contexts

    def pull(self, action, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index, action], action == np.argmax(self.cached_values[cache_index])
        if self.bernoulli:
            return np.random.uniform() < self.action_values[action], action == np.argmax(self.action_values)
        else:
            return self.action_values[action], action == np.argmax(self.action_values)

    def sample(self, values=None, cache_index=None):
        if cache_index is not None:
            return self.cached_rewards[cache_index]

        if values is None:
            values = self.action_values
        if self.bernoulli:
            return np.random.uniform(size=np.shape(values)) < values
        else:
            return values


class Perlin():
    DISTANCE_METRIC = mse

    def __init__(self, n_arms=1, complexity=2, precision=None, dimensions=2, reset=True, invert=False,  bernoulli=True):
        self.precision = precision if precision is not None else int(
            np.round((4000)**(1/dimensions)))
        self.grid_data = None
        self._value_landscape = None
        self.family = 'perlin'

        self.invert = invert

        self.dims = dimensions
        self.complexity = complexity
        self.power = 1

        self.k = n_arms

        self._value_landscape = None

        self.cache_id = -1

        self.bernoulli = bernoulli
        if reset:
            self.reset()

    def reset(self, gradients=None, noise=0):

        self.cache_id = -1
        if gradients is not None:
            new_gradients = {}
            for k, v in gradients.items():
                angle = np.arctan2(*v)
                sign = np.random.choice((-1, 1))
                r = angle+sign*noise**.8*np.pi+wrapcauchy.rvs(0.99)

                new_gradients[k] = np.array([np.sin(r), np.cos(r)])
            self.fac = PerlinNoiseFactory(
                self.dims,  gradients=new_gradients)
        else:
            self.fac = PerlinNoiseFactory(self.dims)
        random_points = np.random.uniform(
            size=(2, self.dims)) * self.complexity
        self.fac(random_points)
        self._value_landscape = None

        self.min_val = -1
        self.max_val = 1
        self.mean_value = 0.5

    def max_distance(self):
        return Perlin.DISTANCE_METRIC(1 - self.value_landscape,
                                      self.value_landscape)

    def distance(self, other):
        return Perlin.DISTANCE_METRIC(other.value_landscape,
                                      self.value_landscape)

    @property
    def value_landscape(self):
        if self._value_landscape is None:
            grid_data = get_coords(self.dims, self.complexity, self.precision)
            self._value_landscape = self.get(grid_data, override=True)
        return self._value_landscape

    def get(self, contexts, override=None):
        scaled_contexts = np.asarray(contexts) * self.complexity
        values = self.fac(scaled_contexts)
        values = ((values - self.min_val) /
                  (self.max_val - self.min_val))

        return values

    def from_bandit(self, desired_distance, enforce_distance=True):

        best_bandit = None
        best_score = INF

        for attempts in range(MAX_ATTEMPTS_BANDIT_DISTANCE):
            prior_bandit = Perlin(
                self.k, self.complexity, self.precision, dimensions=self.dims, reset=False)

            gradients = copy.deepcopy(self.fac.gradient)

            noise = desired_distance

            if enforce_distance and desired_distance > .5:
                for k in gradients:
                    gradients[k] = -gradients[k]

                noise = 1 - desired_distance
            prior_bandit.reset(gradients, noise=noise)

            max_distance = self.max_distance()
            score = np.abs(self.distance(prior_bandit) /
                           max_distance - desired_distance)

            if score < best_score:
                best_bandit, best_score = prior_bandit, score

            if not enforce_distance or score <= BANDIT_DISTANCE_EPSILON:
                break
        else:
            prior_bandit = best_bandit
        return prior_bandit


DATA_ROOT = "./"
dic_int_columns = {}
dic_og_columns = {}
dic_mapped_columns = {}
dic_excl_columns = {}


@lru_cache(None)
def get_permutation_list(family):
    return np.load(DATA_ROOT+f"datasets/{family}/saved_permutations.npy", allow_pickle=True)


@lru_cache(None)
def get_df_data(family):
    if family == 'mushroom':

        int_columns = False
        MUSHROOMS_CSV = DATA_ROOT+"datasets/mushroom/data/mushroom_csv.csv"
        MUSHROOM_DF = pd.read_csv(MUSHROOMS_CSV)
        og_columns = MUSHROOM_DF.columns
        excl_columns = 'class'
        MUSHROOM_DF = pd.get_dummies(MUSHROOM_DF).rename(
            columns={'class_e': 'reward0'})
        mapped_columns = MUSHROOM_DF.columns
        
        k = 2
        df = MUSHROOM_DF
        bernoulli = True
        expected_reward=1/k
    elif family == "shuttle":
        int_columns = True
        SHUTTLE_CSV = DATA_ROOT+"datasets/shuttle/shuttle.tst"
        SHUTTLE_DF = (pd.read_csv(SHUTTLE_CSV, sep=' ', header=None))
        SHUTTLE_DF = pd.concat(
            (SHUTTLE_DF.drop(9, axis=1), pd.get_dummies(SHUTTLE_DF[9])), axis=1)

        k = 7
        df = SHUTTLE_DF
        bernoulli = True
        expected_reward=1/k

    elif family == "covtype":

        COVTYPE_CSV = DATA_ROOT+"datasets/covtype/covtype.data"
        int_columns = False

        FEATURE_NAMES = ["Elevation",
                         "Aspect",
                         "Slope",
                         "Horizontal_Distance_To_Hydrology",
                         "Vertical_Distance_To_Hydrology",
                         "Horizontal_Distance_To_Roadways",
                         "Hillshade_9am",
                         "Hillshade_Noon",
                         "Hillshade_3pm",
                         "Horizontal_Distance_To_Fire_Points"]
        FEATURE_NAMES += [f"Wilderness_Area_{i}" for i in range(4)]
        FEATURE_NAMES += [f"Soil_Type_{i}" for i in range(40)]
        FEATURE_NAMES += ["Cover_Type"]

        COVTYPE_DF = (pd.read_csv(COVTYPE_CSV, sep=',',
                                  header=None, names=FEATURE_NAMES))

        og_columns = np.array(["Elevation",
                               "Aspect",
                               "Slope",
                               "Horizontal_Distance_To_Hydrology",
                               "Vertical_Distance_To_Hydrology",
                               "Horizontal_Distance_To_Roadways",
                               "Hillshade_9am",
                               "Hillshade_Noon",
                               "Hillshade_3pm",
                               "Horizontal_Distance_To_Fire_Points", "Wilderness_Area", "Soil_Type", "Cover_Type"])
        excl_columns = ('Cover_Type',)
        COVTYPE_DF = pd.concat((COVTYPE_DF.drop(
            'Cover_Type', axis=1), pd.get_dummies(COVTYPE_DF['Cover_Type'])), axis=1)
        mapped_columns = COVTYPE_DF.columns

        k = 7
        df = COVTYPE_DF
        bernoulli = True
        expected_reward=1/k

    elif family == "adult":

        int_columns = False

        ADULT_CSV = DATA_ROOT+"datasets/adult/adult.data"
        ADULT_DF = (pd.read_csv(ADULT_CSV, sep=',', header=None, names=('age', 'workclass', 'fnlwgt', 'education',
                                                                        'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                                                                        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income')))
        ADULT_DF = ADULT_DF[ADULT_DF['occupation'] != ' ?']
        og_columns = ADULT_DF.columns
        excl_columns = ('occupation', 'fnlwgt', 'education-num',)
        order = ADULT_DF.columns
        order = [
            i for i in order if i not in excl_columns]+['occupation']
        ADULT_DF = pd.get_dummies(ADULT_DF[order])
        mapped_columns = ADULT_DF.columns

        k = 14
        df = ADULT_DF
        bernoulli = True
        expected_reward=1/k
    elif family == "census":

        int_columns = False

        CENSUS_CSV = DATA_ROOT+"datasets/census/USCensus1990.data.txt"
        CENSUS_DF = (pd.read_csv(CENSUS_CSV, sep=','))
        CENSUS_DF = CENSUS_DF.astype(str)
        order = CENSUS_DF.columns[:]
        og_columns = order
        excl_columns = ('dOccup', 'caseid',)
        order = [i for i in order if i not in excl_columns]+['dOccup']
        CENSUS_DF = pd.get_dummies(CENSUS_DF[order])
        mapped_columns = CENSUS_DF.columns

        k = 9
        df = CENSUS_DF
        bernoulli = True
        expected_reward=1/k
    elif family == "jester":
        int_columns = True

        JESTER_NPY = DATA_ROOT+"datasets/jester/jester_data_40jokes_19181users.npy"
        JESTER_DF = pd.DataFrame(np.load(JESTER_NPY)/20+.5)

        k = 8
        df = JESTER_DF
        expected_reward=0.5
    elif family == "stocks":
        int_columns = True
        STOCKS_CSV = DATA_ROOT+"datasets/stocks/raw_stock_contexts"
        STOCKS_DF = (pd.read_csv(STOCKS_CSV, sep=' '))

        rewards = STOCKS_DF.dot(STOCK_WEIGHTS.T)
        rews = ((rewards-np.min(rewards.values.flatten())) /
                np.ptp(rewards.values.flatten())).values
        STOCKS_DF = pd.DataFrame(pd.np.column_stack([STOCKS_DF, rews]))

        k = 8
        df = STOCKS_DF
        bernoulli = False
        expected_reward=0.5
    else:
        raise Exception(f"Dataset family '{family}' unknown")

    return int_columns, None if int_columns else og_columns, None if int_columns else excl_columns, None if int_columns else mapped_columns, k, df, bernoulli,expected_reward


class DatasetBandit(PerlinBandit):
    metric = mse

    def __init__(self, precision=200, reset=True, invert=False,  reduce_to=0, family='mushroom',  verbose=False, bernoulli=False):

        self.bernoulli = bernoulli

        
        self.int_columns, self.og_columns, self.excl_columns, self.mapped_columns, self.k, self.df, self.bernoulli,self.expected_reward = get_df_data(
            family)

        X_step = ceil(len(self.df)/DATASET_SUBSET_SIZE)
        self.X = self.df.values[::X_step, :-self.k]
        self.X = normalize(self.X, axis=0)

        self.Y = self.df.values[::X_step, -self.k:]
        if verbose:
            sampled_instance = np.random.choice(len(self.Y))
            print(
                f"dataset {family} of shape {self.X.shape} {self.df.shape} with {self.k} arms,example arm: {self.Y[sampled_instance]} mean {np.round(np.mean(self.Y,axis=0),5)}")
        self.step_size = ceil(len(self.X)/DISTANCE_SUBSET_SIZE)
        self.oracle = KNeighborsRegressor(
            n_neighbors=1,  leaf_size=500).fit(self.X, self.Y)
        self.reduced_oracle = KNeighborsRegressor(n_neighbors=1,  leaf_size=500).fit(
            self.X[::self.step_size], self.Y[::self.step_size])
        self.dims = self.X.shape[-1]
        self.cache_id = -1
        self.precision = precision
        self.invert = invert
        self.reduce_to = reduce_to
        self.cached_all_indices = {}
        self.family = family

        self.cached_contexts = None
        self.cached_values = None
        self.cached_rewards = None
        self._value_landscapes = None
        self.permutation_list = None
        self.permutations = np.arange(self.dims)
        if reset:
            self.reset()

    @property
    def value_landscapes(self):
        if self._value_landscapes is None:
            self._value_landscapes = self.get(
                self.X[::self.step_size], reduced=True)
        return self._value_landscapes

    def max_distance(self):
        return 2/self.k

    def distance(self, other):
        if self == other:
            return 0
        return DatasetBandit.metric(self.value_landscapes, other.value_landscapes)

    @property
    def grid_data(self):
        return self.value_landscapes


    def from_bandit(self, desired_distance, enforce_distance=True,  verbose=False, precomputed_permutations=True):

        if desired_distance == 0:
            return self
        best_bandit = None
        best_goal = INF
        offset = 0
        for _ in range(MAX_ATTEMPTS_BANDIT_DISTANCE):

            desired_distance_copy = desired_distance
            prior_bandit = DatasetBandit( self.precision, reset=False,
                                         invert=self.invert,  reduce_to=self.reduce_to, family=self.family,  verbose=verbose)

            prior_bandit.reset()
            prior_bandit.permutations = np.copy(self.permutations)

            if precomputed_permutations and (self.permutations == np.arange(self.dims)).all():

                if self.permutation_list is None:
                    self.permutation_list = get_permutation_list(self.family)
                allowable = [p for p in self.permutation_list if np.abs(
                    p[0]-desired_distance) < BANDIT_DISTANCE_EPSILON]
                prior_bandit.permutations = allowable[np.random.choice(
                    len(allowable))][1]
            else:
                perms = 1

                if desired_distance_copy > .5:
                    prior_bandit.invert = not self.invert
                    desired_distance_copy = 1-desired_distance_copy
                desired_distance_copy = desired_distance_copy*2
                possible_perms = prior_bandit.dims
                step_size = (possible_perms)/min(10, possible_perms)
                all_perms = prob_round(
                    [i for i in np.arange(0, possible_perms+1, step_size)])
                all_perms = list(filter(lambda i: i != 1, all_perms))

                ls = np.array(all_perms)/all_perms[-1]
                for lo, hi in zip(ls, ls[1:]):
                    if lo <= desired_distance_copy and hi >= desired_distance_copy:
                        break
                rng = hi-lo
                dist_lo = desired_distance_copy - lo
                dist_hi = hi-desired_distance_copy

                p_lo = dist_lo/rng
                p_hi = dist_hi/rng
                assert not np.isnan(p_lo), (hi, lo, dist_lo, rng,
                                            p_lo, all_perms, prior_bandit.family)
                assert not np.isnan(p_hi), (hi, lo, dist_hi, rng,
                                            p_hi, all_perms, prior_bandit.family)

                perms = int(np.clip(prob_round(np.tanh((.5-np.abs(desired_distance-.5))*4)
                                               * all_perms[-1])+offset, all_perms[0], all_perms[-1]))

                perms = perms if perms != 1 else np.random.choice([0, 2])
                prior_bandit.permutations = permutate(
                    prior_bandit.permutations, perms)
                attempt = 0
                while np.abs(np.sum(prior_bandit.permutations != self.permutations)-perms) > step_size/2:
                    np.random.shuffle(prior_bandit.permutations)
                    attempt += 1
                    assert attempt < 100, (perms, all_perms,
                                           prior_bandit.family)

            max_distance = self.max_distance()
            scaled_distance = min(1, self.distance(
                prior_bandit) / max_distance)
            goal = np.abs(scaled_distance - desired_distance)
            if (desired_distance <= 0.5 and scaled_distance < desired_distance) or (desired_distance > 0.5 and scaled_distance > desired_distance):
                if offset < 0:
                    offset += prior_bandit.dims//6
                else:
                    offset += prior_bandit.dims//3
            else:
                if offset > 0:
                    offset -= prior_bandit.dims//6
                else:
                    offset -= prior_bandit.dims//3
            if goal < best_goal:
                best_bandit, best_goal = prior_bandit, goal

            if not enforce_distance or goal <= BANDIT_DISTANCE_EPSILON:  # or noise < 1e-9:

                break
        else:
            prior_bandit = best_bandit

        prior_bandit.int_columns = self.int_columns
        if not prior_bandit.int_columns:
            prior_bandit.og_columns = self.og_columns
            prior_bandit.excl_columns = self.excl_columns
            prior_bandit.mapped_columns = self.mapped_columns
        return prior_bandit

    def reset(self):
        self.cache_id = -1
        self._value_landscapes = None
        self.cached_contexts = None
        self.cached_all_indices = {}

    def get(self, contexts,  reduced=False):

        assert np.shape(contexts)[1:] == (
            self.dims,), f"context should be of shape {(-1,self.dims)}, but has shape {np.shape(contexts)}"

        if reduced:
            values = np.copy(self.reduced_oracle.predict(
                contexts[:,  self.permutations]))
        else:
            values = np.copy(self.oracle.predict(
                contexts[:,  self.permutations]))

        if self.invert:
            values = 1 - values
        if self.bernoulli:
            values = values/np.sum(values, axis=1)[:, np.newaxis]

        return values

    def observe_contexts(self, center=.5,  k=None, cache_index=None, steps=None, step=None):
        if cache_index is not None:
            self.contexts = self.cached_contexts[cache_index]
            self.action_values = self.cached_values[cache_index]
            self.optimal_value = np.max(self.action_values)
            return self.contexts

        if k is None:
            k = self.k

        self.contexts = np.zeros((self.dims,))
        if steps is not None:

            if steps > len(self.X):  # random experiences
                all_indices = np.random.choice(len(self.X), size=steps)
            else:  # closest experiences
                if tuple(center) not in self.cached_all_indices:

                    self.cached_all_indices[tuple(center)] = self.oracle.kneighbors(
                        [center], n_neighbors=steps, return_distance=False)[0]

                all_indices = self.cached_all_indices[tuple(center)]

            indices = all_indices[step]

        else:
            indices = np.random.choice(
                len(self.X), size=1, replace=k < len(self.X))
        self.contexts[ :] = self.X[indices]

        self.action_values = self.get(self.contexts[None,:])[0]
        self.optimal_value = np.max(self.action_values)

        return self.contexts

    def cache_contexts(self, t, cache_id):
        if self.cached_contexts is None or len(self.cached_contexts) != t:
            self.cached_contexts = np.zeros((t,  self.dims))
            indices = np.random.choice(
                len(self.X), size=t, replace=t < len(self.X))
            self.cached_contexts[:] = self.X[indices][:, :]

            self.cached_values = self.get(self.cached_contexts[:])
            assert np.shape(self.cached_values) == (
                t, self.k), (np.shape(self.cached_values), "vs", (t, self.k))
            self.cached_rewards = self.sample(self.cached_values)

            assert np.shape(self.cached_rewards) == (t, self.k)
            self.cache_id = cache_id

        return self.cached_contexts
