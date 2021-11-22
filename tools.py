import math
import re
import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae


def greedy_choice(a, axis=None):
    max_values = np.amax(a, axis=axis, keepdims=True)
    choices = (a == max_values).astype(np.float)
    return choices / np.sum(choices,axis=axis,keepdims=True)

def prob_round(x):
    x = np.asarray(x)
    probs = x - np.floor(x)
    added = probs > np.random.uniform(size=np.shape(probs))
    return (np.floor(x)+added).astype(int)


def permutate(v, n):
    assert n != 1
    v = np.copy(v)
    if n == 0:
        return v
    orig = np.random.choice(len(v), size=n, replace=False)
    dest = np.copy(orig)
    dest[:-1] = orig[1:]
    dest[-1] = orig[0]
    straight = np.arange(len(v))
    straight[orig] = straight[dest]

    return v[straight]


def arr2str(array):
    """Converts an array into a one line string

    Arguments:
        array {array} -- Array to convert

    Returns:
        str -- The string representation
    """
    return re.sub(r'\s+', ' ',
                  str(array).replace('\r', '').replace('\n', '').replace(
                      "array", "").replace("\t", " "))


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))


def SMInv(Ainv, u, v, alpha=1):
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))
    return Ainv - np.dot(Ainv, np.dot(np.dot(u, v.T), Ainv)) / (1 + np.dot(v.T, np.dot(Ainv, u)))


def randargmax(b, **kw):
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def scale(a):
    return normalize(a, offset=np.min(a), scale=np.ptp(a))


def max_scale(a):
    return normalize(a, offset=np.min(a), scale=np.ptp(a) if np.ptp(a) != 0 else 1e-9)


def normalize(a, offset=None, scale=None, axis=None):
    a = np.asarray(a)
    if offset is None:
        offset = np.mean(a, axis=axis)
    if scale is None:
        scale = np.std(a, axis=axis)
    if type(scale) in (float, np.float64, np.uint8, np.int64):
        if scale == 0:
            print(f"forcing scale to 1, was {scale}")
            scale = 1
    else:
        try:
            scale[scale == 0] = 1
        except:
            print("a:", np.shape(a), "scale:", np.shape(scale), scale)
            raise
    return (a - offset) / scale


def rescale(a, mu, std):
    a = np.asarray(a)
    return a * std + mu


def get_coords(dims=2, complexity=3, precision=100, flatten=True):

    lin = np.linspace(0, complexity, int(
        (precision)**(2/dims)) + 1, endpoint=True)
    coords = np.array(np.meshgrid(*(lin for _ in range(dims)))).T / complexity
    if flatten:
        coords = coords.reshape((-1, dims))
    return coords
