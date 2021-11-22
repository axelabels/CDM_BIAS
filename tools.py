from scipy.spatial.distance import cdist
from scipy.special import expit
import math
import re
import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

from collections import namedtuple
Result = namedtuple('Result', ('prior', 'bias', 'std',
                               'A', 'N', 'tp', 'str', 'seed', 'scale'))
def greedy_choice(a,axis=None):
    max_values =np.amax(a,axis=axis,keepdims=True)
    return (a==max_values).astype(np.float)


def contains(s, l):
    for e in l:
        if e in s:
            return e
    return False


def prob_round(x):
    x = np.asarray(x)
    probs = x - np.floor(x)
    added = probs > np.random.uniform(size=np.shape(probs))
    return (np.floor(x)+added).astype(int)


def unit_vectors(vector):
    """ Returns the unit vector of the vector.  """
    if len(np.shape(vector)) == 1:
        return np.array([vector / np.linalg.norm(vector)])
    else:
        return vector / np.linalg.norm(vector, axis=-1)[..., np.newaxis]


def permutate(v, n):
    assert n!=1
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


def angle_between(v1):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v2 = (1, 0)

    v1_u = unit_vectors(v1)
    v2_u = unit_vectors(v2)[0]

    angles = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angles[v1_u[..., 1] < 0] = -angles[v1_u[..., 1] < 0]

    return angles


def random_circle(size=1):
    if type(size) is not tuple:
        try:
            size = tuple(size)
        except:
            size = (size, 2)
    d = size[-1]
    size = list(size)
    size[-1] = d+2
    size = tuple(size)

    points = np.random.normal(0, 1, size=(size))
    norm = np.sum(points*points, axis=-1)**0.5
    pts = points[..., :d]/norm[..., np.newaxis]
    return (pts+1)/2


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


def cpu_count():
    return 4


def weighted_random(w):
    return np.searchsorted(np.cumsum(w), np.random.rand(1))[0]


def aid(x):
    return x.__array_interface__['data'][0]


def nothing(a): runner(*a)


def nothing2(a, b, c, d, e, f, g): pass


def safe_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))


def SMInv(Ainv, u, v, alpha=1):
    u = u.reshape((len(u), 1))
    v = v.reshape((len(v), 1))
    return Ainv - np.dot(Ainv, np.dot(np.dot(u, v.T), Ainv)) / (1 + np.dot(v.T, np.dot(Ainv, u)))


def randargmax(b, **kw):
    """ a random tie-breaking argmax"""
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


sigmoid = expit


def get_coords(dims=2, complexity=3, precision=100, flatten=True):

    lin = np.linspace(0, complexity, int(
        (precision)**(2/dims)) + 1, endpoint=True)
    coords = np.array(np.meshgrid(*(lin for _ in range(dims)))).T / complexity
    if flatten:
        coords = coords.reshape((-1, dims))
    return coords


def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r


def cor_dist(a, b):
    return 1 - (corr2(a, b)+1)/2
