import torch as th
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')

def identity(x):
    return x


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1) * (log_p2 - log_p1), 1)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int32) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.
    return one_hot


def index_to_one_hot_tensor(index: th.Tensor, dim):
    one_hot = th.zeros(len(index), dim).to(index.device)
    one_hot.scatter_(1, index.view(-1, 1), 1)
    return one_hot


def onehot_from_logits(logits: th.Tensor):
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs


def sample_gumbel(shape, eps=1e-20):
    u = th.rand(shape, device=DEVICE)
    return -th.log(-th.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1., hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = th.zeros_like(y, device=DEVICE).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def to_tensor_var(x, use_cuda=True, dtype="float"):
    FloatTensor = th.cuda.FloatTensor if use_cuda else th.FloatTensor
    LongTensor = th.cuda.LongTensor if use_cuda else th.LongTensor
    ByteTensor = th.cuda.ByteTensor if use_cuda else th.ByteTensor
    if dtype == "float":
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))
    elif dtype == "long":
        x = np.array(x, dtype=np.long).tolist()
        return Variable(LongTensor(x))
    elif dtype == "byte":
        x = np.array(x, dtype=np.byte).tolist()
        return Variable(ByteTensor(x))
    else:
        x = np.array(x, dtype=np.float64).tolist()
        return Variable(FloatTensor(x))


def to_tensor(x, device='cpu', dtype='float32'):
    x = np.array(x, dtype=dtype)
    return th.from_numpy(x).to(device)


def agg_double_list(l):
    # l: [ [...], [...], [...] ]
    # l_i: result of each step in the i-th episode
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s_mu, s_std


def mean_mean_list(l):
    m = [np.mean(np.array(l_i), 0) for l_i in l]
    m_mu = np.mean(np.array(m), 0)
    m_std = np.std(np.array(m), 0)
    return m_mu, m_std


def sum_mean_std(l):
    # l: [ [l_episode_1],..., [l_episode_E] ]
    # l_episode_e: [ [r_1^1,...,r_N^1],..., [r_1^T,..., r_N^T] ]
    s = [np.sum(np.array(l_i), 0) for l_i in l]
    s_mu = np.mean(np.array(s), 0)
    s_std = np.std(np.array(s), 0)
    return s, s_mu, s_std


# similarity between two cache
def sim(x: np.ndarray, y: np.ndarray):
    distance = np.linalg.norm(x-y, 2)
    return 1 / (1 + distance)
