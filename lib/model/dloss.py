import mindspore
import mindspore.nn as nn
from mindspore import ops


# loss function
def KL(alpha, c):
    beta = ops.ones((1, c))
    S_alpha = ops.sum(alpha, dim=1, keepdim=True)
    S_beta = ops.sum(beta, dim=1, keepdim=True)
    lnB = ops.mvlgamma(S_alpha, 1) - ops.sum(ops.mvlgamma(alpha, 1), dim=1, keepdim=True)
    lnB_uni = ops.sum(ops.mvlgamma(beta, 1), dim=1, keepdim=True) - ops.mvlgamma(S_beta, 1)
    dg0 = ops.digamma(S_alpha)
    dg1 = ops.digamma(alpha)
    kl = ops.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(alpha, p, c=2):
    S = ops.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = ops.sum(label * (ops.digamma(S) - ops.digamma(alpha)), dim=1, keepdim=True)
    alp = E * (1 - label) + 1
    B = KL(alp, c)
    return ops.mean(A + B)