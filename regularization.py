__author__="Joachim Ott"
# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T
from round_op import GradPreserveRoundTensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def dual_copy_rounding(W,integer_bits=0,fractional_bits=1):
    """
    Rounding as described in as in "Robustness of spiking Deep Belief Networks to noise and reduced bit precision
    of neuro-inspired hardware platforms"
    by Stromatidis et al. See http://dx.doi.org/10.3389/fnins.2015.00222
    :param W: Weights
    :param integer_bits: number of bits to represent the integer part
    :param fractional_bits: number of bits to represent the fractional part
    :return:quantized weights
    """
    #print "Dual copy rounding!"
    power = T.cast(2.**fractional_bits, theano.config.floatX) # float !
    max_val = T.cast((2.**(fractional_bits+integer_bits))-1, theano.config.floatX)
    value = W*power
    value = GradPreserveRoundTensor(value) # rounding
    value = T.clip(value, -max_val, max_val) # saturation arithmetic
    Wb = value/power
    return Wb


def binarize_weights(W,H,srng=None,deterministic=False):
    """
    Copied from BinaryNet by Matthieu Courbariaux, https://github.com/MatthieuCourbariaux/BinaryNet
    :param W:
    :param H:
    :param srng:
    :param deterministic:
    :return: quantized weights
    """
    if srng is None:
        rng = np.random.RandomState(666)
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(rng.randint(999999))

    # [-1,1] -> [0,1]
    Wb=T.clip(((W / H)+1.)/2.,0,1)

    # Deterministic BinaryConnect (round to nearest)
    if deterministic:
        # print("det")
        Wb = T.cast(GradPreserveRoundTensor(Wb), theano.config.floatX)

    # Stochastic BinaryConnect
    else:
        # print("stoch")
        Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

    Wb = T.cast(T.switch(Wb, H, -H), theano.config.floatX)

    return Wb


def ternarize_weights(W,W0,deterministic=False,srng=None):
    """
    Changed copy of the code from TernaryConnect by Zhouhan Lin, Matthieu Courbariaux,
    https://github.com/hantek/BinaryConnect/tree/ternary
    :param W: Weights
    :param W0: W0=0.5
    :param deterministic: deterministic rounding
    :param srng: random number generator
    :return: quantized weights
    """
    Wb=None
    #print 'Current W0: ',W0
    if srng is None:
        rng = np.random.RandomState(666)
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(rng.randint(999999))
    if deterministic:
        #print 'Deterministic Ternarization!'

        larger_than_neg_0_5 = T.gt(W, -W0/2.)
        larger_than_pos_0_5 = T.gt(W, W0/2.)
        W_val = larger_than_neg_0_5 * 1 + larger_than_pos_0_5 * 1 - 1
        Wb = W_val * W0

    else:
        #print 'Stochastic Ternarization!'
        w_sign = T.gt(W, 0) * 2 - 1
        p = T.clip(T.abs_(W / (W0)), 0, 1)
        Wb = W0 * w_sign * T.cast(srng.binomial(n=1, p=p, size=T.shape(W)), theano.config.floatX)

    return Wb


def quantize_weights(W,srng=None,bitlimit=None,deterministic=False):
    """
    Exponential quantization
    :param W: Weights
    :param srng: random number generator
    :param bitlimit: limit values to be in power of 2 range, e.g. for values in 2^-22 to 2^9 set it to [-22, 9]
    :param deterministic: deterministic rounding
    :return: quantized weights
    """
    bitlimit=[-22, 9] #hardcoded for experiments
    if srng is None:
        rng = np.random.RandomState(666)
        srng = theano.sandbox.rng_mrg.MRG_RandomStreams(rng.randint(999999))

    if bitlimit:
        index_low = T.clip(
            T.switch(W > 0., T.floor(T.log2(W)), T.floor(T.log2(-W))),
            bitlimit[0], bitlimit[1])
    else:
        index_low = T.switch(
            W > 0., T.floor(T.log2(W)), T.floor(T.log2(-W)))
    sign = T.switch(W > 0., 1., -1.)
    p_up = sign * W / 2 ** (index_low) - 1  # percentage of upper index.
    if deterministic:
        index_deterministic = index_low + T.switch(p_up > 0.5, 1, 0)
        quantized_W = sign * 2 ** index_deterministic
    else:
        index_random = index_low + srng.binomial(
            n=1, p=p_up, size=T.shape(W), dtype=theano.config.floatX)
        quantized_W = sign * 2 ** index_random
    return quantized_W



