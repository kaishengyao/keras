from __future__ import absolute_import
import numpy as np
import time
import sys
import theano
import theano.tensor as T


def cosine_similarity(k, memory):
    k_unit = k / T.sqrt(T.sum(k ** 2))

    memory_lengths = T.sqrt(T.sum(memory ** 2, axis=1)).dimshuffle((0, 'x'))
    memory_unit = memory / memory_lengths

    return T.sum(k_unit * memory_unit, axis=1)


def softmax_of_vector(vec):
    return T.nnet.softmax(vec)

