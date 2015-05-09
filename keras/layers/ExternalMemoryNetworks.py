from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, alloc_zeros_matrix
from ..utils.functions import cosine_similarity, softmax_of_vector
from ..layers.core import Layer
from six.moves import range

'''
to-do: need to carry over external memory from previous sentences
'''
class RNNEM(Layer):
    '''
        RNN-EM uses external memory. It can be considered as a simplification of Neural Turning Machine.

        Eats inputs with shape:
        (nb_samples, max_sample_length (samples shorter than this are padded with zeros at the end), input_dim)

        and returns outputs with shape:
        if not return_sequences:
            (nb_samples, output_dim)
        if return_sequences:
            (nb_samples, max_sample_length, output_dim)

        References:
            Recurrent Neural Networks with External Memory for Language Understanding
                B. Peng and K. Yao, submitted to Interspeech 2015
    '''
    def __init__(self, input_dim, output_dim=128,
        hidden_dim = 100,
        memory_slots_number = 40,
        memory_dim = 100,
        init='glorot_uniform', inner_init='orthogonal',
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False):

        super(RNNEM,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_slots_number = memory_slots_number
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.truncate_gradient = truncate_gradient
        self.return_sequences = return_sequences

        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.input = T.tensor3()

        self.W_ih = self.init((self.input_dim, self.hidden_dim))
        self.W_c  = self.inner_init((self.memory_dim, self.hidden_dim))
        self.b_h = shared_zeros((self.hidden_dim))

        self.W_ho = self.init((self.hidden_dim, self.output_dim))
        self.b_o  = shared_zeros((self.output_dim))

        self.W_k = self.init((self.hidden_dim, self.memory_dim))
        self.b_k = shared_zeros(((self.memory_dim)))

        self.W_beta = self.inner_init((self.hidden_dim, 1))
        self.b_beta = shared_zeros((1))

        self.W_gain = self.init((self.hidden_dim, 1))
        self.b_gain = shared_zeros(1)

        self.W_v = self.init((self.hidden_dim, self.memory_dim))
        self.b_v = shared_zeros((self.memory_dim))

        self.W_erase = self.init((self.hidden_dim, self.memory_dim))
        self.b_erase = shared_zeros((self.memory_dim))

        self.params = [
            self.W_ih, self.b_h,
            self.W_ho, self.b_o,
            self.W_k, self.b_k,
            self.W_beta, self.b_beta,
            self.W_gain, self.b_gain,
            self.W_v, self.b_v,
            self.W_erase, self.b_erase
        ]

        if weights is not None:
            self.set_weights(weights)

    def _retrieve_content(self,
        memory_tm1, weight_tm1):
        c_t = T.dot(weight_tm1, memory_tm1)
        return c_t

    def _update_memory_read_weights(self,
        h_t,
        memory_tm1, weight_tm1,
        b_k, W_k, b_beta, W_beta, b_gain, W_gain):
        k_t = T.dot(h_t, W_k) + b_k
        beta_t = T.dot(h_t, W_beta) + b_beta
        beta_t = T.nnet.softplus(beta_t)

        g_t = self.inner_activation(b_gain + T.dot(h_t, W_gain))

        # compute weight
        weight_t = softmax_of_vector(beta_t * cosine_similarity(k_t, memory_tm1))
        weight_t = g_t * weight_t + (1-g_t) * weight_tm1
        return weight_t

    def _update_memory(self,
        h_t, weight_t,
        memory_tm1,
        b_v, W_v, b_erase, W_erase):
        v_t = T.dot(h_t, W_v) + b_v
        e_t = self.inner_activation(b_erase + T.dot(h_t , W_erase))
        f_t = 1 - weight_t * e_t
        u_t = weight_t
        memory_t = memory_tm1 * f_t + u_t * v_t
        return memory_t

    def _step(self,
        xi_t,
        y_tm1, memory_tm1, weight_tm1,
        W_c, b_k, W_k, b_beta, W_beta, b_gain, W_gain, W_ho, b_o, b_v, W_v, b_erase, W_erase):

        c_t = self._retrieve_content(memory_tm1, weight_tm1)
        h_t = self.inner_activation(xi_t + T.dot(c_t, W_c))
        weight_t = self._update_memory_read_weights(h_t, memory_tm1, weight_tm1, b_k, W_k, b_beta, W_beta, b_gain, W_gain)
        memory_t = self._update_memory(h_t, weight_t, memory_tm1, b_v, W_v, b_erase, W_erase)
        y_t = self.activation(T.dot(h_t, W_ho) + b_o)

        return y_t, memory_t, weight_t

    def output(self, train):
        X = self.get_input(train)
        X = X.dimshuffle((1,0,2))

        xi = T.dot(X, self.W_ih) + self.b_h

        [outputs, memories, weights], updates = theano.scan(
            self._step,
            sequences=xi,
            outputs_info=[
                alloc_zeros_matrix(X.shape[1], self.output_dim),
                alloc_zeros_matrix(self.memory_slots_number, self.memory_dim), # need to carry over from previous sentences
                alloc_zeros_matrix(X.shape[1], self.memory_slots_number)
            ],
            non_sequences=[self.W_c, self.b_k, self.W_k, self.b_beta, self.W_beta, self.b_gain, self.W_gain, self.W_ho, self.b_o, self.b_v, self.W_v, self.b_erase, self.W_erase],
            truncate_gradient=self.truncate_gradient
        )

        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "mem_dim":self.memory_dim,
            "memory_slots_number":self.memory_slots_number,
            "hidden_dim":self.hidden_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

