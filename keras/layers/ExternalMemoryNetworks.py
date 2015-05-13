from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros, alloc_zeros_matrix
from ..utils.functions import cosine_similarity, softmax_of_vector, diag
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
    def __init__(self, input_dim, output_dim,
        mem_dim=2, mem_size = 3,
        init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        inner_activation='tanh',
        truncate_gradient=-1, return_sequences=False):
        super(RNNEM,self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.mem_dim = mem_dim
        self.mem_size = mem_size
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W = self.init((self.input_dim, self.mem_dim))
        self.b = shared_zeros((self.mem_dim))
        self.W_c = self.init((self.mem_dim, self.mem_dim))
        self.W_v = self.init((self.mem_dim, self.mem_dim))
        self.b_v = shared_zeros((self.mem_dim))
        self.W_gain = self.init((self.mem_dim, 1))
        self.b_gain = shared_zeros((1))

        self.W_k = self.init((self.mem_dim, self.mem_dim))
        self.b_k = shared_zeros((self.mem_dim))
        self.W_beta = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (self.mem_dim)).astype(theano.config.floatX))
        self.b_beta = theano.shared(0.0)

        self.W_erase = self.init((self.mem_dim, self.mem_size))
        self.b_erase = shared_zeros((self.mem_size))

        self.W_o = self.init((self.mem_dim, self.output_dim))

        self.params = [self.W, self.b, self.W_c, self.W_v, self.b_v, self.W_gain, self.b_gain, self.W_k, self.b_k, self.W_beta, self.b_beta, self.W_erase, self.b_erase, self.W_o]

        if weights is not None:
            self.set_weights(weights)

    def _retrieve_content(self,
        memory_tm1, weight_tm1):

        weight_tm1 = theano.tensor.flatten(weight_tm1, 1)
        c_t = T.dot(weight_tm1, memory_tm1)

        return c_t

    def _update_memory(self,
        h_t, weight_t,
        memory_tm1):
        v_t = T.dot(h_t, self.W_v) + self.b_v
        e_t = self.inner_activation(self.b_erase + T.dot(h_t , self.W_erase)).T

        f_t = 1 - weight_t * e_t
        u_t = weight_t
        F_t = diag(f_t, self.mem_size)
        U_t = diag(u_t, self.mem_size)

        memory_t = T.dot(F_t, memory_tm1) + T.dot(U_t, T.tile(v_t, [self.mem_size,1]))
        return memory_t

    def _update_memory_read_weights(self,
        h_t, memory_tm1, weight_tm1):

        k_t = T.dot(h_t, self.W_k) + self.b_k  # 1 x mem_dim
        beta_t = T.dot(h_t, self.W_beta) + self.b_beta  # 1
        beta_t = T.nnet.softplus(beta_t)  # log ( 1 + exp(x))

        g_t = self.inner_activation(self.b_gain + T.dot(h_t, self.W_gain))  # 1
        weight_t = softmax_of_vector(beta_t * cosine_similarity(k_t, memory_tm1))
        weight_t = T.tile(g_t, [self.mem_size, 1]) * weight_t + T.tile((1-g_t), [self.mem_size,1]) * weight_tm1

        return weight_t

    def _step(self, x_t, memory_tm1, weight_tm1):
        '''
            Variable names follow the conventions from:
            http://deeplearning.net/software/theano/library/scan.html
        '''

        c_t = self._retrieve_content(memory_tm1, weight_tm1)  # 1 x mem_dim
        h_t = self.inner_activation(x_t + T.dot(c_t, self.W_c))    # 1 x mem_dim
        w_t = self._update_memory_read_weights(h_t, memory_tm1, weight_tm1)
        m_t = self._update_memory(h_t, w_t, memory_tm1)
        o_t = T.dot(h_t, self.W_o)

        '''
        m_t_print = theano.printing.Print('m_t')(m_t)
        w_t_print = theano.printing.Print('w_t')(w_t)
        o_t_print = theano.printing.Print('o_t')(o_t)
        '''

        return [m_t, w_t, o_t]

    def _create_memory(self):
        self.memory_init = 0.2 * np.random.uniform(-1.0, 1.0, (self.mem_size, self.mem_dim)).astype(theano.config.floatX)
        self.params.extend([self.memory_init])

    def _build_model(self):
        self._create_memory()
        memory_init = self.memory_init
        weight_init = T.nnet.softmax(np.random.rand(self.mem_size, 1))
        return self._step, [memory_init, weight_init, None]

    def output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1,0,2))

        X.tag.test_value = np.random.rand(2, 2, self.input_dim).astype(theano.config.floatX)

        x = T.dot(X, self.W) + self.b

        self.recurrence, self.outputs_info = self._build_model()

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        [memory, weights, outputs], updates = theano.scan(
            self.recurrence, # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=x, # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info = self.outputs_info,
            truncate_gradient=self.truncate_gradient
        )

        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "mem_dim":self.mem_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}


class RNNEM2bk(Layer):
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
    def __init__(self, input_dim, output_dim,
        hidden_dim = 10,
        mem_dim=20, mem_width = 30,
        init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        inner_activation='tanh',
        truncate_gradient=-1, return_sequences=False):
        super(RNNEM2bk,self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.mem_width = mem_width
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W_input_hidden = self.init((self.input_dim, self.hidden_dim))
        self.W_read_hidden = self.init((self.mem_width, self.hidden_dim))
        self.b_hidden_0 = shared_zeros((self.hidden_dim))
        self.W_hidden_output = self.init((self.hidden_dim, self.output_dim))
        self.b_output_0 = shared_zeros((self.output_dim))

        self.params = [self.W_input_hidden, self.W_read_hidden, self.W_hidden_output, self.b_hidden_0, self.b_output_0]

        self.scan_recurrence, self.outputs_info = self.build_main_model()

        if weights is not None:
            self.set_weights(weights)

    def controller(self, input_t, read_t):
        """
        add input gate so that the input_t is modulated before feeding to external memory
        :param input_t: current external input
        :param read_t: past memory read
        :return:
        """

        # hidden layer activity
        pre_layer = T.tanh(
                T.dot(input_t, self.W_input_hidden) +
                T.dot(read_t, self.W_read_hidden) +
                self.b_hidden_0
        )

        # RNN output
        output_t = T.nnet.softmax(T.dot(pre_layer, self.W_hidden_output) + self.b_output_0)

        return output_t, pre_layer

    def build_head(self, input_size, mem_width):
        self.W_k = self.init((input_size, mem_width))
        self.b_k = shared_zeros((mem_width))

        self.W_beta = shared_zeros((input_size))
        self.b_beta = shared_zeros((1))

        self.W_g = shared_zeros((input_size))
        self.b_g = shared_zeros((1))

        self.W_erase = self.init((input_size, mem_width))
        self.b_erase = shared_zeros((mem_width))

        self.W_v = self.init((input_size, mem_width))
        self.b_v = shared_zeros((mem_width))

        def forward(input_t):
            key_t = T.dot(input_t, self.W_k) + self.b_k

            beta_t = T.dot(input_t, self.W_beta) + self.b_beta

            beta_t = T.nnet.softplus(beta_t)

            erase_t = T.nnet.sigmoid(
                T.dot(input_t, self.W_erase) + self.b_erase)

            g_t = T.nnet.sigmoid(
                T.dot(input_t, self.W_g) + self.b_g)

            # no sigmoid operation is on input_t, so that this model is closely related to simple RNN
            add_t = T.dot(input_t, self.W_v) + self.b_v

            return key_t, beta_t, g_t, erase_t, add_t

        self.params.extend([
                self.W_k, self.b_k,
                self.W_g, self.b_g,
                self.W_beta, self.b_beta,
                self.W_erase, self.b_erase,
                self.W_v, self.b_v
            ])

        return forward

    # build main model
    def build_main_model(self):

        self.memory_init = shared_zeros((self.mem_dim, self.mem_width))

        memory_init = self.memory_init
        weight_init = T.nnet.softmax(np.random.rand(self.mem_dim, 1))

        controller_size = self.output_dim

        head = self.build_head(controller_size, self.mem_width)

        def recurrence(input_t, memory_tm1, weight_tm1):
            weight_tm1 = theano.tensor.flatten(weight_tm1, 1)

            read_tm1 = T.dot(weight_tm1, memory_tm1)

            # output_controller is the output from RNN
            # output_controller_before is the hidden layer activity
            output_controller, output_controller_before = self.controller(input_t, read_tm1)

            weight_temp, memory_temp = weight_tm1, memory_tm1

            key, beta, g, erase_t, add_t = head(output_controller_before)

            # focusing on content

            weight_t = softmax_of_vector(beta * cosine_similarity(key,memory_temp))

            weight_t = g * weight_t + (1 - g) * weight_temp

            weight_t = T.reshape(weight_t, (weight_t.shape[1], 1))

            memory_erased = memory_temp * (1 - weight_t * erase_t)
            memory_t = memory_erased + weight_t * add_t

            return memory_t, weight_t, output_controller

        self.params.extend([self.memory_init])

        return recurrence, [memory_init, weight_init, None]

    def output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1,0,2))

#        X.tag.test_value = np.random.rand(2, 1, self.input_dim).astype(theano.config.floatX)

        x = X

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        [memory, weight, outputs], updates = theano.scan(
            self.scan_recurrence, # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=x, # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=self.outputs_info,
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "mem_dim":self.mem_dim,
            "mem_width":self.mem_width,
            "hidden_dim":self.hidden_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

    '''
    vectorize memory
    '''
class RNNEMvec(Layer):
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
    def __init__(self, input_dim, output_dim,
        mem_dim=1, mem_width = 30,
        hidden_dim = 10,
        init='glorot_uniform', inner_init='orthogonal', activation='sigmoid', weights=None,
        inner_activation='tanh',
        truncate_gradient=-1, return_sequences=False):
        super(RNNEMvec,self).__init__()
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        self.mem_width = mem_width
        self.output_dim = output_dim
        self.truncate_gradient = truncate_gradient
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.return_sequences = return_sequences
        self.input = T.tensor3()

        self.W_input_hidden = self.init((self.input_dim, self.hidden_dim))
        self.W_read_hidden = self.init((self.mem_width, self.hidden_dim))
        self.b_hidden_0 = shared_zeros((self.hidden_dim))
        self.W_hidden_output = self.init((self.hidden_dim, self.output_dim))
        self.b_output_0 = shared_zeros((self.output_dim))

        self.params = [self.W_input_hidden, self.W_read_hidden, self.W_hidden_output, self.b_hidden_0, self.b_output_0]

        self.scan_recurrence, self.outputs_info = self.build_main_model()

        if weights is not None:
            self.set_weights(weights)

    def controller(self, input_t, read_t):
        """
        add input gate so that the input_t is modulated before feeding to external memory
        :param input_t: current external input
        :param read_t: past memory read
        :return:
        """

        # hidden layer activity
        pre_layer = T.tanh(
                T.dot(input_t, self.W_input_hidden) +
                T.dot(read_t, self.W_read_hidden) +
                self.b_hidden_0
        )

        # RNN output
        output_t = T.nnet.softmax(T.dot(pre_layer, self.W_hidden_output) + self.b_output_0)

        return output_t, pre_layer

    def build_head(self, input_size, mem_width):
        self.W_k = self.init((input_size, mem_width))
        self.b_k = shared_zeros((mem_width))

        self.W_beta = shared_zeros((input_size))
        self.b_beta = shared_zeros((1))

        self.W_g = shared_zeros((input_size))
        self.b_g = shared_zeros((1))

        self.W_erase = self.init((input_size, mem_width))
        self.b_erase = shared_zeros((mem_width))

        self.W_v = self.init((input_size, mem_width))
        self.b_v = shared_zeros((mem_width))

        def forward(input_t):
            key_t = T.dot(input_t, self.W_k) + self.b_k

            beta_t = T.dot(input_t, self.W_beta) + self.b_beta

            beta_t = T.nnet.softplus(beta_t)

            erase_t = T.nnet.sigmoid(
                T.dot(input_t, self.W_erase) + self.b_erase)

            g_t = T.nnet.sigmoid(
                T.dot(input_t, self.W_g) + self.b_g)

            # no sigmoid operation is on input_t, so that this model is closely related to simple RNN
            add_t = T.dot(input_t, self.W_v) + self.b_v

            return key_t, beta_t, g_t, erase_t, add_t

        self.params.extend([
                self.W_k, self.b_k,
                self.W_g, self.b_g,
                self.W_beta, self.b_beta,
                self.W_erase, self.b_erase,
                self.W_v, self.b_v
            ])

        return forward

    # build main model
    def build_main_model(self):

        self.memory_init = shared_zeros((self.mem_dim, self.mem_width))

        memory_init = self.memory_init
        weight_init = T.nnet.softmax(np.random.rand(self.mem_dim, 1))

        controller_size = self.output_dim

        head = self.build_head(controller_size, self.mem_width)

        def recurrence(input_t, memory_tm1, weight_tm1):
#           m_tm1 = T.reshape(memory_tm1, [self.mem_dim, self.mem_width])
            m_tm1 = memory_tm1
            weight_tm1 = theano.tensor.flatten(weight_tm1, 1)

            read_tm1 = T.dot(weight_tm1, m_tm1)

            # output_controller is the output from RNN
            # output_controller_before is the hidden layer activity
            output_controller, output_controller_before = self.controller(input_t, read_tm1)

            weight_temp, memory_temp = weight_tm1, m_tm1

            key, beta, g, erase_t, add_t = head(output_controller_before)

            # focusing on content

            weight_t = softmax_of_vector(beta * cosine_similarity(key,memory_temp))

            weight_t = g * weight_t + (1 - g) * weight_temp

            weight_t = T.reshape(weight_t, (weight_t.shape[1], 1))

            memory_erased = memory_temp * (1 - weight_t * erase_t)
            memory_t = memory_erased + weight_t * add_t

#            m_t = memory_t.dimshuffle((0, 'x'))
            m_t = memory_t
            return m_t, weight_t, output_controller


        self.params.extend([self.memory_init])

        return recurrence, [memory_init, weight_init, None]

    def output(self, train):
        X = self.get_input(train) # shape: (nb_samples, time (padded with zeros at the end), input_dim)
        # new shape: (time, nb_samples, input_dim) -> because theano.scan iterates over main dimension
        X = X.dimshuffle((1,0,2))

#        X.tag.test_value = np.random.rand(2, 1, self.input_dim).astype(theano.config.floatX)

        x = X

        # scan = theano symbolic loop.
        # See: http://deeplearning.net/software/theano/library/scan.html
        # Iterate over the first dimension of the x array (=time).
        [memory, weight, outputs], updates = theano.scan(
            self.scan_recurrence, # this will be called with arguments (sequences[i], outputs[i-1], non_sequences[i])
            sequences=x, # tensors to iterate over, inputs to _step
            # initialization of the output. Input to _step with default tap=-1.
            outputs_info=self.outputs_info,
            truncate_gradient=self.truncate_gradient
        )
        if self.return_sequences:
            return outputs.dimshuffle((1,0,2))
        return outputs[-1]

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "mem_dim":self.mem_dim,
            "mem_width":self.mem_width,
            "hidden_dim":self.hidden_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__,
            "inner_init":self.inner_init.__name__,
            "activation":self.activation.__name__,
            "truncate_gradient":self.truncate_gradient,
            "return_sequences":self.return_sequences}

