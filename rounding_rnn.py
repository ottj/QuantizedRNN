__author__="Joachim Ott"
# -*- coding: utf-8 -*-

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from regularization import *

class Gate(lasagne.layers.Gate):
    """
    This class extends the Lasagne Gate to support rounding of weights
    """
    def __init__(self,mode='normal',H=1.0,nonlinearity=lasagne.nonlinearities.sigmoid,bias_init=lasagne.init.Constant(-1.), **kwargs):
        if mode=='binary':
            if nonlinearity==lasagne.nonlinearities.tanh:
                nonlinearity=binary_tanh_unit
            elif nonlinearity==lasagne.nonlinearities.sigmoid:
                nonlinearity=binary_sigmoid_unit

        super(Gate, self).__init__(nonlinearity=nonlinearity,b=bias_init, **kwargs)



class GRULayer(lasagne.layers.GRULayer):
    """
    This class extends the Lasagne GRULayer to support rounding of weights
    """
    def __init__(self, incoming, num_units,
        stochastic = True, H='glorot',W_LR_scale="Glorot",mode='normal',integer_bits=0,fractional_bits=1,
                 random_seed=666,batch_norm=True,round_hid=True,bn_gamma=lasagne.init.Constant(0.1),mean_substraction_rounding=False,round_bias=True,round_input_weights=True,round_activations=False, **kwargs):

        self.H=H
        self.mode=mode
        self.srng = RandomStreams(random_seed)
        self.stochastic=stochastic
        self.integer_bits=integer_bits
        self.fractional_bits=fractional_bits
        self.batch_norm=batch_norm
        self.round_hid=round_hid
        self.mean_substraction_rounding=mean_substraction_rounding
        self.round_bias=round_bias
        self.round_input_weights=round_input_weights
        self.round_activations=round_activations

        print "Round HID: "+str(self.round_hid)


        if not(mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='normal' or mode=='quantize'):
            raise AssertionError("Unexpected value of 'mode'!", mode)

        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            print 'num_inputs: ',num_inputs
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        if mode=='binary' or mode=='ternary' or mode=='dual-copy' or mode=='quantize':
            super(GRULayer, self).__init__(incoming, num_units, **kwargs)
            # add the binary tag to weights
            if self.round_input_weights:
                self.params[self.W_in_to_updategate]=set(['binary'])
                self.params[self.W_in_to_resetgate]=set(['binary'])
                self.params[self.W_in_to_hidden_update]=set(['binary'])
            if self.round_hid:
                self.params[self.W_hid_to_updategate]=set(['binary'])
                self.params[self.W_hid_to_resetgate]=set(['binary'])
                self.params[self.W_hid_to_hidden_update]=set(['binary'])
            if self.round_bias:
                self.params[self.b_updategate] = set(['binary'])
                self.params[self.b_resetgate] = set(['binary'])
                self.params[self.b_hidden_update] = set(['binary'])


        else:
            super(GRULayer, self).__init__(incoming, num_units, **kwargs)

        self.high = np.float32(np.sqrt(6. / (num_inputs + num_units)))
        self.high_hid = np.float32(np.sqrt(6. / (num_units + num_units)))
        self.W0 = np.float32(self.high)
        self.W0_hid = np.float32(self.high_hid)


        input_shape = self.input_shapes[0]
        print 'Input shape: {0}'.format(input_shape)
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            print "BatchNorm activated!"
            self.bn =lasagne.layers.BatchNormLayer(input_shape,axes=(0,1),gamma=bn_gamma)  # create BN layer for correct input shape
            self.params.update(self.bn.params)  # make BN params your params
        else:
            print "BatchNorm deactivated!"


        #Create dummy variables
        self.W_hid_to_updategate_d = T.zeros(self.W_hid_to_updategate.shape, self.W_hid_to_updategate.dtype)
        self.W_hid_to_resetgate_d = T.zeros(self.W_hid_to_resetgate.shape, self.W_hid_to_updategate.dtype)
        self.W_hid_to_hidden_update_d = T.zeros(self.W_hid_to_hidden_update.shape, self.W_hid_to_hidden_update.dtype)
        self.W_in_to_updategate_d = T.zeros(self.W_in_to_updategate.shape, self.W_in_to_updategate.dtype)
        self.W_in_to_resetgate_d = T.zeros(self.W_in_to_resetgate.shape, self.W_in_to_resetgate.dtype)
        self.W_in_to_hidden_update_d = T.zeros(self.W_in_to_hidden_update.shape, self.W_in_to_hidden_update.dtype)
        self.b_updategate_d=T.zeros(self.b_updategate.shape, self.b_updategate.dtype)
        self.b_resetgate_d=T.zeros(self.b_resetgate.shape, self.b_resetgate.dtype)
        self.b_hidden_update_d=T.zeros(self.b_hidden_update.shape, self.b_hidden_update.dtype)



    def get_output_for(self, inputs,deterministic=False, **kwargs):
        if not self.stochastic and not deterministic:
            deterministic=True
        print "deterministic mode: ",deterministic
        def apply_regularization(weights,hid=False):
            current_w0 = self.W0
            if hid:
                current_w0 = self.W0_hid

            if self.mean_substraction_rounding:
                return weights
            elif self.mode == "ternary":

                return ternarize_weights(weights, W0=current_w0, deterministic=deterministic,
                                          srng=self.srng)
            elif self.mode == "binary":
                return binarize_weights(weights, 1., self.srng, deterministic=deterministic)
            elif self.mode == "dual-copy":
                return dual_copy_rounding(weights, self.integer_bits, self.fractional_bits)
            elif self.mode == "quantize":
                return quantize_weights(weights, srng=self.srng,deterministic=deterministic)
            else:
                return weights
        if self.round_input_weights:
            self.Wb_in_to_updategate = apply_regularization(self.W_in_to_updategate)
            self.Wb_in_to_resetgate = apply_regularization(self.W_in_to_resetgate)
            self.Wb_in_to_hidden_update = apply_regularization(self.W_in_to_hidden_update)

        if self.round_hid:
            self.Wb_hid_to_updategate = apply_regularization(self.W_hid_to_updategate,hid=True)
            self.Wb_hid_to_resetgate = apply_regularization(self.W_hid_to_resetgate,hid=True)
            self.Wb_hid_to_hidden_update = apply_regularization(self.W_hid_to_hidden_update,hid=True)
        if self.round_bias:
            self.bb_updategate= apply_regularization(self.b_updategate)
            self.bb_resetgate= apply_regularization(self.b_resetgate)
            self.bb_hidden_update=apply_regularization(self.b_hidden_update)


        #Backup high precision values

        Wr_in_to_updategate = self.W_in_to_updategate
        Wr_in_to_resetgate = self.W_in_to_resetgate
        Wr_in_to_hidden_update = self.W_in_to_hidden_update


        if self.round_hid:
            Wr_hid_to_updategate = self.W_hid_to_updategate
            Wr_hid_to_resetgate = self.W_hid_to_resetgate
            Wr_hid_to_hidden_update = self.W_hid_to_hidden_update
        if self.round_bias:
            self.br_updategate = self.b_updategate
            self.br_resetgate = self.b_resetgate
            self.br_hidden_update = self.b_hidden_update

        #Overwrite weights with binarized weights
        if self.round_input_weights:
            self.W_in_to_updategate = self.Wb_in_to_updategate
            self.W_in_to_resetgate = self.Wb_in_to_resetgate
            self.W_in_to_hidden_update = self.Wb_in_to_hidden_update

        if self.round_hid:
            self.W_hid_to_updategate = self.Wb_hid_to_updategate
            self.W_hid_to_resetgate = self.Wb_hid_to_resetgate
            self.W_hid_to_hidden_update = self.Wb_hid_to_hidden_update
        if self.round_bias:
            self.b_updategate = self.bb_updategate
            self.b_resetgate = self.bb_resetgate
            self.b_hidden_update = self.bb_hidden_update

        # Retrieve the layer input
        input = inputs[0]

        #Apply BN
        #https://github.com/Lasagne/Lasagne/issues/577
        if self.batch_norm:
            input = self.bn.get_output_for(input,deterministic=deterministic, **kwargs)
            if len(inputs) > 1:
                new_inputs=[input,inputs[1]]
            else:
                new_inputs=[input]
        else:
            new_inputs=inputs

        inputs=new_inputs

        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape



        # Stack input weight matrices into a (num_inputs, 3*num_units)
        #matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate+self.W_in_to_resetgate_d, self.W_in_to_updategate+self.W_in_to_updategate_d,
             self.W_in_to_hidden_update+self.W_in_to_hidden_update_d], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate+self.W_hid_to_resetgate_d, self.W_hid_to_updategate+self.W_hid_to_updategate_d,
             self.W_hid_to_hidden_update+self.W_hid_to_hidden_update_d], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate+self.b_resetgate_d, self.b_updategate+self.b_updategate_d,
             self.b_hidden_update+self.b_hidden_update_d], axis=0)

        if self.precompute_input:

            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w(x, n):
            return x[:, n * self.num_units:(n + 1) * self.num_units]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous, *args):

            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input = T.dot(hid_previous, W_hid_stacked)

            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input = theano.gradient.grad_clip(
                    hid_input, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Reset and update gates
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)

            # Compute W_{xc}x_t + r_t \odot (W_{hc} h_{t - 1})
            hidden_update_in = slice_w(input_n, 2)
            hidden_update_hid = slice_w(hid_input, 2)

            hidden_update = hidden_update_in + resetgate * hidden_update_hid
            if self.grad_clipping:
                hidden_update = theano.gradient.grad_clip(
                    hidden_update, -self.grad_clipping, self.grad_clipping)
            hidden_update = self.nonlinearity_hid(hidden_update)


            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous + updategate * hidden_update

            return hid

        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            hid = T.switch(mask_n, hid, hid_previous)

            return hid


        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if not isinstance(self.hid_init, lasagne.layers.Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]


        #Copy back high precision values
        if self.round_input_weights:
            self.W_in_to_updategate = Wr_in_to_updategate
            self.W_in_to_resetgate = Wr_in_to_resetgate
            self.W_in_to_hidden_update = Wr_in_to_hidden_update

        if self.round_hid:
            self.W_hid_to_updategate = Wr_hid_to_updategate
            self.W_hid_to_resetgate = Wr_hid_to_resetgate
            self.W_hid_to_hidden_update = Wr_hid_to_hidden_update
        if self.round_bias:
            self.b_updategate = self.br_updategate
            self.b_resetgate = self.br_resetgate
            self.b_hidden_update = self.br_hidden_update

        return hid_out

# This function computes the gradient of the binary weights
def compute_rnn_grads(loss,network):

    layers = lasagne.layers.get_all_layers(network)
    grads = []

    for layer in layers:

        params = layer.get_params(binary=True)
        if params:
            for param in params:
                print(param.name)
                if param.name=='W_in_to_updategate':
                    grads.append(theano.grad(loss, wrt=layer.W_in_to_updategate_d))
                elif param.name=='W_in_to_resetgate':
                    grads.append(theano.grad(loss, wrt=layer.W_in_to_resetgate_d))
                elif param.name=='W_in_to_hidden_update':
                    grads.append(theano.grad(loss, wrt=layer.W_in_to_hidden_update_d))
                elif param.name=='W_hid_to_updategate':
                    grads.append(theano.grad(loss, wrt=layer.W_hid_to_updategate_d))
                elif param.name=='W_hid_to_resetgate':
                    grads.append(theano.grad(loss, wrt=layer.W_hid_to_resetgate_d))
                elif param.name=='W_hid_to_hidden_update':
                    grads.append(theano.grad(loss, wrt=layer.W_hid_to_hidden_update_d))
                elif param.name=='b_updategate':
                    grads.append(theano.grad(loss, wrt=layer.b_updategate_d))
                elif param.name == 'b_resetgate':
                    grads.append(theano.grad(loss, wrt=layer.b_resetgate_d))
                elif param.name == 'b_hidden_update':
                    grads.append(theano.grad(loss, wrt=layer.b_hidden_update_d))

    return grads


#Copied from BinaryNet by Matthieu Courbariaux
def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh_unit(x):
    return 2.*GradPreserveRoundTensor(hard_sigmoid(x))-1.

def binary_sigmoid_unit(x):
    return GradPreserveRoundTensor(hard_sigmoid(x))