"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, action_shape,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0):

        self.input_width = input_width
        self.input_height = input_height
        self.action_shape = action_shape
        # The final output will be D x G
        # This may be wasteful if granularity is not the same across all actions,
        # but this lets us use numpy style broadcasting
        self.degrees_of_freedom = len(action_shape)
        self.max_granularity = max(action_shape)

        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng

        lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.l_out = self.build_network(network_type, input_width, input_height,
                                        action_shape, num_frames, batch_size)
        if self.freeze_interval > 0:
            self.next_l_out = self.build_network(network_type, input_width,
                                                 input_height, action_shape,
                                                 num_frames, batch_size)
            self.reset_q_hat()

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        # Normally, actions would have shape (batch_size, 1)
        # Now, they need to have shape (batch_size, D), so this is now an imatrix
        actions = T.imatrix('actions')
        terminals = T.icol('terminals')

        self.states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.next_states_shared = theano.shared(
            np.zeros((batch_size, num_frames, input_height, input_width),
                     dtype=theano.config.floatX))

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, self.degrees_of_freedom), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        # q_vals before: a (batch_size, output_dim) matrix, where [sample][i]
        # is q_value for ith action for that sample
        # what we would like: a (batch_size, D, G) matrix, where entry
        # [sample][i][a_i] is the value for choosing a_i for the ith component

        q_vals = lasagne.layers.get_output(self.l_out, states / input_scale)

        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out,
                                                    next_states / input_scale)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out,
                                                    next_states / input_scale)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        # building up the best next action from factored representation
        # TODO check this axis
        # shape before this is (batch_size, D, G)
        per_action_maxes = T.max(next_q_vals, axis=2)
        # now shape is (batch_size, D)
        # TODO is this the right axis?
        overall_q_max = T.sum(per_action_maxes, axis=1, keepdims=True)
        # now shape is (batch_size, 1)

        target = (rewards +
                  (T.ones_like(terminals) - terminals) *
                  self.discount * overall_q_max)
        # q_vals shape: (batch_size, D, G)
        # actions shape: (batch_size, D)
        # Construct the Q values for the specified actions
        # Recall numpy indexing convention: zips given iterables
        # TODO UGH DO THIS RIGHT
        # Can't think about numpy indexing properly right now
        action_values = [
            q_vals[T.arange(batch_size), i, actions[..., i]]
            for i in xrange(self.degrees_of_freedom)
        ]
        action_values = T.sum(action_values, axis=0)

        # TODO check if this still needs to be reshaped
        diff = target - action_values.reshape((-1, 1))

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out)  
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss, q_vals], updates=updates,
                                      givens=givens)
        self._q_vals = theano.function([], q_vals,
                                       givens={states: self.states_shared})

    def build_network(self, network_type, input_width, input_height,
                      output_shape, num_frames, batch_size):
        # EVERYTHING EXCEPT NIPS DNN IS BROKEN
        if network_type == "nature_cuda":
            return self.build_nature_network(input_width, input_height,
                                             output_shape, num_frames, batch_size)
        if network_type == "nature_dnn":
            return self.build_nature_network_dnn(input_width, input_height,
                                                 output_shape, num_frames,
                                                 batch_size)
        elif network_type == "nips_cuda":
            return self.build_nips_network(input_width, input_height,
                                           output_shape, num_frames, batch_size)
        elif network_type == "nips_dnn":
            return self.build_nips_network_dnn(input_width, input_height,
                                               output_shape, num_frames,
                                               batch_size)
        elif network_type == "linear":
            return self.build_linear_network(input_width, input_height,
                                             output_shape, num_frames, batch_size)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))



    def train(self, states, actions, rewards, next_states, terminals):
        """
        Train one batch.

        Arguments:

        states - b x f x h x w numpy array, where b is batch size,
                 f is num frames, h is height and w is width.
        actions - b x d numpy array of integers
        rewards - b x 1 numpy array
        next_states - b x f x h x w numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """

        self.states_shared.set_value(states)
        self.next_states_shared.set_value(next_states)
        self.actions_shared.set_value(actions)
        self.rewards_shared.set_value(rewards)
        self.terminals_shared.set_value(terminals)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
        loss, _ = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        states = np.zeros((self.batch_size, self.num_frames, self.input_height,
                           self.input_width), dtype=theano.config.floatX)
        states[0, ...] = state
        self.states_shared.set_value(states)
        # Returns a D x G array
        return self._q_vals()[0]

    def choose_action(self, state, epsilon):
        if self.rng.rand() < epsilon:
            return tuple(self.rng.randint(0, a_dim) for a_dim in self.action_shape)
        q_vals = self.q_vals(state)
        # Got a D x G array of values
        # Get the D array of best actions
        return np.argmax(q_vals, axis=1)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    def build_nature_network(self, input_width, input_height, output_shape,
                             num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import cuda_convnet

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(), # Defaults to Glorot
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv3 = cuda_convnet.Conv2DCCLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out


    def build_nature_network_dnn(self, input_width, input_height, output_shape,
                                 num_frames, batch_size):
        """
        Build a large network consistent with the DeepMind Nature paper.
        """
        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=32,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=64,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_conv3 = dnn.Conv2DDNNLayer(
            l_conv2,
            num_filters=64,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1)
        )

        return l_out



    def build_nips_network(self, input_width, input_height, output_shape,
                           num_frames, batch_size):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        from lasagne.layers import cuda_convnet
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_conv1 = cuda_convnet.Conv2DCCLayer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_conv2 = cuda_convnet.Conv2DCCLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(c01b=True),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1),
            dimshuffle=True
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_shape,
            nonlinearity=None,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        return l_out


    def build_nips_network_dnn(self, input_width, input_height, output_shape,
                               num_frames, batch_size):
        """
        Build a network consistent with the 2013 NIPS paper.
        """
        # Import it here, in case it isn't installed.
        from lasagne.layers import dnn

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )


        l_conv1 = dnn.Conv2DDNNLayer(
            l_in,
            num_filters=16,
            filter_size=(8, 8),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_conv2 = dnn.Conv2DDNNLayer(
            l_conv1,
            num_filters=32,
            filter_size=(4, 4),
            stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_conv2,
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(),
            W=lasagne.init.Normal(.01),
            b=lasagne.init.Constant(.1)
        )

        # Create an output layer with D*G outputs, then reshape
        # I tried having one output layer for each degree of freedom, but
        # that made it hard to build an expression for the Q values in Theano
        # (The reshape might be costly, but matrix indexing without the
        # reshape is a nightmare.)
        # Old comment below:
        #
        # There's one output layer for each degree of freedom.
        # Note that technically, we could have a single output layer
        # with sum(output_shape) output units, and this is probably
        # more optimal because we'd have 1 matrix instead of D smaller
        # ones (where D = degrees of freedom.) However, doing it this way
        # makes computing the best action easier.
        # Consider changing this around if this bottlenecks badly.

        l_out = lasagne.layers.DenseLayer(
                l_hidden1,
                num_units=self.degrees_of_freedom * self.max_granularity,
                nonlinearity=None,
                #W=lasagne.init.HeUniform(),
                W=lasagne.init.Normal(.01),
                b=lasagne.init.Constant(.1)
        )
        l_shaped_out = l_out.reshape(
            (self.degrees_of_freedom, self.max_granularity)
        )

        return l_shaped_out


    def build_linear_network(self, input_width, input_height, output_shape,
                             num_frames, batch_size):
        """
        Build a simple linear learner.  Useful for creating
        tests that sanity-check the weight update code.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_frames, input_width, input_height)
        )

        l_out = lasagne.layers.DenseLayer(
            l_in,
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.Constant(0.0),
            b=None
        )

        return l_out

def main():
    net = DeepQLearner(84, 84, 16, 4, .99, .00025, .95, .95, 10000,
                       32, 'nature_cuda')


if __name__ == '__main__':
    main()
