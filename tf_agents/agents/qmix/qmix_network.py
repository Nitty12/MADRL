import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common

"""Nitty: Hyper nets for QMIX"""

@gin.configurable
class HyperNetwork(network.Network):
    """Creates an actor network."""

    def __init__(self,
                 input_tensor_spec,
                 fc_layer_params=None,
                 dropout_layer_params=None,
                 conv_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 name='HyperNetwork'):
        """Creates an instance of `ActorNetwork`.

        Args:
          input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
            inputs.
          output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
            the outputs.
          fc_layer_params: Optional list of fully_connected parameters, where each
            item is the number of units in the layer.
          dropout_layer_params: Optional list of dropout layer parameters, each item
            is the fraction of input units to drop or a dictionary of parameters
            according to the keras.Dropout documentation. The additional parameter
            `permanent', if set to True, allows to apply dropout at inference for
            approximated Bayesian inference. The dropout layers are interleaved with
            the fully connected layers; there is a dropout layer after each fully
            connected layer, except if the entry in the list is None. This list must
            have the same length of fc_layer_params, or be None.
          conv_layer_params: Optional list of convolution layers parameters, where
            each item is a length-three tuple indicating (filters, kernel_size,
            stride).
          activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
          name: A string representing name of the network.

        Raises:
          ValueError: If `input_tensor_spec` or `action_spec` contains more than one
            item, or if the action data type is not `float`.
        """

        super(HyperNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        if len(tf.nest.flatten(input_tensor_spec)) > 1:
            raise ValueError('Only a single observation is supported by this network')

        # Replace mlp_layers with encoding networks.
        self.hyper_w1 = utils.hyper_layers(
            conv_layer_params,
            fc_layer_params['hyper_w1'],
            dropout_layer_params['hyper_w1'],
            activation_fn=activation_fn['hyper_w1'],
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='hyper_w1')

        self.hyper_w2 = utils.hyper_layers(
            conv_layer_params,
            fc_layer_params['hyper_w2'],
            dropout_layer_params['hyper_w2'],
            activation_fn=activation_fn['hyper_w2'],
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='hyper_w2')

        self.hyper_b1 = utils.hyper_layers(
            conv_layer_params,
            fc_layer_params['hyper_b1'],
            dropout_layer_params['hyper_b1'],
            activation_fn=None,
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='hyper_b1',
            activation_needed=False)

        self.hyper_b2 = utils.hyper_layers(
            conv_layer_params,
            fc_layer_params['hyper_b2'],
            dropout_layer_params['hyper_b2'],
            activation_fn=activation_fn['hyper_b2'],
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=1. / 3., mode='fan_in', distribution='uniform'),
            name='hyper_b2',
            activation_needed=True)


    def call(self, q_values, observations, step_type=(), network_state=(), training=False):
        del step_type  # unused.
        observations = tf.nest.flatten(observations)
        w1 = tf.math.abs(self.hyper_w1(observations))
        b1 = self.hyper_b1(observations)
        w1 = tf.reshape(w1, [-1, nAgents, qmixHiddenDim])
        b1 = tf.reshape(b1, [-1, 1, qmixHiddenDim])
        hidden = tf.keras.activations.elu((tf.matmul(q_values, w1) + b1))

        w2 = tf.math.abs(self.hyper_w2(observations))
        b1 = self.hyper_b2(observations)
        w2 = tf.reshape(w2, [-1, qmixHiddenDim, 1])
        b2 = tf.reshape(b2, [-1, 1, 1])

        output = tf.matmul(hidden, w2) + b2
        return output