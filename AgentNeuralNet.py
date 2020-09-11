import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.agents import DdpgAgent
from tf_agents.policies import random_tf_policy
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common
import optuna
import gin
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import training as training_lib
from tf_agents.trajectories import trajectory
import copy


class MADDPGAgent:
    def __init__(self):
        # Hyperparameters
        self.learning_rate = 1e-3
        self.fc_layer_params_actor = (100,)
        self.fc_dropout_layer_params_actor = None
        self.fc_layer_params_critic_obs = (50,)
        self.fc_dropout_layer_params_critic_obs = None
        self.fc_layer_params_critic_merged = (100,)
        self.fc_dropout_layer_params_critic_merged = None
        self.discount_factor = 0.99
        self.actor_activation_fn = tf.keras.activations.relu
        self.critic_activation_fn = tf.keras.activations.relu

        self.ActorNetwork = None
        self.CriticNetwork = None
        self.optimizer = None
        self.agent = None
        self.eval_policy = None
        self.collect_policy = None
        self.random_policy = None
        self.replay_buffer = None

    def hyperParameterInit(self, parameterDict):
        """hyperparameter optimization with optuna"""
        self.learning_rate = parameterDict['learning_rate']
        self.discount_factor = parameterDict['discount_factor']
        if parameterDict['actor_activation_fn']=='relu':
            self.actor_activation_fn = tf.keras.activations.relu
        elif parameterDict['actor_activation_fn']=='leaky_relu':
            self.actor_activation_fn = tf.nn.leaky_relu
        if parameterDict['critic_activation_fn']=='relu':
            self.critic_activation_fn = tf.keras.activations.relu
        elif parameterDict['critic_activation_fn']=='leaky_relu':
            self.critic_activation_fn = tf.nn.leaky_relu
        self.fc_layer_params_actor = parameterDict['fc_layer_params_actor']
        self.fc_dropout_layer_params_actor = parameterDict['fc_dropout_layer_params_actor']
        self.fc_layer_params_critic_obs = parameterDict['fc_layer_params_critic_obs']
        self.fc_dropout_layer_params_critic_obs = parameterDict['fc_dropout_layer_params_critic_obs']
        self.fc_layer_params_critic_merged = parameterDict['fc_layer_params_critic_merged']
        self.fc_dropout_layer_params_critic_merged = parameterDict['fc_dropout_layer_params_critic_merged']

    def initialize(self, train_env, train_step_counter, index):
        self.ActorNetwork = tf_agents.agents.ddpg.actor_network.ActorNetwork(
            input_tensor_spec=train_env.observation_spec()[index],
            output_tensor_spec=train_env.action_spec()[index],
            fc_layer_params=self.fc_layer_params_actor,
            dropout_layer_params=self.fc_dropout_layer_params_actor, conv_layer_params=None,
            activation_fn=self.actor_activation_fn, name='ActorNetwork')

        # centralized critic
        """Flatten the observation and action tuple as the critic network only accepts single obs and act
            We need it to see the obs and act of all agents"""
        CriticNetwork = tf_agents.agents.ddpg.critic_network.CriticNetwork(
            input_tensor_spec=(train_env.total_observation_spec(), train_env.total_action_spec()),
            observation_fc_layer_params=self.fc_layer_params_critic_obs,
            observation_dropout_layer_params=self.fc_dropout_layer_params_critic_obs,
            joint_fc_layer_params=self.fc_layer_params_critic_merged,
            joint_dropout_layer_params=self.fc_dropout_layer_params_critic_merged,
            name='CriticNetwork')

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.agent = DdpgAgent(time_step_spec=train_env.time_step_spec(), action_spec=train_env.action_spec()[index],
                               actor_network=self.ActorNetwork, critic_network=CriticNetwork,
                               actor_optimizer=self.optimizer, critic_optimizer=self.optimizer, ou_stddev=0.2,
                               ou_damping=0.15,
                               target_update_tau=0.001, target_update_period=10, gamma=self.discount_factor,
                               train_step_counter=train_step_counter,
                               debug_summaries=True)

        self.agent.initialize()
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                             train_env.action_spec())

class QMIXAgent:
    def __init__(self):
        #fc_layer_params['hyper_w1'] = (hyper hidden dim, nAgents*qmix hidden dim)
        #activation_fn['hyper_w1'] = Relu
        #fc_layer_params['hyper_w2'] = (hyper hidden dim, qmix hidden dim)
        #activation_fn['hyper_w2'] = Relu
        #fc_layer_params['hyper_b1'] = (nAgents*qmix hidden dim,)
        #fc_layer_params['hyper_b2'] = (qmix hidden dim,1)
        # activation_fn['hyper_b2'] = Relu
        self.fc_layer_params = (100,)
        self.learning_rate = 1e-3
        pass
    def hyperParameterInit(self, parameterDict):
        pass
    def initialize(self, train_env, train_step_counter, index):
        self.QNetwork = q_network.QNetwork(
                                input_tensor_spec=train_env.observation_spec()[index],
                                action_spec=train_env.action_spec()[index],
                                fc_layer_params=self.fc_layer_params)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.agent = dqn_agent.DqnAgent(
                            train_env.time_step_spec(),
                            train_env.action_spec()[index],
                            q_network=self.QNetwork,
                            optimizer=self.optimizer,
                            td_errors_loss_fn=common.element_wise_squared_loss,
                            train_step_counter=train_step_counter)
        self.agent.initialize()
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                             train_env.action_spec())


class QMIXMixingNetwork(network.Network):
    def __init__(self,
                 fc_layer_params=None,
                 dropout_layer_params=None,
                 conv_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 name='HyperNetwork'):

        super(QMIXMixingNetwork, self).__init__(
            input_tensor_spec=None,
            state_spec=(),
            name=name)

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


class QMIX():
    def __init__(self):
        self.QMIXNet = QMIXMixingNetwork()
        self.TargetQMIXNet = copy.deepcopy(self.QMIXNet)
        self._epsilon_greedy = epsilon_greedy
        self._n_step_update = n_step_update
        self._boltzmann_temperature = boltzmann_temperature
        self._optimizer = optimizer
        self._td_errors_loss_fn = common.element_wise_squared_loss
        self._gamma = gamma
        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period)
        self.train_step_counter = None
    def train(self):


    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

        For each weight w_s in the q network, and its corresponding
        weight w_t in the target_q_network, a soft update is:
        w_t = (1 - tau) * w_t + tau * w_s
        """
        with tf.name_scope('update_targets'):
            def update():
                return common.soft_variables_update(
                    self.QMIXNet.variables,
                    self.TargetQMIXNet.variables,
                    tau,
                    tau_non_trainable=1.0)

            return common.Periodically(update, period, 'periodic_update_targets')

    def _train(self, experience, weights):
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                weights=weights,
                training=True)
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = self.QMIXNet.trainable_weights
        non_trainable_weights = self.QMIXNet.non_trainable_weights
        assert list(variables_to_train), "No variables in the agent's QMIX network."
        grads = tape.gradient(loss_info.loss, variables_to_train)
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = list(zip(grads, variables_to_train))

        training_lib.apply_gradients(
            self._optimizer, grads_and_vars, global_step=self.train_step_counter)

        self._update_target()

        return loss_info

    def _loss(self,
              experience,
              td_errors_loss_fn,
              gamma=1.0,
              reward_scale_factor=1.0,
              weights=None,
              training=False):

        squeeze_time_dim = not self.QMIXNet.state_spec
        if self._n_step_update == 1:
            time_steps, policy_steps, next_time_steps = (
                trajectory.experience_to_transitions(experience, squeeze_time_dim))
            actions = policy_steps.action
        else:
            # To compute n-step returns, we need the first time steps, the first
            # actions, and the last time steps. Therefore we extract the first and
            # last transitions from our Trajectory.
            first_two_steps = tf.nest.map_structure(lambda x: x[:, :2], experience)
            last_two_steps = tf.nest.map_structure(lambda x: x[:, -2:], experience)
            time_steps, policy_steps, _ = (
                trajectory.experience_to_transitions(
                    first_two_steps, squeeze_time_dim))
            actions = policy_steps.action
            _, _, next_time_steps = (
                trajectory.experience_to_transitions(
                    last_two_steps, squeeze_time_dim))

        with tf.name_scope('loss'):
            q_values = self._compute_q_values(time_steps, actions, training=training)

            next_q_values = self._compute_next_q_values(
                next_time_steps, policy_steps.info)

            if self._n_step_update == 1:
                # Special case for n = 1 to avoid a loss of performance.
                td_targets = compute_td_targets(
                    next_q_values,
                    rewards=reward_scale_factor * next_time_steps.reward,
                    discounts=gamma * next_time_steps.discount)
            else:
                # When computing discounted return, we need to throw out the last time
                # index of both reward and discount, which are filled with dummy values
                # to match the dimensions of the observation.
                rewards = reward_scale_factor * experience.reward[:, :-1]
                discounts = gamma * experience.discount[:, :-1]

                # TODO(b/134618876): Properly handle Trajectories that include episode
                # boundaries with nonzero discount.

                td_targets = value_ops.discounted_return(
                    rewards=rewards,
                    discounts=discounts,
                    final_value=next_q_values,
                    time_major=False,
                    provide_all_returns=False)

            valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
            td_error = valid_mask * (td_targets - q_values)

            td_loss = valid_mask * td_errors_loss_fn(td_targets, q_values)

            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

            # Aggregate across the elements of the batch and add regularization loss.
            # Note: We use an element wise loss above to ensure each element is always
            #   weighted by 1/N where N is the batch size, even when some of the
            #   weights are zero due to boundary transitions. Weighting by 1/K where K
            #   is the actual number of non-zero weight would artificially increase
            #   their contribution in the loss. Think about what would happen as
            #   the number of boundary samples increases.

            agg_loss = common.aggregate_losses(
                per_example_loss=td_loss,
                sample_weight=weights,
                regularization_loss=self._q_network.losses)
            total_loss = agg_loss.total_loss

            losses_dict = {'td_loss': agg_loss.weighted,
                           'reg_loss': agg_loss.regularization,
                           'total_loss': total_loss}

            common.summarize_scalar_dict(losses_dict,
                                         step=self.train_step_counter,
                                         name_scope='Losses/')

            if self._summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in self._q_network.trainable_weights:
                        tf.compat.v2.summary.histogram(
                            name=var.name.replace(':', '_'),
                            data=var,
                            step=self.train_step_counter)

            if self._debug_summaries:
                diff_q_values = q_values - next_q_values
                common.generate_tensor_summaries('td_error', td_error,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('td_loss', td_loss,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('q_values', q_values,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('next_q_values', next_q_values,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                                 self.train_step_counter)

            return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,
                                                             td_error=td_error))

    def _compute_q_values(self, time_steps, actions, training=False):
        network_observation = time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        q_values, _ = self._q_network(network_observation, time_steps.step_type,
                                      training=training)
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = self._action_spec.shape.rank > 0
        return common.index_with_actions(
            q_values,
            tf.cast(actions, dtype=tf.int32),
            multi_dim_actions=multi_dim_actions)

    def _compute_next_q_values(self, next_time_steps, info):
        """Compute the q value of the next state for TD error computation.

        Args:
          next_time_steps: A batch of next timesteps
          info: PolicyStep.info that may be used by other agents inherited from
            dqn_agent.

        Returns:
          A tensor of Q values for the given next state.
        """
        network_observation = next_time_steps.observation

        if self._observation_and_action_constraint_splitter is not None:
            network_observation, _ = self._observation_and_action_constraint_splitter(
                network_observation)

        next_target_q_values, _ = self._target_q_network(
            network_observation, next_time_steps.step_type)
        batch_size = (
                next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
        dummy_state = self._target_greedy_policy.get_initial_state(batch_size)
        # Find the greedy actions using our target greedy policy. This ensures that
        # action constraints are respected and helps centralize the greedy logic.
        greedy_actions = self._target_greedy_policy.action(
            next_time_steps, dummy_state).action

        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = tf.nest.flatten(self._action_spec)[0].shape.rank > 0
        return common.index_with_actions(
            next_target_q_values,
            greedy_actions,
            multi_dim_actions=multi_dim_actions)