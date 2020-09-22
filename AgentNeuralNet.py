import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.agents import DdpgAgent
from tf_agents.policies import random_tf_policy
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.dqn.dqn_agent import DqnLossInfo
from tf_agents.networks import q_network
from tf_agents.utils import common
import optuna
import gin
from tf_agents.agents import tf_agent
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import value_ops
from tf_agents.utils import training as training_lib
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
import copy
import re


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
        if parameterDict['actor_activation_fn'] == 'relu':
            self.actor_activation_fn = tf.keras.activations.relu
        elif parameterDict['actor_activation_fn'] == 'leaky_relu':
            self.actor_activation_fn = tf.nn.leaky_relu
        if parameterDict['critic_activation_fn'] == 'relu':
            self.critic_activation_fn = tf.keras.activations.relu
        elif parameterDict['critic_activation_fn'] == 'leaky_relu':
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
                               target_update_tau=0.01, target_update_period=10, gamma=self.discount_factor,
                               train_step_counter=train_step_counter,
                               debug_summaries=True)

        self.agent.initialize()
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                             train_env.action_spec())


class QAgent:
    def __init__(self, type):
        self.fc_layer_params = (100,)
        self.dropout_layer_params = None
        self.learning_rate = 1e-3
        self.type = type

    def hyperParameterInit(self, parameterDict):
        self.learning_rate = parameterDict['learning_rate']
        self.fc_layer_params = parameterDict['fc_layer_params']
        self.dropout_layer_params = parameterDict['fc_dropout_layer_params']

    def initialize(self, train_env, train_step_counter, index):
        if self.type == 'sbm_spot':
            pos = 0
        elif self.type == 'sbm_flex':
            pos = 24
        else:
            pos = 48
        obs_spec = train_env.total_qmix_observation_spec()[index]
        action_spec = train_env.total_qmix_action_spec()[index+pos]
        self.QNetwork = q_network.QNetwork(
            input_tensor_spec=obs_spec,
            action_spec=action_spec,
            fc_layer_params=self.fc_layer_params,
            dropout_layer_params=self.dropout_layer_params)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        individual_qmix_time_step_spec = ts.get_individual_qmix_time_step_spec(train_env.time_step_spec())
        self.agent = dqn_agent.DqnAgent(
            individual_qmix_time_step_spec,
            action_spec,
            q_network=self.QNetwork,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            target_update_tau=0.01,
            target_update_period=10)
        self.agent.initialize()
        self.eval_policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                             train_env.action_spec())

    def _compute_q_values(self, time_steps, actions, index, time, training=False):
        """spot and flex dispatches of last day (48) + MCP current hour + current state (spot/ flex)"""
        obs_indices = tf.convert_to_tensor(list(range(0, 48)) + [48 + time] + [71])
        observation = tf.gather(time_steps.observation[index], indices=obs_indices, axis=-1)
        network_observation = observation
        q_values, _ = self.agent._q_network(network_observation, time_steps.step_type,
                                      training=training)
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = self.agent._action_spec.shape.rank > 0
        actions = tf.reshape(actions, [-1, 1])
        return common.index_with_actions(q_values, tf.cast(actions, dtype=tf.int32),
                                         multi_dim_actions=multi_dim_actions)

    def _compute_next_q_values(self, next_time_steps, index, time):
        """spot and flex dispatches of last day (48) + MCP current hour + current state (spot/ flex)"""
        obs_indices = tf.convert_to_tensor(list(range(0, 48)) + [48 + time] + [71])
        observation = tf.gather(next_time_steps.observation[index], indices=obs_indices, axis=-1)
        network_observation = observation
        next_target_q_values, _ = self.agent._target_q_network(network_observation, next_time_steps.step_type)
        batch_size = (next_target_q_values.shape[0] or tf.shape(next_target_q_values)[0])
        dummy_state = self.agent._target_greedy_policy.get_initial_state(batch_size)
        # Find the greedy actions using our target greedy policy. This ensures that
        # action constraints are respected and helps centralize the greedy logic.
        next_individual_qmix_time_step = ts.get_individual_qmix_time_step(next_time_steps, index, time)
        greedy_actions = self.agent._target_greedy_policy.action(next_individual_qmix_time_step, dummy_state).action
        # Handle action_spec.shape=(), and shape=(1,) by using the multi_dim_actions
        # param. Note: assumes len(tf.nest.flatten(action_spec)) == 1.
        multi_dim_actions = tf.nest.flatten(self.agent._action_spec)[0].shape.rank > 0
        return common.index_with_actions(next_target_q_values, greedy_actions,
                                         multi_dim_actions=multi_dim_actions)


class QMIXMixingNetwork(network.Network):
    def __init__(self,
                 fc_layer_params=None,
                 dropout_layer_params=None,
                 conv_layer_params=None,
                 activation_fn=None,
                 nAgents=1,
                 qmix_hidden_dim=64,
                 name='HyperNetwork'):
        self.nAgents = nAgents
        self.qmix_hidden_dim = qmix_hidden_dim

        super(QMIXMixingNetwork, self).__init__(
            input_tensor_spec=None,
            state_spec=(),
            name=name)

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
        """convert to joint observation"""
        observations = tf.concat(observations, axis=-1)
        q_values = tf.reshape(q_values, [-1,1,self.nAgents])

        output = observations
        for layer in self.hyper_w1:
            output = layer(output, training=training)
        w1 = tf.math.abs(output)

        output = observations
        for layer in self.hyper_b1:
            output = layer(output, training=training)
        b1 = output

        w1 = tf.reshape(w1, [-1, self.nAgents, self.qmix_hidden_dim])
        b1 = tf.reshape(b1, [-1, 1, self.qmix_hidden_dim])
        hidden = tf.keras.activations.elu((tf.matmul(q_values, w1) + b1))

        output = observations
        for layer in self.hyper_w2:
            output = layer(output, training=training)
        w2 = tf.math.abs(output)

        output = observations
        for layer in self.hyper_b2:
            output = layer(output, training=training)
        b2 = output

        w2 = tf.reshape(w2, [-1, self.qmix_hidden_dim, 1])
        b2 = tf.reshape(b2, [-1, 1, 1])

        output = tf.matmul(hidden, w2) + b2
        return output, network_state


class QMIX():
    def __init__(self, nAgents, time_step_spec, train_step_counter, summary_writer=None,
                 debug_summaries=False, summarize_grads_and_vars=False, enable_summaries=True):
        self.hyper_hidden_dim = 256
        self.qmix_hidden_dim = 64
        self.learning_rate = 0.001
        self.nAgents =nAgents
        self.time_step_spec = time_step_spec
        self._epsilon_greedy = 0.1
        self._n_step_update = 1
        self._td_errors_loss_fn = common.element_wise_squared_loss
        self._gamma = 0.99
        self._update_target = self._get_target_updater(tau=0.01, period=10)
        self.train_step_counter = train_step_counter
        self.summary_writer = summary_writer
        self.debug_summaries = debug_summaries
        self.summarize_grads_and_vars = summarize_grads_and_vars
        self.enable_summaries = enable_summaries

    def hyperParameterInit(self, parameterDict):
        # self.hyper_hidden_dim = parameterDict['hyper_hidden_dim']
        # self.qmix_hidden_dim = parameterDict['qmix_hidden_dim']
        # self.learning_rate = parameterDict['qmix_learning_rate']
        fc_layer_params = {}
        dropout_layer_params = {}
        activation_fn = {}
        fc_layer_params['hyper_w1'] = (self.hyper_hidden_dim, self.nAgents * self.qmix_hidden_dim)
        dropout_layer_params['hyper_w1'] = None
        activation_fn['hyper_w1'] = tf.keras.activations.relu
        fc_layer_params['hyper_w2'] = (self.hyper_hidden_dim, self.qmix_hidden_dim)
        dropout_layer_params['hyper_w2'] = None
        activation_fn['hyper_w2'] = tf.keras.activations.relu
        fc_layer_params['hyper_b1'] = (self.qmix_hidden_dim,)
        dropout_layer_params['hyper_b1'] = None
        fc_layer_params['hyper_b2'] = (self.qmix_hidden_dim, 1)
        dropout_layer_params['hyper_b2'] = None
        activation_fn['hyper_b2'] = tf.keras.activations.relu
        self._optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.QMIXNet = QMIXMixingNetwork(fc_layer_params=fc_layer_params,
                                         dropout_layer_params=dropout_layer_params,
                                         activation_fn=activation_fn,
                                         nAgents=self.nAgents, qmix_hidden_dim=self.qmix_hidden_dim)
        self.TargetQMIXNet = copy.deepcopy(self.QMIXNet)


    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.
        For each weight w_s in the network, and its corresponding
        weight w_t in the target_network, a soft update is:
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

    def train(self, experience, agents, nameDict, networkDict):
        """QMIX - get the Q values from the target network and main network of all the agents"""
        time_steps, policy_steps, next_time_steps = (
            trajectory.experience_to_transitions(experience, squeeze_time_dim=True))

        with tf.GradientTape() as tape:
            loss_info = self._loss(
                time_steps, policy_steps, next_time_steps,
                agents, nameDict, networkDict,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                training=True)
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = getTrainableVariables(networkDict)
        variables_to_train.append(self.QMIXNet.trainable_weights)
        variables_to_train = tf.nest.flatten(variables_to_train)
        assert list(variables_to_train), "No variables in the agent's QMIX network."
        grads = tape.gradient(loss_info.loss, variables_to_train)
        grads_and_vars = list(zip(grads, variables_to_train))
        training_lib.apply_gradients(
            self._optimizer, grads_and_vars, global_step=self.train_step_counter)
        self._update_target()
        return loss_info

    def _loss(self, time_steps, policy_steps, next_time_steps,
              agents, nameDict, networkDict,
              td_errors_loss_fn,
              gamma=1.0,
              weights=None,
              training=False):
        with tf.name_scope('loss'):
            total_agents_target = []
            total_agents_main = []
            for i, flexAgent in enumerate(agents):
                for node in nameDict:
                    target = None
                    for type, names in nameDict[node].items():
                        if flexAgent.id in names:
                            target = []
                            main = []
                            for net in networkDict[node][type]:
                                action_index = -1
                                for t in range(24):
                                    action_index += 1
                                    actions = tf.gather(policy_steps.action[i], indices=action_index, axis=-1)
                                    individual_target = net._compute_next_q_values(next_time_steps, index=i, time=t)
                                    individual_main = net._compute_q_values(time_steps, actions,
                                                                            index=i, time=t, training=True)
                                    target.append(tf.reshape(individual_target, [-1, 1]))
                                    main.append(tf.reshape(individual_main, [-1, 1]))
                            break
                    if target is not None:
                        break
                total_agents_target.append(tf.concat(target, -1))
                total_agents_main.append(tf.concat(main, -1))
            total_agents_target = tf.concat(total_agents_target, -1)
            total_agents_main = tf.concat(total_agents_main, -1)

            q_total, _ = self.QMIXNet(total_agents_main, time_steps.observation, training=training)
            q_total = tf.squeeze(q_total)
            target_q_total, _ = self.TargetQMIXNet(total_agents_target, next_time_steps.observation, training=False)
            target_q_total = tf.squeeze(target_q_total)
            """using the mean reward for all the agents"""
            mean_reward = tf.reduce_mean(next_time_steps.reward, axis=1)
            td_targets = tf.stop_gradient(mean_reward + gamma * next_time_steps.discount * target_q_total)

            valid_mask = tf.cast(~time_steps.is_last(), tf.float32)
            td_error = valid_mask * (td_targets - q_total)
            td_loss = valid_mask * tf.compat.v1.losses.absolute_difference(td_targets, q_total,
                                                                        reduction=tf.compat.v1.losses.Reduction.NONE)
            # td_loss = valid_mask * td_errors_loss_fn(td_targets, q_total)

            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

            # Aggregate across the elements of the batch and add regularization loss.
            agg_loss = common.aggregate_losses(
                per_example_loss=td_loss,
                sample_weight=weights)
            total_loss = agg_loss.total_loss

            if self.summary_writer is not None:
                with self.summary_writer.as_default():
                    tf.summary.scalar('loss', total_loss, step=self.train_step_counter)

            losses_dict = {'td_loss': agg_loss.weighted,
                           'reg_loss': agg_loss.regularization,
                           'total_loss': total_loss}

            common.summarize_scalar_dict(losses_dict,
                                         step=self.train_step_counter,
                                         name_scope='Losses/')

            if self.summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in self.QMIXNet.trainable_weights:
                        tf.compat.v2.summary.histogram(
                            name=var.name.replace(':', '_'),
                            data=var,
                            step=self.train_step_counter)

            if self.debug_summaries:
                diff_q_values = q_total - target_q_total
                common.generate_tensor_summaries('td_error', td_error,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('td_loss', td_loss,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('q_total', q_total,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('target_q_total', target_q_total,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                                 self.train_step_counter)

            return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,
                                                             td_error=td_error))

def getTrainableVariables(networkDict):
    # get the weights for the individual Q networks of all agents for QMIX
    networkList = []
    variables_to_train = []
    for node in networkDict:
        for net in networkDict[node].values():
            if isinstance(net, list):
                networkList.extend(net)
            else:
                networkList.append(net)
    for net in networkList:
        variables_to_train.append(net.agent._q_network.trainable_weights)
    # variables_to_train=networkList[0].agent._q_network.trainable_weights
    return variables_to_train

class IQL():
    def __init__(self, networkDict, nameDict, nameList, time_step_spec, train_step_counter, summary_writer=None,
                 debug_summaries=False, summarize_grads_and_vars=False, enable_summaries=True):
        self.learning_rate = 0.001
        self.networkDict = networkDict
        self.nameDict = nameDict
        self.nameList = nameList
        self.time_step_spec = time_step_spec
        self._epsilon_greedy = 0.1
        self._n_step_update = 1
        self._td_errors_loss_fn = common.element_wise_squared_loss
        self._gamma = 0.99
        self.train_step_counter = train_step_counter
        self.summary_writer = summary_writer
        self.debug_summaries = debug_summaries
        self.summarize_grads_and_vars = summarize_grads_and_vars
        self.enable_summaries = enable_summaries

    def train(self, experience, agents, nameDict, networkDict):
        time_steps, policy_steps, next_time_steps = (
            trajectory.experience_to_transitions(experience, squeeze_time_dim=True))
        loss_list = []
        for i, flexAgent in enumerate(agents):
            for node in nameDict:
                for type, names in nameDict[node].items():
                    if flexAgent.id in names:
                        for net in networkDict[node][type]:
                            action_index = -1
                            for t in range(24):
                                action_index += 1
                                actions = tf.gather(policy_steps.action[i], indices=action_index, axis=-1)
                                individual_iql_time_step = ts.get_individual_iql_time_step(time_steps, index=i, time=t)
                                individual_iql_next_time_step = ts.get_individual_iql_time_step(next_time_steps, index=i, time=t)
                                train_loss = self.train_single_net(net, individual_iql_time_step,
                                                                   individual_iql_next_time_step,
                                                                   time_steps, actions, next_time_steps, i, t).loss
                                loss_list.append(train_loss)
                    break
        self.train_step_counter.assign_add(1)
        if self.summary_writer is not None:
            with self.summary_writer.as_default():
                avg_loss = sum(loss_list)/len(loss_list)
                tf.summary.scalar('loss', avg_loss, step=self.train_step_counter)
        return avg_loss

    def train_single_net(self, net, individual_iql_time_step, individual_iql_next_time_step,
                         time_steps, actions, next_time_steps, i, t):
        with tf.GradientTape() as tape:
            loss_info = self._loss(net,
                                   individual_iql_time_step, individual_iql_next_time_step,
                                   time_steps, actions, next_time_steps, i, t,
                                   td_errors_loss_fn=net.agent._td_errors_loss_fn,
                                   gamma=net.agent._gamma,
                                   training=True)
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = net.agent._q_network.trainable_weights
        non_trainable_weights = net.agent._q_network.non_trainable_weights
        assert list(variables_to_train), "No variables in the agent's QMIX network."
        grads = tape.gradient(loss_info.loss, variables_to_train)
        grads_and_vars = list(zip(grads, variables_to_train))
        training_lib.apply_gradients(
            net.agent._optimizer, grads_and_vars, global_step=net.agent.train_step_counter)
        net.agent._update_target()
        return loss_info

    def _loss(self, net, individual_iql_time_step, individual_iql_next_time_step,
              time_steps, actions, next_time_steps, i, t,
              td_errors_loss_fn,
              gamma=1.0,
              weights=None,
              training=False):
        with tf.name_scope('loss'):

            individual_target = tf.reshape(net._compute_next_q_values(next_time_steps, index=i, time=t), [-1,1])
            individual_main = tf.reshape(net._compute_q_values(time_steps, actions,
                                                    index=i, time=t, training=True), [-1,1])

            reward = tf.reshape(individual_iql_next_time_step.reward, [-1,1])
            discount = tf.reshape(individual_iql_next_time_step.discount, [-1, 1])
            td_targets = tf.stop_gradient(reward + gamma * discount * individual_target)

            valid_mask = tf.reshape(tf.cast(~individual_iql_time_step.is_last(), tf.float32), [-1,1])
            td_error = valid_mask * (td_targets - individual_main)
            td_loss = valid_mask * tf.compat.v1.losses.absolute_difference(td_targets, individual_main,
                                                                        reduction=tf.compat.v1.losses.Reduction.NONE)
            # td_loss = valid_mask * td_errors_loss_fn(td_targets, q_total)

            if nest_utils.is_batched_nested_tensors(
                    individual_iql_time_step, net.agent.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                td_loss = tf.reduce_sum(input_tensor=td_loss, axis=1)

            # Aggregate across the elements of the batch and add regularization loss.
            agg_loss = common.aggregate_losses(
                per_example_loss=td_loss,
                sample_weight=weights)
            total_loss = agg_loss.total_loss

            losses_dict = {'td_loss': agg_loss.weighted,
                           'reg_loss': agg_loss.regularization,
                           'total_loss': total_loss}

            common.summarize_scalar_dict(losses_dict,
                                         step=self.train_step_counter,
                                         name_scope='Losses/')

            if self.summarize_grads_and_vars:
                with tf.name_scope('Variables/'):
                    for var in net.agent.trainable_weights:
                        tf.compat.v2.summary.histogram(
                            name=var.name.replace(':', '_'),
                            data=var,
                            step=self.train_step_counter)

            if self.debug_summaries:
                diff_q_values = individual_main - individual_target
                common.generate_tensor_summaries('td_error', td_error,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('td_loss', td_loss,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('q_total', individual_main,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('target_q_total', individual_target,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('diff_q_values', diff_q_values,
                                                 self.train_step_counter)

            return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,
                                                             td_error=td_error))
