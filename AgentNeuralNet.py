import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.agents import DdpgAgent
from tf_agents.policies import random_tf_policy
import optuna


class AgentNeuralNet:
    def __init__(self, trial=None):
        # Hyperparameters
        if trial is None:
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

        else:
            """hyperparameter optimization with optuna"""
            self.learning_rate = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
            self.discount_factor = trial.suggest_int('discount_factor', 0.95, 0.999, log=True)

            self.actor_activation_fn = trial.suggest_categorical('actor_activation_fn',
                                                                 [tf.keras.activations.relu, tf.nn.leaky_relu], log=True)
            self.critic_activation_fn = trial.suggest_categorical('critic_activation_fn',
                                                                 [tf.keras.activations.relu, tf.nn.leaky_relu], log=True)

            self.fc_layer_params_actor = []
            self.fc_dropout_layer_params_actor = []
            actor_n_layers = trial.suggest_int('actor_n_layers', 1, 3, log=True)
            for i in actor_n_layers:
                num_hidden = trial.suggest_int("actor_n_units_L{}".format(i+1), 50, 300, log=True)
                dropout_rate = trial.suggest_float("actor_dropout_rate_L{}".format(i+1), 0, 0.7, log=True)
                self.fc_layer_params_actor.append(num_hidden)
                self.fc_dropout_layer_params_actor.append(dropout_rate)

            self.fc_layer_params_critic_obs = []
            self.fc_dropout_layer_params_critic_obs = []
            critic_obs_n_layers = trial.suggest_int('critic_obs_n_layers', 1, 2)
            for i in critic_obs_n_layers:
                num_hidden = trial.suggest_int("critic_obs_n_units_L{}".format(i+1), 50, 300, log=True)
                dropout_rate = trial.suggest_float("critic_obs_dropout_rate_L{}".format(i+1), 0, 0.7, log=True)
                self.fc_layer_params_critic_obs.append(num_hidden)
                self.fc_dropout_layer_params_critic_obs.append(dropout_rate)

            self.fc_layer_params_critic_merged = []
            self.fc_dropout_layer_params_critic_merged = []
            critic_merged_n_layers = trial.suggest_int('critic_merged_n_layers', 1, 2)
            for i in critic_merged_n_layers:
                num_hidden = trial.suggest_int("critic_merged_n_units_L{}".format(i+1), 50, 300, log=True)
                dropout_rate = trial.suggest_float("critic_merged_dropout_rate_L{}".format(i+1), 0, 0.7, log=True)
                self.fc_layer_params_critic_merged.append(num_hidden)
                self.fc_dropout_layer_params_critic_merged.append(dropout_rate)

        self.ActorNetwork = None
        self.CriticNetwork = None
        self.optimizer = None
        self.agent = None
        self.eval_policy = None
        self.collect_policy = None
        self.random_policy = None
        self.replay_buffer = None

        # for testing
        self.lowSpotBidLimit = None
        self.highSpotBidLimit = None
        self.lowFlexBidLimit = None
        self.highFlexBidLimit = None
        self.lowPriceLimit = None
        self.highPriceLimit = None

    def initialize(self, train_env, train_step_counter, index):
        self.ActorNetwork = tf_agents.agents.ddpg.actor_network.ActorNetwork(
            input_tensor_spec=train_env.observation_spec()[index],
            output_tensor_spec=train_env.action_spec()[index],
            fc_layer_params=self.fc_layer_params_actor,
            dropout_layer_params=None, conv_layer_params=None,
            activation_fn=tf.keras.activations.relu, name='ActorNetwork')

        # centralized critic
        """Flatten the observation and action tuple as the critic network only accepts single obs and act
            We need it to see the obs and act of all agents"""

        CriticNetwork = tf_agents.agents.ddpg.critic_network.CriticNetwork(
            input_tensor_spec=(train_env.total_observation_spec(), train_env.total_action_spec()),
            observation_fc_layer_params=self.fc_layer_params_critic_obs,
            joint_fc_layer_params=self.fc_layer_params_critic_merged, name='CriticNetwork')

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

