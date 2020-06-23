import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.agents import DdpgAgent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.environments import wrappers
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.environments import suite_gym


class AgentNeuralNet:
    def __init__(self):
        # Hyperparameters
        self.num_iterations = 50000
        self.initial_collect_steps = 2000
        self.collect_steps_per_iteration = 1
        self.learning_rate = 1e-5
        self.log_interval = 500
        self.num_eval_episodes = 2
        self.eval_interval = 2000
        self.fc_layer_params_actor = (100,)
        self.fc_layer_params_critic_obs = (50,)
        self.fc_layer_params_critic_merged = (100,)
        self.discount_factor = 0.99

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
        # print(train_env.total_observation_spec())
        # print(train_env.total_action_spec())

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

    # for testing
    def setBidLimits(self, sLow, sHigh, fLow, fHigh, pLow, pHigh):
        self.lowSpotBidLimit = sLow
        self.highSpotBidLimit = sHigh
        self.lowFlexBidLimit = fLow
        self.highFlexBidLimit = fHigh
        self.lowPriceLimit = pLow
        self.highPriceLimit = pHigh

    # for testing
    def action(self, obs):
        spotBidMultiplier = np.random.uniform(self.lowSpotBidLimit, self.highSpotBidLimit, size=24)
        flexBidMultiplier = np.random.uniform(self.lowFlexBidLimit, self.highFlexBidLimit, size=24)
        flexBidPriceMultiplier = np.random.uniform(self.lowPriceLimit, self.highPriceLimit, size=24)

        return np.concatenate((spotBidMultiplier, flexBidMultiplier, flexBidPriceMultiplier))