import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tf_agents.agents import DdpgAgent
from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg.critic_network import CriticNetwork

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


env_name = 'Pendulum-v0'
env = suite_gym.load(env_name)
env.reset()

"""
In the Pendulum environment:
    Try to keep a frictionless pendulum standing up.
    observation space:
        cos(theta), sin(theta), theta dot
    reward is a scalar float value -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
    action is Joint effort or torque applied:
"""

print('Observation Spec:')
print(env.time_step_spec().observation)

print('Reward Spec:')
print(env.time_step_spec().reward)

print('Action Spec:')
print(env.action_spec())

time_step = env.reset()
print('Time step:')
print(time_step)

action = np.array((1.0,), dtype=np.float32)

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)


train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# Hyperparameters
num_iterations = 50000
initial_collect_steps = 2000
collect_steps_per_iteration = 1
replay_buffer_capacity = 100000
batch_size = 128
learning_rate = 1e-5
log_interval = 500
num_eval_episodes = 2
eval_interval = 2000
fc_layer_params_actor = (100,)
fc_layer_params_critic_obs = (50,)
fc_layer_params_critic_merged = (100,)
discount_factor = 0.99

ActorNetwork = ActorNetwork(input_tensor_spec=train_env.observation_spec(), output_tensor_spec=train_env.action_spec(),
                            fc_layer_params=fc_layer_params_actor,
                            dropout_layer_params=None, conv_layer_params=None,
                            activation_fn=tf.keras.activations.relu, name='ActorNetwork')

CriticNetwork = CriticNetwork(input_tensor_spec=(train_env.observation_spec(), train_env.action_spec()),
                              observation_fc_layer_params=fc_layer_params_critic_obs,
                              joint_fc_layer_params=fc_layer_params_critic_merged)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = DdpgAgent(time_step_spec=train_env.time_step_spec(), action_spec=train_env.action_spec(),
                  actor_network=ActorNetwork, critic_network=CriticNetwork,
                  actor_optimizer=optimizer, critic_optimizer=optimizer, ou_stddev=0.2, ou_damping=0.15,
                  target_update_tau=0.001, target_update_period=10, gamma=discount_factor,
                  train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=train_env.batch_size,
                                                               max_length=replay_buffer_capacity)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return


dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

replay_observer = [replay_buffer.add_batch]

train_metrics = [tf_metrics.AverageReturnMetric()]

driver = dynamic_step_driver.DynamicStepDriver(train_env, collect_policy, observers=replay_observer + train_metrics,
                                               num_steps=collect_steps_per_iteration)

print(compute_avg_return(eval_env, eval_policy, num_eval_episodes))

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

time_step = train_env.reset()
final_time_step, policy_state = driver.run(time_step)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
returns = [avg_return]

for i in range(1000):
    final_time_step, _ = driver.run(final_time_step, policy_state)

for i in range(num_iterations):
    final_time_step, _ = driver.run(final_time_step)

    experience, _ = next(iterator)
    train_loss = agent.train(experience=experience)
    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, eval_policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)

plt.plot(returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations *'+str(eval_interval))
plt.show()