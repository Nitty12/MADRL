from EVehicle import EVehicle
from PVGen import PVG
from WindGen import WG
from HeatPump import HeatPump
from BatteryStorage import BatStorage
from DSM import DSM
from SpotMarket import SpotMarket
from DSO import DSO
from Grid import Grid
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.environments import suite_gym
import os

# TODO why it is not reproducible even though np is seeded?
np.random.seed(0)
tf.random.set_seed(0)
'''
generation is -ve qty and load is +ve qty
'''


def agentsInit():
    path = os.getcwd()
    datapath = os.path.join(path, "data\RA_RD_Import_.csv")
    data = pd.read_csv(datapath, sep=';', comment='#', header=0, skiprows=0, error_bad_lines=False)
    data = data.loc[1:, ['Name', 'Location', 'Un', 'min. P', 'max. P']]
    data.columns = ['Name', 'Location', 'Un_kV', 'P_min_MW', 'P_max_MW']
    data.reset_index(inplace=True, drop=True)
    data[['Un_kV', 'P_min_MW', 'P_max_MW']] = data[['Un_kV', 'P_min_MW', 'P_max_MW']].apply(pd.to_numeric)
    PV = []
    wind = []
    homeStorage = []
    biomass = []
    for name in data['Name']:
        if name.endswith('_solar') or name.endswith('_nsPVErsatzeinsp'):
            PV.append(name)
        elif name.endswith('_wea'):
            wind.append(name)
        elif name.endswith('_nsHeimSpeicherErsatzeinsp'):
            homeStorage.append(name)
        elif name.endswith('_biomasse'):
            biomass.append(name)

    agentsDict = {}
    for name in PV[:1]:
        details = data.loc[data['Name'] == name]
        loc = details['Location'].values[0]
        voltage_level = details['Un_kV'].values[0]
        min_power = details['P_min_MW'].values[0]
        max_power = details['P_max_MW'].values[0]
        agentsDict[name] = PVG(id=name, location=loc, minPower=min_power, maxPower=max_power,
                               voltageLevel= voltage_level, marginalCost=0)
    return agentsDict

def gridInit():
    grid = Grid()
    return grid


def replayBufferInit(train_env):
    """replay buffer stores transition of all the agents"""
    replay_buffer_capacity = 100000
    policy_step_spec = policy_step.PolicyStep(action=train_env.action_spec(),
                                              state=(),
                                              info=())
    replay_buffer_data_spec = trajectory.from_transition(train_env.time_step_spec(),
                                                         policy_step_spec,
                                                         train_env.time_step_spec())
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=replay_buffer_data_spec,
                                                                   batch_size=train_env.batch_size,
                                                                   max_length=replay_buffer_capacity)
    return replay_buffer


def one_step(environment, policySteps):
    """iteration alternate between spot and flex states"""
    total_agents_action = []
    time_step = environment.current_time_step()
    for i, policyStep in enumerate(policySteps):
        individual_time_step = ts.get_individual_time_step(time_step, i)
        action = policyStep(individual_time_step)
        total_agents_action.append(action)
    next_time_step = environment.step(tuple(total_agents_action))
    return time_step, total_agents_action, next_time_step


def compute_avg_return(environment, policySteps, num_steps=10):
    total_return = None
    for step in range(num_steps):
        _, _, next_time_step = one_step(environment, policySteps)
        if step == 0:
            total_return = next_time_step.reward
        else:
            total_return += next_time_step.reward
    avg_return = total_return/num_steps
    return avg_return.numpy()[0]


def collect_step(environment, policySteps, buffer):
    time_step, total_agents_action, next_time_step = one_step(environment, policySteps)
    traj = trajectory.from_transition(time_step, total_agents_action, next_time_step, joint_action=True)
    buffer.add_batch(traj)
    # tf_metrics.AverageReturnMetric(traj)


def get_target_and_main_actions(experience):
    """get the actions from the target actor network and main actor network of all the agents"""
    total_agents_target_actions = []
    total_agents_main_actions = []
    time_steps, policy_steps, next_time_steps = (
        trajectory.experience_to_transitions(experience, squeeze_time_dim=True))
    for i, flexAgent in enumerate(agents):
        target_action, _ = flexAgent.NN.agent._target_actor_network(next_time_steps.observation[i],
                                                                     next_time_steps.step_type,
                                                                     training=False)
        main_action, _ = flexAgent.NN.agent._actor_network(time_steps.observation[i],
                                                       time_steps.step_type,
                                                       training=True)
        total_agents_target_actions.append(target_action)
        total_agents_main_actions.append(main_action)
    return time_steps, policy_steps, next_time_steps, \
           tuple(total_agents_target_actions), tuple(total_agents_main_actions)


# agentsDict = agentsInit()

dsm1 = DSM(id=3, maxPower = 2, marginalCost = 20)
sm = SpotMarket()
sm.addParticipants([dsm1])
grid = gridInit()
dso = DSO(grid)
dso.addflexAgents([dsm1])

# grid = gridInit()
# time = np.arange(24)
# congestion = grid.checkCongestion(time)
#
# sm = SpotMarket()
# sm.addParticipants([ba1, hp1, dsm1, pv1, wg1])
#
# dso = DSO(grid)
# dso.addflexAgents([ba1, hp1, dsm1, pv1, wg1])
#
# load the train Gym environment
env = suite_gym.load("gym_LocalFlexMarketEnv:LocalFlexMarketEnv-v0",
                     gym_kwargs={'SpotMarket': sm, 'DSO': dso, 'grid': grid})
# evaluation environment
eval_py_env = suite_gym.load("gym_LocalFlexMarketEnv:LocalFlexMarketEnv-v0",
                             gym_kwargs={'SpotMarket': sm, 'DSO': dso, 'grid': grid})
env.reset()

# convert to tf environment
train_env = tf_py_environment.TFPyEnvironment(env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

time_step = train_env.reset()

replay_buffer = replayBufferInit(train_env)
train_step_counter = tf.Variable(0, trainable=False)
batch_size = 8
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).prefetch(3)
iterator = iter(dataset)

# initialize the ddpg agents
agents = train_env.pyenv.envs[0].gym.agents
for index, agent in enumerate(agents):
    agent.NN.initialize(train_env, train_step_counter, index)

"""This is the data collection policy"""
collect_policySteps = [agent.NN.collect_policy.action for agent in agents]
"""This is the evaluation policy"""
eval_policySteps = [agent.NN.eval_policy.action for agent in agents]

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, eval_policySteps, num_steps=10)
returns = [avg_return]

# initialize trainer
for flexAgent in agents:
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    flexAgent.NN.agent.train = common.function(flexAgent.NN.agent.train)
    # Reset the train step
    flexAgent.NN.agent.train_step_counter.assign(0)

# Training the agents
num_iterations = 100
collect_steps_per_iteration = 2
log_interval = 20
eval_interval = 20
time_step = train_env.reset()

for num_iter in range(1, num_iterations+1):
    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
        collect_step(train_env, collect_policySteps, replay_buffer)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    time_steps, policy_steps, next_time_steps, total_agents_target_actions, total_agents_main_actions = \
        get_target_and_main_actions(experience)

    train_step_counter.assign_add(1)

    for i, flexAgent in enumerate(agents):
        train_loss = flexAgent.NN.agent.train(time_steps, policy_steps, next_time_steps,
                                              total_agents_target_actions,
                                              total_agents_main_actions,
                                              index=i).loss
        step = flexAgent.NN.agent.train_step_counter.numpy()
        if step % log_interval == 0:
            print('Agent ID = {0} step = {1}: loss = {2}'.format(flexAgent.id, step, train_loss))

    if train_step_counter % log_interval == 0:
        print("=====================================================================================")

    if num_iter % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, eval_policySteps, num_steps=2)
        print('step = {0}: Average Return = {1}'.format(num_iter, avg_return))
        print("=====================================================================================")
        returns.append(avg_return)

"""Plot the returns"""
steps = range(0, num_iterations + 1, eval_interval)
idx = 0
agent_returns = [ret[idx] for ret in returns]
plt.plot(steps, agent_returns)
plt.ylabel('Average Return - Agent {}'.format(idx))
plt.xlabel('Step')
plt.show()